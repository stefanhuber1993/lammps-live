"""The MesoMem membrane force field (Sillano, Marrink & Idema 2026).

Each particle is a one-particle-thick patch of bilayer carrying an orientation
vector (director) n_i = the local membrane normal, and the potential is additive:

    U = U_iso(r) + [ U_tilt(n_i, n_j, rhat) + U_splay(n_i, n_j) ] w(r)

with a soft 4-2 repulsive core and a cosine-squared attraction (U_iso), a tilt
term penalizing directors tipping away from the local surface normal, and a splay
term penalizing neighbouring directors misaligning. Forces AND torques are
evaluated pairwise.

The forces come from the authors' actual C++ pair style, compiled as a runtime
LAMMPS plugin (see mesomem_ff/ and playground/plugin.py). `energy_terms` below is
the same expression in vectorized numpy, and exists for two reasons: it drives the
live additive-energy panels, and verify.py checks its sum against LAMMPS' own
potential energy -- which turns "did I implement my force field correctly?" into a
test. It is NOT in the force loop.

This one file replaces the parameter block, pair_coeff emission and energy
decomposition that were duplicated across mesomem_hex.py, mesomem_sheet.py and
mesomem_assembly.py -- the energy expression alone was written out five times.
"""
import math
import os

import numpy as np

from ..playground.forcefield import ForceField, register
from ..playground.params import Param, Tier
from ..playground.plugin import PluginSpec

_FF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesomem_ff")

# One library, two pair styles: the membrane's own, and the rigid-rod/point LJ
# the wrapping-rod playground pairs it with under `pair_style hybrid`. A hybrid
# style needs every sub-style present in the same LAMMPS instance, and there is
# nothing to gain from compiling the membrane twice, so both live here and the
# playgrounds that want only the membrane simply never name the second.
MESOMEM_PLUGIN = PluginSpec(
    directory=_FF_DIR,
    sources=("mesomemplugin.cpp", "pair_membrane_sillano_v2.cpp",
             "pair_rod_lj.cpp"),
    headers=("pair_membrane_sillano_v2.h", "pair_rod_lj.h", "lammpsplugin.h",
             "version.h"),
    lib_stem="mesomem",
)

# --- Paper "standard conditions" (Sec. II-III) --------------------------------
SIGMA = 1.0        # bead diameter / length unit
EPS = 1.0          # energy unit (LJ well depth)
C0 = 0.0           # spontaneous curvature (0 -> flat preferred)

# Term labels, shared by the energy panels and the verifier so a label can never
# drift between them.
# "van der Waals" rather than "isotropic": it is the term everyone already has a
# name for -- a hard core with an attractive tail -- and the panels and the
# two-bead connector both show the short form (everything before the bracket), so
# that name is what the reader is left with.
ISO = "van der Waals  (repel + attract)"
TILT = "tilt  (directors normal to bonds)"
SPLAY = "splay  (neighbour directors align)"


@register
class MesoMem(ForceField):
    """The real MesoMem pair style, with its coefficients exposed as live dials.

    Slider ranges bracket the regimes the preprint explores, with the paper's
    recommended value marked as each dial's `optimum`:
      - k_tilt: floppy membrane up through the stiff-planar regime (planar above
        ~10); optimum 12. This is the critical control parameter -- below ~10
        aggregates stay compact and isotropic, above it they flatten.
      - k_splay: around its soft default; optimum 1.0.
      - zeta: steepness of the cosine-squared attractive branch; optimum 5.
      - rc: isotropic cutoff. The paper needs rc >= 2.5 sigma to sustain
        aggregation; beyond that stability is largely insensitive -> optimum 2.5.
      - wc: orientational (tilt/splay) cutoff, effectively upper-bounded by rc.
        Below that bound it barely affects structure but strongly tunes stiffness
        (Sec. III D 4) -> optimum 2.0.
    Both cutoffs reach down to 0 so the interactions can be switched fully off.
    """

    name = "mesomem"
    units = "lj"
    dimension = 3
    atom_style = "hybrid sphere dipole"
    n_types = 1
    plugin = MESOMEM_PLUGIN
    has_directors = True
    energy_terms_labels = (ISO, TILT, SPLAY)
    # The three terms' arguments, and the whole point of showing them: the first
    # depends on the separation ALONE, which is why its torque is identically zero
    # and why "orientational" means the other two. `rhat` is the unit vector along
    # the bond and `ni`/`nj` are the two directors, written the way the docstring
    # of `energy_terms` below writes them.
    energy_terms_arguments = ("r", "ni.rhat, nj.rhat, r", "ni.nj, r")
    # The attraction term dominates and runs to O(1) per pair; a bead has ~6-12
    # neighbours, so a per-particle half-range of ~3 keeps the bars readable at
    # both 7 and 1500 beads.
    energy_scale_per_particle = 3.0

    params = (
        Param("k_tilt", 12.0, "k_tilt", 0.0, 50.0, optimum=12.0, fmt="{:.1f}",
              doc="tilt modulus -- above ~10 the membrane stays planar"),
        Param("k_splay", 1.0, "k_splay", 0.0, 3.0, optimum=1.0, fmt="{:.2f}",
              doc="splay modulus"),
        Param("zeta", 5.0, "zeta (attraction falloff, higher=shorter reach)",
              0.0, 12.0, optimum=5.0, fmt="{:.1f}",
              doc="steepness/width of the cosine-squared attractive branch"),
        Param("splay_symmetry", 0.0, "splay symmetry (0=signed, 1=|dot|)",
              0.0, 1.0, fmt="{:.2f}", advanced=True,
              doc="blends the splay term's signed director dot product toward "
                  "|dot|, so parallel and antiparallel neighbours are penalised "
                  "identically (the term then cares only about the shared axis)"),
        # rc moves the pair style's GLOBAL cutoff, and therefore the neighbour
        # list, so it re-declares `pair_style` before the coefficients. Declared
        # via the tier rather than an `if key == "rc"` branch in every system.
        Param("rc", 2.5, "rc (interaction cutoff)", 0.0, 3.0, optimum=2.5,
              fmt="{:.2f}", advanced=True, tier=Tier.HOT_RESTYLE,
              doc="isotropic interaction cutoff"),
        # The paper caps wc at rc ("effectively upper-bounded by rc"), so a wc
        # slider dragged past the current rc is clamped rather than feeding an
        # ill-posed wc > rc to the pair style. Declaring the clamp here means it
        # applies to the pair_coeff line AND the energy decomposition below --
        # the old code had a separate _effective_wc() helper in each of three
        # system modules and had to remember to call it in each place.
        Param("wc", 2.0, "wc (orientation cutoff)", 0.0, 3.0, optimum=2.0,
              fmt="{:.2f}", advanced=True,
              clamp=lambda v, vals: min(v, vals["rc"]),
              doc="orientational (tilt/splay) cutoff, capped at rc"),
    )

    def __init__(self, bead_diameter=1.0, mass=1.0):
        # Sphere diameter drives the moment of inertia LAMMPS uses for the
        # director's rotational dynamics: the paper's I = (2/5) m sigma^2 comes
        # out of a sphere of RADIUS sigma, i.e. diameter 2 sigma. The 7-bead
        # patch uses that; the larger sheets use diameter = sigma so overlapping
        # beads read as a continuous membrane.
        self.bead_diameter = bead_diameter
        self.mass = mass

    def setup_commands(self, params):
        return [
            f"mass 1 {self.mass}",
            f"set group all diameter {self.bead_diameter}",
        ]

    def pair_commands(self, params):
        return [f"pair_style mesomem {params['rc']}"] + self.coeff_commands(params)

    def coeff_commands(self, params):
        """The 11-argument mesomem pair_coeff.

        The trailing `splay_symmetry` (0..1) is a local addition to the authors'
        code: 0 gives the paper's original splay term on the signed director dot
        product ni.nj; 1 makes it use |ni.nj|. Intermediate values blend the two
        continuously -- see the pair style's compute().
        """
        return [
            f"pair_coeff 1 1 {SIGMA} {EPS} {params['k_tilt']} {params['k_splay']} "
            f"{params['rc']} {params['wc']} {params['zeta']} {C0} "
            f"{params['splay_symmetry']}"
        ]

    def interaction_cutoff(self, params):
        return float(params["rc"])

    def pair_landmarks(self, params):
        """The radii the potential's story is told against (see
        ForceField.pair_landmarks), inside out.

        sigma is where the 4-2 core takes over from the attractive branch, so it is
        where the isotropic term turns from a pull into a wall: the radial force
        passes through zero there and comes back out the other side. It is the
        force field's length unit and therefore fixed at 1.

        rc is where everything stops, exactly, and it is drawn because the pair
        connector already changes with it -- the bond goes dashed outside rc, and a
        ring at the radius where that happens is the same statement twice, which is
        the good kind.

        wc USED TO BE HERE AND IS NOT ANY MORE. It is the outer edge of the
        orientational weight w(r) = exp(r^2 / (rga^2 ((r/wc)^4 - 1))), and w does
        vanish there exactly -- but it vanishes with an essential singularity, so
        just inside it w is e^-40 and the tilt and splay energies read zero to
        every decimal the callout prints. A ring at 2.0 promised an audience that
        something starts at 2.0, and then the numbers sat at 0.00 until about 1.8.
        What is drawn instead is each term's MEASURED onset (see
        pair_probe._term_onsets), against the same threshold as the callout's own
        rounding, so a bead crossing the tilt ring is a bead making the tilt row
        light up.
        """
        return (("sigma", SIGMA, 0),
                ("rc", float(params["rc"]), 0))

    # ---- the Python reference expression ------------------------------------

    def energy_terms(self, state, pairs, params):
        """The three additive MesoMem energies, per pair, fully vectorized.

        Mirrors the pair style's compute() (paper Eqs. 2-6, c0 = 0):

          U_iso   4-2 soft core for r < sigma, then -eps cos(g)^(2 zeta) with
                  g = (pi/2)(r - sigma)/(rc - sigma) out to rc
          w(r)    orientational weight, exp(r^2 / (rga^2 ((r/wc)^4 - 1))) inside
                  wc and zero outside, with rga = wc/2
          U_tilt  (k_tilt/2)[(n_i.rhat)^2 + (n_j.rhat)^2] w(r)
          U_splay (k_splay/2)(n_i.n_j - 1)^2 w(r), with the dot product blended
                  toward |n_i.n_j| by splay_symmetry so the panel matches the
                  force actually being applied

        Returns per-pair arrays, so the caller sums them over everything (whole
        system) or over the pairs touching one particle (that bead's share) from
        a single evaluation.
        """
        if not len(pairs):
            z = np.zeros(0)
            return {ISO: z, TILT: z.copy(), SPLAY: z.copy()}

        rc = float(params["rc"])
        wc = float(params["wc"])
        zeta = float(params["zeta"])
        k_tilt = float(params["k_tilt"])
        k_splay = float(params["k_splay"])
        sym = float(params["splay_symmetry"])

        r = pairs.r
        iso = np.zeros_like(r)
        core = r < SIGMA
        if core.any():
            t2 = (SIGMA / r[core]) ** 2
            iso[core] = EPS * (t2 * t2 - 2.0 * t2)
        # The attractive branch only exists when rc > sigma. Dragging the rc
        # slider to or below sigma leaves nothing outside the core (the pair
        # list is already cut at rc), so `att` is empty and the division by
        # (rc - sigma) never happens.
        att = (~core) & (r < rc)
        if att.any() and rc > SIGMA:
            g = math.pi * 0.5 * (r[att] - SIGMA) / (rc - SIGMA)
            iso[att] = -EPS * np.cos(g) ** (2.0 * zeta)

        tilt = np.zeros_like(r)
        splay = np.zeros_like(r)
        dirs = state.directors
        if dirs is not None and wc > 0.0:
            # Orientational weight: smoothly zero at wc and, being multiplied
            # into both terms below, zero outside it -- so tilt/splay need no
            # separate masking.
            rga = 0.5 * wc
            w = np.zeros_like(r)
            with np.errstate(divide="ignore", invalid="ignore"):
                denom = (r / wc) ** 4 - 1.0
            inside = (r < wc) & (denom < -1e-14)
            if inside.any():
                w[inside] = np.exp((r[inside] ** 2) / (rga * rga * denom[inside]))
                rhat = pairs.d / r[:, None]
                ni, nj = dirs[pairs.a], dirs[pairs.b]
                nir = np.einsum("ij,ij->i", ni, rhat)
                njr = np.einsum("ij,ij->i", nj, rhat)
                ninj = np.einsum("ij,ij->i", ni, nj)
                ninj_eff = (1.0 - sym) * ninj + sym * np.abs(ninj)
                tilt = 0.5 * k_tilt * (nir * nir + njr * njr) * w
                splay = 0.5 * k_splay * (ninj_eff - 1.0) ** 2 * w

        return {ISO: iso, TILT: tilt, SPLAY: splay}
