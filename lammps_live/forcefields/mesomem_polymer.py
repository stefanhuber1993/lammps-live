"""MesoMem membrane plus a melt of ring polymers -- the vesicle-and-chromatin field.

A second species is added to the membrane of `mesomem.py`: bonded beads forming
closed chains. Nothing about the membrane changes; what is added is a topology and
one more pair interaction.

    1-1   `mesomem`, untouched -- the membrane among itself.
    1-2   purely repulsive LJ, cut at its minimum. The polymer does not STICK to
      2-2 the membrane and the chains do not stick to each other; the only thing
          either can do is take up space. That is the whole model, and it is why a
          vesicle that bulges is telling you something: nothing pulled it, so
          something inside pushed.

    bonds  FENE, the standard Kremer-Grest chain: a finitely-extensible spring
           that cannot be crossed through, which is what keeps a melt of rings
           genuinely entangled rather than a soup of chains passing through one
           another. The entanglement is the physics -- rings that could cross
           would collapse to a point.
    angles cosine, at `k_bend`. A live dial, and the interesting one: it is the
           chain's PERSISTENCE LENGTH, so it decides whether the melt inside the
           vesicle is a floppy tangle that ignores the wall or a semi-stiff
           network that presses on it.

The collaborator's original deck is `polymer/polymer.lmp` at the repo root, and
this reproduces its physics -- with two departures, both deliberate. Its bond and
angle constants are FIXED here (they are the Kremer-Grest reference values and
moving them means a different polymer model, not a different setting), and its two
separate Langevin baths are one, because the app thermostats a single group and
because the deck's own comment records that two of them double-free on the
cluster's Kokkos build.

WHERE THE ATOM STYLE COMES FROM. `hybrid sphere dipole angle` locally -- the
membrane's own style with the molecule/bond fields added. On the cluster the host
profile substitutes `dipole_sphere_angle`, which is the collaborator's purpose-
built style and already carries exactly those fields (see remote/hosts.py); that
it does is not a coincidence, it is the style this system was written for.
"""
import numpy as np

from ..playground.forcefield import register
from ..playground.params import Param, Tier
from .mesomem import MesoMem

# The repulsive branch of a 12-6 LJ: cut at its minimum, so only the core is left.
# Written out at full precision because it is the cutoff AND the contact distance,
# and rounding it puts a small step in the potential at contact.
WCA_CUT = 2.0 ** (1.0 / 6.0)
# Kremer-Grest: spring constant, maximum extension, and the LJ pair the FENE bond
# is defined against. Not parameters -- these four numbers ARE the standard bead-
# spring model, and a chain with other values is a different model rather than the
# same one tuned.
FENE_K = 30.0
FENE_R0 = 1.5
FENE_EPS = 1.0
FENE_SIGMA = 1.0

EXCLUDED = "excluded volume  (polymer)"


@register
class MesoMemPolymer(MesoMem):
    """The membrane force field with a bonded second species.

    Two dials on top of the membrane's own:

      - k_bend: the chains' bending modulus (LAMMPS `angle_style cosine`, whose
        energy is k(1 + cos theta), so k IS the persistence length in bond
        lengths). The collaborator's deck runs 2.0. At 0 the rings are ideal
        flexible chains and collapse into compact globules; turned up they swell,
        and past a few times the default a ring is stiff enough to press the
        vesicle out of round.
      - eps_poly: how hard the polymer's own core is, and how hard it pushes on
        the membrane. Purely repulsive whatever it is set to -- the cutoff sits at
        the LJ minimum -- so this is the strength of a contact, never an adhesion.
    """

    name = "mesomem_polymer"
    n_types = 2
    # The membrane's style plus `angle`, which is what brings the molecule ID and
    # the bond/angle arrays a chain needs.
    atom_style = "hybrid sphere dipole angle"
    n_bond_types = 1
    n_angle_types = 1
    # A ring bead has exactly 2 bonds and is the centre of exactly 1 angle, but it
    # is also an END of two more, and LAMMPS counts an angle against every atom in
    # it. The special-neighbour allowance is the 1-2, 1-3 and 1-4 exclusions the
    # FENE special_bonds setting implies: 6 for a chain, with margin. All three are
    # cheap (a few bytes an atom) and impossible to raise later, so they are set
    # comfortably rather than exactly.
    box_extras = ("extra/bond/per/atom 2", "extra/angle/per/atom 3",
                  "extra/special/per/atom 8")
    energy_terms_labels = MesoMem.energy_terms_labels + (EXCLUDED,)

    params = MesoMem.params + (
        Param("k_bend", 2.0, "k_bend (chain stiffness)", 0.0, 20.0, optimum=2.0,
              fmt="{:.2f}",
              doc="bending modulus of the polymer, in kT per bond -- and so its "
                  "persistence length in bond lengths. 0 is an ideal flexible "
                  "chain; the reference deck runs 2"),
        Param("eps_poly", 1.0, "eps_poly (polymer excluded volume)", 0.1, 4.0,
              optimum=1.0, fmt="{:.2f}", tier=Tier.HOT_RESTYLE,
              doc="strength of the polymer's purely repulsive core, against "
                  "itself and against the membrane. Never attractive: the pair "
                  "is cut at the LJ minimum"),
    )

    def __init__(self, bead_diameter=1.0, mass=1.0, polymer_mass=1.0):
        super().__init__(bead_diameter=bead_diameter, mass=mass)
        # A polymer bead weighs what a membrane bead weighs -- the reference
        # deck's choice, and the one that keeps the two species' thermal speeds
        # the same so neither is visibly more agitated than the other.
        self.polymer_mass = polymer_mass

    # ---- LAMMPS side --------------------------------------------------------

    def setup_commands(self, params):
        """Per TYPE, not on `all`: the membrane's diameter drives its rotational
        inertia and must not be applied to beads that do not rotate.

        The polymer's own diameter and mass are deliberately NOT here. Its atoms
        do not exist yet at this point in the deck -- they arrive with the
        scenario's molecule template, which is issued afterwards and carries them
        (see VesiclePolymer._write_template). The per-type `mass 2` below is the
        bookkeeping entry LAMMPS wants regardless.
        """
        return [
            f"mass 1 {self.mass}",
            f"set type 1 diameter {self.bead_diameter}",
            f"mass 2 {self.polymer_mass}",
        ]

    def pair_commands(self, params):
        return ([f"pair_style hybrid mesomem {params['rc']} lj/cut {WCA_CUT:.12f}"]
                + self.coeff_commands(params))

    def coeff_commands(self, params):
        eps = float(params["eps_poly"])
        # The inherited line is `pair_coeff 1 1 <args>`; under hybrid it needs the
        # sub-style name after the type pair.
        membrane = [c.replace("pair_coeff 1 1 ", "pair_coeff 1 1 mesomem ", 1)
                    for c in super().coeff_commands(params)]
        return membrane + [
            f"pair_coeff 1 2 lj/cut {eps} {FENE_SIGMA} {WCA_CUT:.12f}",
            f"pair_coeff 2 2 lj/cut {eps} {FENE_SIGMA} {WCA_CUT:.12f}",
        ]

    def bonded_commands(self, params):
        """The chain: FENE bonds, cosine angles, and the special-bond weights.

        `special_bonds fene` is the 0 1 1 setting the FENE bond assumes -- the
        pair interaction between two BONDED beads is switched off, because the
        bond potential already contains it. Getting this wrong does not error; it
        silently doubles the repulsion along every chain.
        """
        return [
            "bond_style fene",
            f"bond_coeff 1 {FENE_K} {FENE_R0} {FENE_EPS} {FENE_SIGMA}",
            "special_bonds fene",
            "angle_style cosine",
        ] + self.angle_commands(params)

    def angle_commands(self, params):
        return [f"angle_coeff 1 {float(params['k_bend'])}"]

    def live_commands(self, params, changed_name):
        """`k_bend` is an angle coefficient, not a pair one, so the base class's
        pair-coefficient re-issue would silently do nothing for it."""
        if changed_name == "k_bend":
            return self.angle_commands(params)
        return super().live_commands(params, changed_name)

    def bonded_bead_energies(self, positions, loops, params):
        """Each chain bead's BENDING energy: the `angle_style cosine` term
        k_bend (1 + cos theta) of the one angle centred on it, theta being the
        angle between its two bonds -- 0 for a straight run, k_bend at a right-
        angle corner, 2 k_bend folded back on itself.

        The whole angle's energy, not LAMMPS' per-atom third of it: the angle
        belongs to the bead at its vertex, and "how bent is the chain HERE" is
        the question the colour answers. Bond stretching is left out on purpose;
        FENE at these constants barely moves, and what the stiffness knob
        changes is bending.
        """
        k = float(params["k_bend"])
        p = np.asarray(positions, dtype=float)
        out = np.empty(len(p))
        start = 0
        for length in loops:
            q = p[start:start + length]
            u = np.roll(q, 1, axis=0) - q
            v = np.roll(q, -1, axis=0) - q
            cos = np.einsum("ij,ij->i", u, v) / np.maximum(
                np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-12)
            out[start:start + length] = k * (1.0 + np.clip(cos, -1.0, 1.0))
            start += length
        return out

    # ---- the Python reference expression ------------------------------------

    def energy_terms(self, state, pairs, params):
        """The membrane's three terms plus the polymer's excluded volume.

        Split by TYPE exactly as `pair_style hybrid` routes the forces: the
        membrane terms exist only between two membrane beads, and the repulsive
        core only where a polymer bead is involved. Zeroing the membrane terms on
        polymer pairs is not cosmetic -- evaluated there they would read a polymer
        bead's absent director as an orientation and report a tilt energy for it.

        The BONDED energy is deliberately absent. This decomposition is over the
        analysis PAIR LIST, and a bond is not a pair -- it is a fixed topology the
        client does not have (nothing about it travels over the wire on a remote
        playground). What the panels show is therefore the non-bonded energy of
        the system, which is the part the sliders move.
        """
        terms = super().energy_terms(state, pairs, params)
        if not len(pairs):
            terms[EXCLUDED] = np.zeros(0)
            return terms

        is_poly = self._polymer_mask(state)
        if is_poly is None:
            # A single-species state (the verifier's synthetic membrane patches,
            # or a frame that arrived without types): no polymer, no core.
            terms[EXCLUDED] = np.zeros(len(pairs))
            return terms

        poly_pair = is_poly[pairs.a] | is_poly[pairs.b]
        for label in MesoMem.energy_terms_labels:
            if label in terms:
                terms[label] = np.where(poly_pair, 0.0, terms[label])
        terms[EXCLUDED] = self._excluded_volume(pairs, params, poly_pair)
        return terms

    @staticmethod
    def _polymer_mask(state):
        types = state.types
        if types is None or not len(types):
            return None
        mask = np.asarray(types) == 2
        return mask if mask.any() else None

    def _excluded_volume(self, pairs, params, poly_pair):
        """Per-pair WCA: a 12-6 LJ shifted up by eps and cut at its minimum, so it
        is zero at contact and rises steeply inside it. Bonded neighbours are not
        excluded here as `special_bonds` excludes them in LAMMPS -- see the
        docstring above on what this decomposition is and is not."""
        out = np.zeros(len(pairs))
        eps = float(params["eps_poly"])
        inside = poly_pair & (pairs.r < WCA_CUT) & (pairs.r > 1e-9)
        if not inside.any():
            return out
        sr6 = (FENE_SIGMA / pairs.r[inside]) ** 6
        out[inside] = 4.0 * eps * (sr6 * sr6 - sr6) + eps
        return out
