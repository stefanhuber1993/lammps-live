"""Declared observables, and the scheduler that keeps them inside the frame
budget.

A playground names the observables it wants; they are plotted live and are the
output columns of a headless sweep. This replaces the previous approach of
formatting ad-hoc strings in a per-system `get_hud_lines`, which could not be
plotted, could not be logged, and had to be rewritten for each new system.

Cost control matters here -- this is Python running between LAMMPS steps at 60
fps. Three things keep it cheap:

  * one pair list per analysis frame, shared by every consumer (the old code
    built a KD-tree twice per frame in the patch system, and again for the RDF);
  * a declared cadence per observable, so a heavy whole-system quantity runs
    every Nth frame and returns its cached value in between;
  * observables that do not need the pair list say so, and are not allowed to
    trigger building one;
  * the ones that DO need it are aligned onto the same frames rather than spread
    out over them.

THAT LAST POINT USED TO BE THE OPPOSITE, and the change is worth recording. The
phases were originally staggered on the reasoning that several every-4-frames
observables landing together would spike that frame. With a SHARED pair list that
is exactly backwards: the expensive thing is not the observables (0.4 ms each at
10k beads) but the list they all read, and spreading three of them over three
different frames built it three times instead of once. Measured on the 10k remote
playground, coarsened to 22 neighbours per bead:

    build_pairs   29 ms      energy_terms  9.7 ms      all observables  0.8 ms

so the schedule was averaging 17.4 ms per frame -- more than a whole 60 fps frame
-- to produce 10.5 ms of actual work. Aligned, and with the two observables that
never touch the list no longer triggering it, the same schedule averages 8.7 ms.
The cost is a bigger peak on the frames where everything lands, which is the right
trade here because the analysis runs on the stepper's thread alongside the drawing
(see remote/client.py) and has a whole frame to hide in.
"""
from dataclasses import replace

import numpy as np

from .state import PairData, build_pairs, principal_normal, segment_distance

_REGISTRY = {}


def observable(name, label=None, unit="", every=4, needs_directors=False,
               needs_pairs=False):
    """Register a function(state, pairs, params) -> float as a named observable.

    `needs_pairs` is the expensive declaration: the pair list costs 29 ms at 10k
    beads, so an observable that only reads positions or directors must not be the
    reason one gets built. It defaults to False, which means a new observable that
    forgets to declare it gets an empty pair list rather than a slow one -- an
    obviously wrong answer instead of a quietly expensive right one.
    """
    def decorate(fn):
        _REGISTRY[name] = Observable(
            name=name, label=label or name, unit=unit, every=every,
            needs_directors=needs_directors, needs_pairs=needs_pairs, fn=fn,
        )
        return fn
    return decorate


class Observable:
    def __init__(self, name, label, unit, every, needs_directors, fn,
                 needs_pairs=False):
        self.name = name
        self.label = label
        self.unit = unit
        self.every = max(1, int(every))
        self.needs_directors = needs_directors
        self.needs_pairs = needs_pairs
        self.fn = fn

    def __call__(self, state, pairs, params):
        return self.fn(state, pairs, params)


def get(name):
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown observable {name!r}. Available: {known}") from None


def available():
    return dict(_REGISTRY)


# --- the bundled observables --------------------------------------------------

@observable("nematic_S", "nematic order S", every=4, needs_directors=True)
def _nematic_order(state, pairs, params):
    """Nematic order parameter S = (3<cos^2 theta> - 1)/2 of the directors about
    their own mean axis, from the largest eigenvalue of the ordering tensor
    Q = <3 n n - I>/2.

    S = 1 is perfectly aligned directors (a flat, well-ordered membrane), S = 0
    is isotropic (a disordered gas). This is the single number that says whether
    the self-assembly run has actually formed membranes, and it responds sharply
    to k_tilt crossing the compact-vs-planar transition -- the thing a slider
    session is trying to find by eye.
    """
    n = state.directors
    if n is None or len(n) < 2:
        return float("nan")
    q = 1.5 * (n[:, :, None] * n[:, None, :]).mean(axis=0) - 0.5 * np.eye(3)
    return float(np.linalg.eigvalsh(q)[-1])


@observable("thickness", "membrane thickness", every=4)
def _thickness(state, pairs, params):
    """RMS spread of the particles along their own best-fit plane normal.

    For a monolayer this reads out buckling/roughening: near zero when the sheet
    is flat, growing as it corrugates or breaks up.
    """
    p = state.positions
    if len(p) < 3:
        return float("nan")
    n = principal_normal(p)
    return float(np.sqrt(np.mean(((p - p.mean(axis=0)) @ n) ** 2)))


@observable("area_per_particle", "area per particle", every=8)
def _area_per_particle(state, pairs, params):
    """In-plane area divided by particle count.

    Uses the periodic cell's cross-section when the box is periodic in-plane
    (the meaningful definition for a tension-free sheet), and the projected
    bounding box otherwise.
    """
    p = state.positions
    if not len(p):
        return float("nan")
    box = state.box
    if box is not None and box.periodic[0] and box.periodic[1]:
        area = box.lengths[0] * box.lengths[1]
    else:
        span = p[:, :2].max(axis=0) - p[:, :2].min(axis=0)
        area = float(span[0] * span[1])
    return area / len(p)


@observable("coordination", "mean neighbours within rc", every=4, needs_pairs=True)
def _coordination(state, pairs, params):
    """Mean number of neighbours inside the interaction cutoff -- the simplest
    read on condensation: a dilute gas sits near zero, a packed membrane near the
    lattice's coordination number."""
    n = len(state.positions)
    if not n:
        return float("nan")
    # Divided by the dilution because a subsampled list drops each of a bead's
    # neighbours with probability (1 - dilution): the raw mean is then that
    # fraction of the real coordination, and correcting it here is what lets the
    # number mean the same thing whether or not the analysis had to subsample.
    return 2.0 * len(pairs) / n / max(pairs.dilution, 1e-9)


# --- the two-bead tutorial ----------------------------------------------------
# Both of these are statements about the FIRST TWO particles, which is meaningless
# on a membrane and is exactly the readout the bead-and-partner scene wants: the
# force field as a function of one separation and one angle, with the numbers on
# screen while the hand moves. See scenario.BeadAndPartner.

@observable("pair_separation", "separation", unit=" sigma", every=1)
def _pair_separation(state, pairs, params):
    """Centre-to-centre distance between the first two particles.

    Every one on the energy panel has a landmark on this scale -- sigma is where
    the core takes over, wc is where the orientational terms switch on, rc is
    where everything stops -- so this is the x-axis the panels are being read
    against, said out loud. Every frame (`every=1`): it is two subtractions, and it
    is what the hand is changing.
    """
    p = state.positions
    if len(p) < 2:
        return float("nan")
    d = p[1] - p[0]
    if state.box is not None:
        d = state.box.minimum_image(d[None, :])[0]
    return float(np.linalg.norm(d))


@observable("pair_director_angle", "director vs. bond", unit=" deg", every=1,
            needs_directors=True)
def _pair_director_angle(state, pairs, params):
    """Angle between the FIRST particle's director and the line joining the pair.

    The quantity the tilt term is built on: it penalises (n . rhat), so 90 degrees
    -- the director broadside to its neighbour -- is what a flat membrane looks
    like locally and what the term is holding, and driving this away from 90 is
    what makes the tilt bar on the energy panel move. Reported for the driven
    particle, since the partner's is fixed by construction.
    """
    p, d = state.positions, state.directors
    if d is None or len(p) < 2:
        return float("nan")
    r = p[1] - p[0]
    if state.box is not None:
        r = state.box.minimum_image(r[None, :])[0]
    norm = np.linalg.norm(r) * np.linalg.norm(d[0])
    if norm < 1e-12:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(float(r @ d[0]) / norm, -1.0, 1.0))))


@observable("mean_tilt_deg", "mean director tilt", unit=" deg", every=4,
            needs_directors=True)
def _mean_tilt(state, pairs, params):
    """Mean angle between each director and the cloud's best-fit plane normal.
    Zero for a perfectly flat, untilted membrane; this is what the tilt modulus
    is holding down."""
    d = state.directors
    if d is None or len(d) < 3:
        return float("nan")
    normal = principal_normal(state.positions)
    # Directors are bidirectional (both +n and -n are minima of the tilt term),
    # so fold onto [0, 90] via |cos|.
    c = np.clip(np.abs(d @ normal), 0.0, 1.0)
    return float(np.degrees(np.arccos(c)).mean())


# --- the rod (type 2) against the membrane (type 1) ---------------------------
# The three numbers that say how far a wrapping run has got. None of them needs
# the pair list: there is one rod and it is measured against every bead directly,
# which is a single O(N) numpy pass -- far cheaper than the tree the pair list
# would have to build (see `observable`'s needs_pairs).


def _rod_and_membrane(state):
    """(rod index, membrane mask), or None when the frame has no rod in it.

    Type 2 is the rod by the convention MesoMemRod fixes. A single-species frame
    (every other playground) has no types at all, and every rod observable is a
    no-op on it rather than an exception -- an observable that is asked for on a
    system it does not apply to should report nothing, not break the HUD.
    """
    types = state.types
    if types is None or not len(types):
        return None
    rods = np.flatnonzero(np.asarray(types) == 2)
    if len(rods) != 1:
        return None
    return int(rods[0]), np.asarray(types) == 1


@observable("rod_height", "rod height over membrane", every=4)
def _rod_height(state, pairs, params):
    """The rod's centre height above the membrane's mean plane, along that
    plane's own normal.

    The single number the wrapping story is told in: it falls as the rod is
    pulled into contact, and keeps falling -- past zero -- as the membrane closes
    over it, because "wrapped" means the mean surface has moved to the far side of
    the rod's centre.
    """
    found = _rod_and_membrane(state)
    if found is None:
        return float("nan")
    rod, membrane = found
    p = state.positions[membrane]
    if len(p) < 3:
        return float("nan")
    normal = principal_normal(p)
    return float((state.positions[rod] - p.mean(axis=0)) @ normal)


@observable("rod_contacts", "beads touching the rod", every=4,
            needs_directors=True)
def _rod_contacts(state, pairs, params):
    """Membrane beads within a contact shell of the rod's SURFACE.

    Measured to the rod's axis segment and compared against the contact distance
    sigma_eff = r_mem + r_rod, so it counts beads that are actually touching the
    body rather than ones that happen to be near its centre. This is what climbs
    as the membrane engulfs the rod, and it saturates when the wrap closes.
    """
    found = _rod_and_membrane(state)
    if found is None or not params.has("rod_length"):
        return float("nan")
    rod, membrane = found
    offsets = state.positions[membrane] - state.positions[rod]
    if state.box is not None:
        offsets = state.box.minimum_image(offsets)
    d = segment_distance(offsets, state.directors[rod],
                         float(params["rod_length"]))
    # r_mem = sigma/2 in the paper's reduced units, and a quarter of the contact
    # distance of slack past it -- comfortably inside the attractive well, so a
    # bead is counted when it is held rather than merely nearby.
    sigma_eff = 0.5 + float(params["rod_radius"])
    return float(np.count_nonzero(d < 1.25 * sigma_eff))


@observable("rod_tilt_deg", "rod tilt out of membrane", unit=" deg", every=4,
            needs_directors=True)
def _rod_tilt(state, pairs, params):
    """Angle between the rod's long axis and the membrane's mean plane.

    0 is lying flat along the surface (the orientation adhesion wants, and the
    one that gets wrapped); 90 is standing on end, pointing into the membrane.
    Folded onto [0, 90], because a rod has no head and no tail.
    """
    found = _rod_and_membrane(state)
    if found is None or state.directors is None:
        return float("nan")
    rod, membrane = found
    p = state.positions[membrane]
    if len(p) < 3:
        return float("nan")
    normal = principal_normal(p)
    c = np.clip(np.abs(state.directors[rod] @ normal), 0.0, 1.0)
    return float(90.0 - np.degrees(np.arccos(c)))


# --- the vesicle-with-polymer numbers -----------------------------------------
# What a closed envelope with something pressing on it from inside is doing, in
# three numbers: how big it is, how swollen the thing inside it is, and how much
# of that thing is actually against the wall. Type 1 is the membrane and type 2
# the polymer, the convention MesoMemPolymer fixes.


def _membrane_and_polymer(state):
    """(membrane mask, polymer mask), or None on a frame with no polymer in it.

    A single-species frame (every other playground) has no types at all, and each
    of these reports nothing on it rather than raising -- the same rule the rod's
    observables follow, for the same reason.
    """
    types = state.types
    if types is None or not len(types):
        return None
    types = np.asarray(types)
    polymer = types == 2
    if not polymer.any():
        return None
    return types == 1, polymer


@observable("vesicle_radius", "vesicle radius", every=4)
def _vesicle_radius(state, pairs, params):
    """Mean distance of the membrane beads from their own centre.

    The envelope's size, and so the number the polymer's push shows up in: a melt
    that swells against the wall inflates it, and a floppy chain that collapses
    lets it relax back. Its SPREAD would be the shape rather than the size, which
    is what `thickness` measures on a flat sheet -- here that number is dominated
    by the sphere's own curvature and says nothing.
    """
    found = _membrane_and_polymer(state)
    membrane = None if found is None else found[0]
    p = state.positions if membrane is None else state.positions[membrane]
    if len(p) < 3:
        return float("nan")
    return float(np.linalg.norm(p - p.mean(axis=0), axis=1).mean())


@observable("polymer_gyration", "polymer radius of gyration", every=4)
def _polymer_gyration(state, pairs, params):
    """Radius of gyration of the whole melt, about its own centre.

    Read against `vesicle_radius`: the ratio is how much of the lumen the polymer
    occupies. A uniformly filled sphere of radius R has Rg = sqrt(3/5) R ~ 0.775 R,
    so a melt sitting near three quarters of the envelope's radius is filling it,
    and one well below that has pulled away into the middle.
    """
    found = _membrane_and_polymer(state)
    if found is None:
        return float("nan")
    p = state.positions[found[1]]
    if len(p) < 2:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum((p - p.mean(axis=0)) ** 2, axis=1))))


# The contact shell for the two species, in sigma: the repulsive core's own reach
# (its 12-6 minimum at 2^(1/6)) with a quarter of a sigma of slack, so a pair is
# counted when it is actually pressing rather than merely nearby. Quoted here
# rather than imported from the force field, which is the direction this module
# deliberately does not depend in -- and it is the same 1.25x the rod's contact
# count uses, for the same reason.
_POLYMER_CONTACT_SHELL = 1.25 * 2.0 ** (1.0 / 6.0)


@observable("polymer_contact", "membrane contacts per 100 polymer beads",
            every=4, needs_pairs=True)
def _polymer_contact(state, pairs, params):
    """How hard the melt is leaning on the envelope.

    The direct measure of the only coupling in this model: nothing sticks, so
    every one of these pairs is a polymer bead being pushed back by the membrane.
    Zero while the melt is clear of the wall, and it climbs as the chains swell
    into contact and stay there.

    A per-bead RATE rather than a count, and that is not cosmetic. Above
    Analysis.MAX_PAIR_BEADS the pair list is built over a random subsample, and a
    raw count off it is short by the dilution SQUARED with no way to tell from the
    number itself; a mean per polymer bead is short by one factor of it, which is
    exactly the correction `coordination` already applies. Per hundred beads
    because per one it is a couple of per cent and reads as noise.
    """
    found = _membrane_and_polymer(state)
    if found is None or not len(pairs):
        return float("nan")
    membrane, polymer = found
    n_poly = int(np.count_nonzero(polymer))
    if not n_poly:
        return float("nan")
    a, b = pairs.a, pairs.b
    cross = (membrane[a] & polymer[b]) | (polymer[a] & membrane[b])
    touching = np.count_nonzero(cross & (pairs.r < _POLYMER_CONTACT_SHELL))
    return 100.0 * touching / n_poly / max(pairs.dilution, 1e-9)


# --- scheduling ---------------------------------------------------------------

def analysis_pairs(force_field, state, params):
    """The pair list every consumer of a force field's energy must use.

    One function, because there are two callers and they have to agree: the live
    Analysis below, and verify.py's cross-check against LAMMPS. It is the global
    cutoff's pairs plus whatever a long-ranged species names for itself (see
    ForceField.extended_pairs) -- and the verifier building only the first half of
    that reported the force field as WRONG when the code was right, which is
    exactly the kind of false alarm a verifier must not raise.
    """
    pairs = build_pairs(state.positions,
                        force_field.interaction_cutoff(params), state.box)
    extra = force_field.extended_pairs(state, pairs, params)
    if extra is None or not len(extra):
        return pairs
    return PairData(np.concatenate([pairs.a, extra.a]),
                    np.concatenate([pairs.b, extra.b]),
                    np.concatenate([pairs.d, extra.d]),
                    np.concatenate([pairs.r, extra.r]),
                    dilution=pairs.dilution)


class Analysis:
    """Runs the energy decomposition and the requested observables on a budget.

    Holds the shared pair list, the cached energy terms and the cached observable
    values. The runtime calls `update` once per frame; only the work that is due
    on that frame actually runs.
    """

    # The energy panels are the most expensive consumer (a full pass over every
    # pair) and the aggregate barely changes frame to frame.
    ENERGY_EVERY = 4

    # THE PAIR LIST IS WHAT AN ANALYSIS FRAME COSTS, and the cost is linear in
    # the bead count with a large constant -- measured at 50k beads: 14 ms of
    # KD-tree build, 66 ms of query_pairs over 313k pairs, ~15 ms of exact
    # minimum-image filtering, so ~95 ms at best and ~190 ms warm. That arrives
    # as one lump every ENERGY_EVERY frames, and no amount of threading hides a
    # lump six times the size of the frame it lands in: on the remote demo it is
    # the stall you can see a few times a second while the camera turns.
    #
    # So above this many particles the pair work runs on a uniform random
    # SUBSAMPLE, redrawn every analysis frame, and what it feeds is corrected for
    # the dilution (PairData.dilution). Nothing it feeds is worse off for that: a
    # subsample of fraction f keeps each particle with probability f and each
    # pair with probability f^2, so a per-particle mean over f and a pair sum
    # over f^2 are both unbiased -- the HUD numbers and the energy bars come back
    # with a per cent or two of sampling noise instead of a stall, and because
    # the sample is redrawn each time the noise averages away over a second
    # rather than sitting there as a fixed wrong answer.
    #
    # The budget is a particle count rather than a millisecond target because the
    # cost is knowable from it and a self-tuning budget would make the numbers on
    # screen depend on how busy the machine is. 6000 leaves every local
    # playground (<= 6000 beads) untouched and costs ~5 ms.
    MAX_PAIR_BEADS = 6000

    def __init__(self, force_field, names=(), energy_every=None, enabled=True):
        self.force_field = force_field
        self.observables = tuple(get(n) for n in names)
        # Switched off entirely for a simulation whose analysis runs somewhere
        # else: the remote demo's server integrates and the CLIENT measures, on
        # the frames it receives (see remote/client.py). Leaving this on there
        # would pay the per-bead pair-list cost twice, once on each machine, and
        # the panels would be built where nothing can draw them.
        self.enabled = bool(enabled)
        if energy_every is not None:
            self.ENERGY_EVERY = max(1, int(energy_every))
        self._frame = 0
        self._pairs = None
        self._energy = None          # {label: per-pair array}
        # The pair list the cached energy was computed WITH. It has to be kept
        # alongside: the energy runs on its own (slower) cadence, while the pair
        # list is rebuilt whenever any observable is due, so the live list can be a
        # different length than the cached energy arrays. Masking one with the
        # other is then an index error -- which is exactly what happened.
        self._energy_pairs = None
        self._values = {}            # observable name -> last value
        # Redrawn every analysis frame that has to subsample; seeded once so a
        # replayed run is reproducible.
        self._rng = np.random.default_rng(0)

    @property
    def pairs(self):
        return self._pairs

    @property
    def energy(self):
        return self._energy

    def values(self):
        return dict(self._values)

    def _due(self, every):
        return self._frame % every == 0

    def update(self, state, params, force=False, keep_index=None):
        """Advance one frame. Rebuilds the pair list only when something that
        needs it is due, so a frame where nothing is scheduled costs nothing.

        `keep_index` names a particle whose OWN neighbourhood is going to be read
        off the result (the pulled bead, for the single-particle energy panel).
        Sampling cannot serve that: one bead's f-fraction of its own neighbours is
        a handful of pairs, and no scale factor turns that into a panel worth
        looking at. So naming a particle switches the subsampling off for the
        whole frame -- the panels it exists for belong to the puller playgrounds,
        which run an order of magnitude below the budget anyway.
        """
        if not self.enabled:
            return
        self._frame += 1
        energy_due = force or self._due(self.ENERGY_EVERY)
        due_obs = [ob for ob in self.observables
                   if force or self._due(ob.every)]
        if not energy_due and not due_obs:
            return

        # The pair list is built only for the consumers that actually read one,
        # and -- above the budget -- only over a sample of the particles. Whatever
        # reads the pairs must read the same state they were built from, so the
        # sampled state travels with them; everything else keeps the full one.
        pair_state = state
        if energy_due or any(ob.needs_pairs for ob in due_obs):
            pair_state, dilution = self._pair_state(state, keep_index)
            self._pairs = analysis_pairs(self.force_field, pair_state, params)
            self._pairs.dilution = dilution
        elif self._pairs is None:
            self._pairs = PairData.empty()

        if energy_due:
            self._energy = self.force_field.energy_terms(pair_state, self._pairs,
                                                         params)
            self._energy_pairs = self._pairs
        for ob in due_obs:
            ob_state = pair_state if ob.needs_pairs else state
            if ob.needs_directors and ob_state.directors is None:
                continue
            self._values[ob.name] = ob(ob_state, self._pairs, params)

    def _pair_state(self, state, keep_index):
        """`state` itself, or a uniform random subsample of it, plus the fraction
        of the system that came back. See MAX_PAIR_BEADS."""
        n = len(state.positions)
        if n <= self.MAX_PAIR_BEADS or keep_index is not None:
            return state, 1.0
        # Without replacement: a repeated particle would sit on top of itself and
        # the pair at zero separation would be dropped by build_pairs anyway,
        # leaving the sample quietly smaller than it says it is.
        pick = self._rng.choice(n, self.MAX_PAIR_BEADS, replace=False)
        return replace(state,
                       positions=state.positions[pick],
                       directors=(None if state.directors is None
                                  else state.directors[pick]),
                       types=None if state.types is None else state.types[pick],
                       ids=None if state.ids is None else state.ids[pick]), \
            self.MAX_PAIR_BEADS / n

    def energy_panel(self, title, scale, index=None):
        """(title, [(label, value), ...], scale) for the renderer's signed-bar
        panel, or None.

        `index` selects one particle's share: the pairs touching it. Whole-system
        and single-particle panels therefore come from ONE evaluation of the
        energy expression, which is why per-pair arrays are the interface.
        """
        if self._energy is None or self._energy_pairs is None:
            return None
        dilution = self._energy_pairs.dilution
        if index is not None and dilution < 1.0:
            # A sampled frame knows nothing about a particular particle: `index`
            # is an index into the whole system and these pairs are numbered
            # within the sample. There is no panel to draw rather than a wrong
            # one -- and this cannot happen while the caller passes the same
            # particle to update() as keep_index, which is what switches the
            # sampling off (see update).
            return None
        # A pair survives a sample of fraction f with probability f^2, so the sum
        # over the sampled pairs is that fraction of the system's own sum. One
        # factor of f is the particles that were drawn, the other the partners
        # they kept.
        gain = 1.0 / (dilution * dilution)
        # Mask with the pair list the energy was computed with, NOT the live one.
        mask = None if index is None else self._energy_pairs.touching(index)
        terms = []
        for label in self.force_field.energy_terms_labels:
            arr = self._energy.get(label)
            if arr is None:
                continue
            total = float(arr.sum() if mask is None else arr[mask].sum())
            terms.append((label, gain * total))
        return (title, terms, scale) if terms else None

    def hud_lines(self):
        """The requested observables as short display strings, in declared
        order."""
        out = []
        for ob in self.observables:
            v = self._values.get(ob.name)
            if v is None:
                continue
            out.append(f"{ob.label} = {v:.3f}{ob.unit}")
        return out
