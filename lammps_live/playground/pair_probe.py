"""ONE pair of particles, taken apart term by term -- the numbers the two-bead
tutorial draws between its beads.

The energy panels answer "how much energy is in each term", summed over a bead's
neighbours or over the whole box. That is the right question on a membrane and the
wrong one on a scene with exactly two particles in it, where the interesting
statement is smaller and sharper: at THIS separation and THIS pair of
orientations, each additive term of the force field is worth this much energy and
is pushing along the bond this hard. That is what this module computes, and what
the renderer draws as an annotated connector between the two beads.

WHY THE FORCES ARE FINITE-DIFFERENCED. Every force field here already owns one
honest, tested expression for its energy decomposition (`ForceField.energy_terms`,
cross-checked against LAMMPS' own potential energy by verify.py). Differentiating
it again by hand, per term, would be a second expression of the same physics with
no verifier behind it -- exactly the duplication the force-field module was written
to end. So the radial force in each term is measured off that one expression:

    F_r = -(U(r + h) - U(r - h)) / 2h,     with the two directors and rhat FIXED
    tau = -(U(n_i rotated +h) - U(n_i rotated -h)) / 2h,   about each world axis

Displacing along the bond is what makes the first exact rather than approximate in
spirit: rhat does not turn, so the orientational terms' angular arguments are
untouched and the difference isolates the term's radial derivative. The second is
the same trick in the other domain -- a torque IS minus the energy gradient with
respect to a rotation, so rotating the driven bead's director about an axis and
differencing gives the torque about that axis, term by term. Nine configurations
in all (the pair as it stands, two displaced, six rotated), evaluated in ONE
vectorized call, and a test (tests/test_pair_annotation.py) pins both sums against
the force and the torque LAMMPS actually applied.

WHY THE TORQUE EARNS ITS COLUMN. It is where the force field stops looking like a
pair potential. Two of the three terms twist the bead and the third cannot -- the
van der Waals term depends on nothing but r, so its torque is identically zero,
and seeing that zero sit there while the tilt row swings is the fastest way to
learn what "orientational" means. It is also the only column that responds to the
twist axis alone, with the bead standing still.

AND IT DISAGREES WITH THE RUNNING SIMULATION IN ONE TERM, which is worth knowing
before the discrepancy is discovered from the screen. MesoMem's C++ pair style
applies the SPLAY torque with the sign of +d(U_splay)/d(n_i.n_j) instead of minus
it, so as compiled that term twists neighbouring directors apart rather than into
alignment. A torque is minus the energy gradient, so what this module reports is
the force field's own definition and the pair style is the thing out of step: its
tilt torque has the correct sign, and its energy matches this expression to machine
precision (verify.py). The size of the disagreement is twice the splay torque --
about 4% of the total at the paper's coefficients, where k_splay = 1 against
k_tilt = 12. tests/test_pair_annotation.py pins it explicitly, and will fail the
day the sign is fixed, which is the right time to hear about it.

WHAT IT IS NOT. `F_r` is the component along the bond, not the whole force: the
tilt term also pushes sideways, which is the reason it exists. The full measured
reaction vector is already on screen -- the red arrow at the bead and the header
line -- so this is the one component that decomposition adds, and the renderer
labels it as such. The torque, by contrast, is reported whole, and reduced to one
number only for display: the component about the normal of the plane the bead is
driven in, which is the axis the twist control turns it about and the axis the
torque arc on the bead already depicts.

WHAT IT REPLACES: both energy panels. A playground that turns this on has a scene
that is one pair, where the pulled bead's panel and the whole-system panel are the
same three numbers as each other and as this -- so they are switched off with it
(see PlaygroundSystem.get_pair_annotation) and this is the only reading on screen.
That is also why there are no bars in here: a bar needs a fixed full-scale to be
read against, and what the pair wants said is the number, next to the separation
and the angle it belongs to, with the direction of each term's push beside it.
"""
import math
from dataclasses import dataclass

import numpy as np

from .state import FrameState, PairData

# Central-difference step for the radial derivative, in the force field's own
# length unit (sigma for MesoMem). Small enough that the 4-2 core's curvature is
# resolved -- it is the steepest thing any of these potentials does -- and large
# enough to stay far above the float64 cancellation floor of an O(1) energy.
FD_STEP = 1e-3


@dataclass(frozen=True)
class PairTerm:
    """One additive term of the force field, evaluated on one pair."""
    label: str
    energy: float          # U, in the force field's energy unit
    radial_force: float    # -dU/dr on `i`: negative pulls the pair together
    # The term's torque on `i`, as the world 3-vector it is (the axis the rotation
    # is about, right-hand rule), and that vector's component about the driven
    # particle's plane normal -- which is the number the display shows, and the
    # whole of it in any scene where the directors and the bond stay in that plane.
    # `twist` falls back to the magnitude where there is no plane to project onto.
    torque: tuple = (0.0, 0.0, 0.0)
    twist: float = 0.0
    # What this term is a function of, as short symbolic text ("r", "ni.nj, r"),
    # from the force field's `energy_terms_arguments`. Empty where the force field
    # declares none. It rides along with the term rather than being looked up by
    # the renderer because the renderer is handed an annotation, not a force
    # field -- and because a term and its arguments are one fact.
    argument: str = ""


@dataclass(frozen=True)
class PairAnnotation:
    """Everything drawn between the two beads. Indices are into the arrays the
    renderer already has (`get_positions_3d` order, which is id order)."""
    i: int
    j: int
    r: float
    # Angle between particle `i`'s director and the line to `j`, in degrees and on
    # the same 0..180 convention as the `pair_director_angle` observable -- so the
    # number in the annotation and the number in the HUD are the same number.
    angle_deg: float
    terms: tuple           # of PairTerm, in energy_terms_labels order
    # ((label, radius, term_index), ...) -- the force field's own landmark radii
    # (see ForceField.pair_landmarks), drawn as rings around the fixed partner.
    shells: tuple = ()
    # World unit normal of the plane the driven particle is confined to, or None.
    # The landmark shells are spheres, and what the driven particle can actually
    # cross is their intersection with ITS OWN plane -- a circle in that plane,
    # which is what the renderer draws (projected, so it appears as the ellipse the
    # camera sees that plane as). Without a plane there is no such circle and the
    # renderer falls back to the sphere's silhouette.
    plane_normal: tuple = None
    # The torque that fills a torque display at full scale (Control's
    # reaction_torque_max), so the little arc drawn beside each term's number and
    # the big arc drawn on the bead itself are calibrated by one number.
    torque_scale: float = 1.0

    @property
    def total_energy(self):
        return sum(t.energy for t in self.terms)

    @property
    def total_radial_force(self):
        return sum(t.radial_force for t in self.terms)

    @property
    def total_twist(self):
        return sum(t.twist for t in self.terms)

    @property
    def in_range(self):
        """Whether the pair is interacting at all -- i.e. whether there is
        anything but zeros to read. Outside the outermost shell every term is
        exactly zero, and saying so is more use than three zeros."""
        reach = max((radius for _, radius, _ in self.shells), default=0.0)
        return self.r < reach if reach > 0.0 else any(
            t.energy or t.radial_force for t in self.terms)


def _rotate(v, axis, angle):
    """`v` turned by `angle` about the unit `axis` (Rodrigues), exactly -- not to
    first order in the angle, which would shorten the director as it turned and
    leave a spurious energy difference behind in the terms that read its length."""
    c, s = math.cos(angle), math.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * float(axis @ v) * (1.0 - c)


def probe_pair(force_field, state, params, i=0, j=1, plane_normal=None,
               torque_scale=1.0):
    """Decompose the interaction between particles `i` and `j` of `state`.

    Returns a PairAnnotation, or None if the force field offers no decomposition
    or the state does not hold both particles (a scene mid-rebuild, or a remote
    playground that has not received its first frame).
    """
    labels = force_field.energy_terms_labels
    positions = np.asarray(state.positions, dtype=float)
    if not labels or len(positions) <= max(i, j):
        return None
    d = positions[i] - positions[j]
    if state.box is not None:
        d = state.box.minimum_image(d[None, :])[0]
    r = float(np.linalg.norm(d))
    if r < 1e-9:
        return None

    directors = state.directors
    if directors is not None:
        directors = np.asarray(directors, dtype=float)[[i, j]]
        ni, nj = directors[0], directors[1]

    # EVERY configuration the two derivatives need, as one batch. The pair is
    # rebuilt in a frame of its own -- `i` at the origin, `j` down the bond -- and
    # a batch is just K such pairs side by side in one state, particles 2k and
    # 2k+1 for configuration k. That makes the whole decomposition ONE call into
    # the force field's energy expression instead of nine, which is what keeps a
    # per-frame readout out of the frame budget.
    #
    #   0        the pair as it stands: the energies
    #   1, 2     displaced by -+FD_STEP along the bond: the radial force
    #   3..8     the driven director rotated -+FD_STEP about x, y, z: the torque
    #
    # The displaced pair is skipped inside FD_STEP of contact (a step through
    # r = 0 is not a separation), and the rotated ones where the particles carry
    # no director at all.
    configs = [(r, None)]
    if r > FD_STEP:
        configs += [(r - FD_STEP, None), (r + FD_STEP, None)]
    n_radial = len(configs)
    axes = np.eye(3)
    if directors is not None:
        for axis in axes:
            configs += [(r, _rotate(ni, axis, -FD_STEP)),
                        (r, _rotate(ni, axis, FD_STEP))]

    pos = np.zeros((2 * len(configs), 3))
    dirs = None if directors is None else np.zeros((2 * len(configs), 3))
    for k, (radius, turned) in enumerate(configs):
        pos[2 * k + 1] = -d * (radius / r)
        if dirs is not None:
            dirs[2 * k] = ni if turned is None else turned
            dirs[2 * k + 1] = nj
    a_idx = np.arange(0, 2 * len(configs), 2)
    pairs = PairData(a_idx, a_idx + 1, pos[a_idx + 1] * -1.0,
                     np.array([radius for radius, _ in configs]))
    bag = force_field.energy_terms(
        FrameState(positions=pos, directors=dirs), pairs, params) or {}

    def column(label):
        arr = bag.get(label)
        return (np.zeros(len(configs)) if arr is None
                else np.asarray(arr, dtype=float))

    # Zipped by POSITION against the labels, and short by design on a force field
    # that declares none: `energy_terms_arguments` is optional, and a force field
    # that extends another's terms (the rod's adhesion, the polymer's exclusion)
    # may declare fewer than it has labels. A missing one is simply blank.
    arguments = tuple(force_field.energy_terms_arguments)

    terms = []
    for t_index, label in enumerate(labels):
        u = column(label)
        force = 0.0 if n_radial < 3 else -(u[2] - u[1]) / (2.0 * FD_STEP)
        torque = np.zeros(3)
        if directors is not None:
            for k in range(3):
                torque[k] = -(u[n_radial + 2 * k + 1] - u[n_radial + 2 * k]) \
                    / (2.0 * FD_STEP)
        if plane_normal is not None:
            normal = np.asarray(plane_normal, dtype=float)
            twist = float(torque @ normal / max(np.linalg.norm(normal), 1e-12))
        else:
            twist = float(np.linalg.norm(torque))
        terms.append(PairTerm(label=label, energy=float(u[0]),
                              radial_force=float(force),
                              torque=tuple(float(c) for c in torque),
                              twist=twist,
                              argument=(arguments[t_index]
                                        if t_index < len(arguments) else "")))
    terms = tuple(terms)

    angle = float("nan")
    if directors is not None:
        # -d points from i to j, which is the convention `pair_director_angle`
        # reports on (it measures against p[1] - p[0]).
        cos = float(-d @ directors[0]) / max(r * np.linalg.norm(directors[0]), 1e-12)
        angle = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    return PairAnnotation(
        i=i, j=j, r=r, angle_deg=angle, terms=terms,
        shells=tuple(force_field.pair_landmarks(params)),
        plane_normal=(None if plane_normal is None
                      else tuple(float(c) for c in plane_normal)),
        torque_scale=float(torque_scale),
    )
