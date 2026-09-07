"""Scenarios: geometry, box, relaxation and per-frame housekeeping.

A Scenario answers "what am I simulating, and in what cell" -- independent of
which force field runs on it and of whether the user is playing with it or
watching it run. Its core contract is pure numpy:

    build(params, rng) -> ScenarioBuild(positions, directors, types, box, bonds)

so scenarios are unit-testable with no LAMMPS instance, and the particles reach
LAMMPS through a single `create_atoms` call instead of one command per particle
(the old sheet system issued 900 of them).

Housekeeping forces are also pure numpy: a scenario returns an (N, 3) force array
and the runtime applies it as a momentum kick. These are NOT force-field physics
-- they are the soft corrections that keep a small free membrane well-posed for
interactive play (centred, face-on, not drifting out of frame).

Relaxation has two hooks because the two shipped patterns genuinely differ:
`pre_control_settle` runs before the thermostat and any control fixes exist (the
sheet's barostat relaxation to zero lateral tension, which must not fight a
puller); `post_control_settle` runs with everything already in place (the patch's
short silent settle).
"""
import itertools
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .params import ParamSet, structural
from .state import (Box, hex_lattice_2d, hex_ring_2d, icosphere_faces,
                    icosphere_spacing, lattice_ring, principal_normal,
                    random_points_min_separation)

# The polymer's two-stop ramp, in display-space bytes (see
# VesiclePolymer.render_tints). Deliberately nothing like the membrane's own
# colours: those are a pale blue and a pale yellow, both of them light, so the
# chains are given a saturated cool-to-warm pair that reads as a different KIND of
# object rather than as more membrane in another shade -- and one that holds up on
# the light background these scenes are drawn on, where a pale ramp would wash out.
POLYMER_COLD = (26, 148, 140)      # teal
POLYMER_WARM = (206, 66, 148)      # magenta


@dataclass
class ScenarioBuild:
    """The initial configuration, as arrays."""
    positions: np.ndarray                 # (N, 3); empty if LAMMPS places them
    box: Box
    directors: np.ndarray = None          # (N, 3) or None
    types: np.ndarray = None              # (N,) int, defaults to all type 1
    bonds: tuple = ()                     # index pairs to draw as sticks
    # Per-particle render brightness multiplier (1.0 = normal), or None. Used to
    # spotlight a tracked cluster so it can be followed as it diffuses.
    brightness: np.ndarray = None
    # How many of `positions` the runtime uploads itself; the rest are created by
    # the scenario's own `create_commands`. None -> all of them, which is every
    # scenario but one.
    #
    # It exists for a scenario carrying BONDED particles. Those cannot arrive
    # through `create_atoms` -- it places points, and a polymer is points plus a
    # topology -- so they come in through a LAMMPS molecule template instead (see
    # VesiclePolymer.create_commands). But everything ELSE about the system still
    # wants to know they are there: the client of a remote playground reads the
    # composition off this build to know how many beads to expect and what colour
    # to paint each one, and it has no LAMMPS to ask. So the build describes the
    # WHOLE system and this says where the runtime's own uploading stops.
    n_direct: int = None

    def __post_init__(self):
        if self.types is None:
            self.types = np.ones(len(self.positions), dtype=int)

    @property
    def n_uploaded(self):
        """How many leading positions the runtime uploads with create_atoms."""
        return len(self.positions) if self.n_direct is None else int(self.n_direct)


class Scenario(ABC):
    """Base class for a scenario."""

    name = ""
    # Declared structural parameters (counts, spacings, box size, stiffnesses).
    # Structural by definition: changing any of them means rebuilding, so they
    # live in the playground file and never become sliders.
    params = ()
    # MD timestep and how much simulated time one rendered frame advances. Both
    # live here rather than on the force field because they are properties of the
    # scenario's size and dynamics: the 1500-particle assembly wants a coarser
    # step and a much bigger per-frame slice than a 7-particle patch being probed.
    timestep = 0.005
    sim_time_per_frame = 0.05
    langevin_damp = 0.5     # implicit-solvent relaxation time
    # Neighbour-list skin. A denser scenario wants a smaller one (less wasted
    # pair checking); a dilute gas of fast-diffusing particles wants enough that
    # the list does not need rebuilding every step.
    neighbor_skin = 1.0

    # Rendering hints, surfaced on the generated SystemSpec.
    director_arrows = True     # per-particle director spikes; clutter above ~50
    wrap_fade_fraction = 0.0   # periodic-seam crossfade, as a fraction of the side

    # Whether the cell keeps changing size while the simulation runs -- a barostat
    # that is still installed after setup. The runtime then re-reads the box every
    # frame instead of trusting the geometry it asked for, because the drawn
    # outline, the periodic seam handling and the minimum-imaging in the analysis
    # pair list all key off it. False for a scenario whose cell is frozen once
    # relaxed, which is every other one: it is six `extract_global` calls a frame,
    # cheap but pointless when the answer cannot have changed.
    cell_is_live = False

    # The thermostat this scenario wants. Langevin (implicit solvent) suits a
    # coarse-grained membrane in water; a crystal slab wants CSVR, which leaves the
    # dynamics deterministic. See thermostat.py.
    thermostat = None          # None -> Langevin, set in __init_subclass__ terms

    def get_thermostat(self):
        from .thermostat import Langevin
        return self.thermostat or Langevin()

    def thermostat_damp(self, params):
        """Relaxation time for the thermostat. A scenario that declares its own
        `thermostat_damp` parameter uses it; otherwise the Langevin time."""
        if params.has("thermostat_damp"):
            return params["thermostat_damp"]
        return self.langevin_damp

    # Constructor keywords that set an attribute rather than a parameter: they
    # describe the scenario's dynamics rather than its geometry, and nothing reads
    # them through the ParamSet.
    _ATTR_KWARGS = ("timestep", "sim_time_per_frame", "langevin_damp",
                    "neighbor_skin", "director_arrows", "wrap_fade_fraction")

    def __init__(self, **overrides):
        """Keyword arguments are structural-parameter overrides, so a playground
        file reads `HexPatch(n_rings=2)` instead of assembling a ParamSet."""
        for name in self._ATTR_KWARGS:
            if name in overrides:
                setattr(self, name, overrides.pop(name))
        self.defaults = dict(overrides)

    def new_params(self, overrides=None):
        """Declared defaults, then this scenario's construction overrides, then
        anything the playground passes -- in that precedence order."""
        merged = dict(self.defaults)
        merged.update(overrides or {})
        return ParamSet.build(self.params, merged)

    @abstractmethod
    def build(self, params, rng):
        """Return a ScenarioBuild. `rng` is a seeded numpy Generator, so a
        scenario's randomness is reproducible from the playground's seed."""

    def atom_creation_commands(self, params, seed):
        """LAMMPS-side particle creation, for scenarios that let LAMMPS place
        particles (e.g. `create_atoms random`, which does its own overlap
        rejection). None -> the runtime uploads build().positions instead."""
        return None

    def create_commands(self, params, build, seed):
        """Commands issued right after the atoms exist -- per-particle attributes
        the scenario (not the force field) owns, e.g. the initial directors."""
        return []

    def wall_commands(self, box):
        """Reflecting walls on every non-periodic face.

        A particle flung out during violent play, or one that thermally
        evaporates out of a monolayer, feels no pair force past the cutoff and
        would otherwise random-walk to the box edge and be lost. An elastic
        reflection injects no energy and keeps it close enough for the membrane's
        own attraction to recapture it -- the implicit-solvent confinement the
        model otherwise omits.
        """
        faces = []
        for axis, per in zip("xyz", box.periodic):
            if not per:
                faces += [f"{axis}lo EDGE", f"{axis}hi EDGE"]
        return ["fix walls all wall/reflect " + " ".join(faces)] if faces else []

    def group_commands(self, params, controlled_id):
        """Groups this scenario needs beyond the mode's own (a frozen floor, a
        mobile subset the thermostat acts on). Issued after the mode's groups, so
        `controlled` already exists."""
        return []

    def thermostat_group(self):
        """Group the thermostat acts on, or None to let the mode decide (which is
        normally "everything except the controlled particle")."""
        return None

    def integrator_commands(self, params):
        """Time integrators this scenario installs itself, when a single global
        one will not do -- a frozen floor plus a mobile crystal, say. Empty ->
        the force field's global integrator is used."""
        return []

    def outline(self, params):
        """(build, natoms) -- the cell and the composition, for a caller that does
        NOT need the positions.

        The caller is the client of a remote playground (see remote/client.py): it
        has no LAMMPS to ask, so it reads the box to frame its camera with, the count
        to size its buffers to, and the species to colour each bead by, off the
        scenario -- and it does that every time the app switches to that playground,
        which is every Tab through the list. Placing 50,000 particles to answer three
        questions about them is half a second of an interface not responding, for
        arrays that are then thrown away.

        The default answer IS a build, because for most scenarios the geometry is the
        cheap part and two constructions would be two things to keep in step. A
        scenario whose PLACEMENT is the expensive part overrides this; RandomFill
        does. An override may return an empty `positions`, so the count comes from
        the second element and never from `len(build.positions)`.
        """
        build = self.build(params, np.random.default_rng(0))
        return build, len(build.positions)

    def extra_setup_commands(self, params):
        """Anything else the deck needs: LAMMPS variables read per frame,
        comm_modify tweaks, native computes."""
        return []

    def make_rdf(self, params, lmp, box):
        """Override the runtime's RDF choice. None -> let it pick from the box's
        periodicity."""
        return None

    def frame_commands(self, params, lmp):
        """Commands to issue once per frame, when they depend on measured state
        and so cannot be a fix (e.g. a drag proportional to this frame's centre-of-
        mass velocity). `lmp` is the live instance, for reading variables."""
        return []

    def pre_control_settle(self, params, seed):
        """Relaxation run BEFORE the thermostat and control fixes are defined.
        Owns its own bath/barostat and its own `run`; anything it defines must be
        listed in settle_cleanup_commands."""
        return []

    def settle_cleanup_commands(self):
        return []

    def post_control_settle(self, params):
        """Relaxation run with the thermostat and control fixes already in
        place."""
        return []

    def housekeeping(self, positions, params, controlled=None, box=None):
        """Per-frame soft correction forces (N, 3), or None.

        `controlled` is the index of the interactively-controlled particle (or
        None), which must be excluded: its position IS the user's input, and a
        correction force would fight it. `box` is the live cell, which a periodic
        scenario needs to know where "the middle" is.
        """
        return None

    def director_housekeeping(self, positions, directors, params, controlled=None,
                              box=None):
        """Per-frame soft correction ANGULAR velocities (N, 3) on the particles'
        directors, or None.

        The counterpart of `housekeeping` for orientation. It exists separately
        because in a periodic cell it is the only one of the two that can steer
        which way a structure faces: translating everything is fine on a torus,
        rotating everything is not.
        """
        return None

    def render_tints(self, params):
        """Static per-particle colour, (N, 4) or None (the default).

        Display-space r, g, b in 0..255 and a MIX in 0..1 saying how much of it
        to paint over the director banding -- see MDSystem3D.get_bead_tints, which
        is where it ends up. A scenario holding two species that should not look
        alike is what this is for; one holding a single membrane says nothing and
        lets the banding stand.
        """
        return None

    def camera(self, box):
        """dict(eye=, target=, up=, fov_deg=) for the 3D scene. The runtime then
        zooms to fit_points, so this only sets a good angle and rough distance."""
        span = max(box.lengths)
        return dict(eye=(0.0, -0.9 * span, 0.6 * span), target=(0.0, 0.0, 0.0),
                    up=(0.0, 0.0, 1.0), fov_deg=34.0)

    def fit_points(self, params, box):
        """World points the camera should frame. The box corners keep the drawn
        box outline in view at any aspect ratio."""
        return box.corners()


# --- housekeeping building blocks ---------------------------------------------
# Shared by the patch and the sheet, which independently derived the same
# "rotate the smallest principal axis to +z" correction.

def _ramp(x, lo, hi):
    """Smoothstep gate: 0 below lo, 1 above hi, no kink in between. Every
    whole-system correction here is gated by one, so it fades in with the
    structure it is correcting instead of switching on."""
    t = min(max((x - lo) / max(hi - lo, 1e-9), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def align_normal_rate(points, strength):
    """Rigid-rotation rate turning the cloud's best-fit plane normal toward +z,
    as omega = strength * (n x z). A flat cloud gets no rotation; a tilted one is
    turned back flat, with magnitude ~sin(tilt)."""
    n = principal_normal(points)
    return strength * np.cross(n, np.array([0.0, 0.0, 1.0]))


def align_normal_forces(points, strength):
    """Per-point forces producing a net normal-up alignment torque and ZERO net
    force, via F_i = a x r_i' with a = Iang^-1 T about the centroid. Used where
    the cluster is small enough that a genuine torque (rather than a rotation
    rate) is the right correction."""
    com = points.mean(axis=0)
    q = points - com
    torque = align_normal_rate(points, strength)
    # Inertia-like tensor Iang = sum(|r'|^2 I - r' r'^T).
    iang = np.eye(3) * np.sum(q * q) - q.T @ q
    a = np.linalg.solve(iang + 1e-6 * np.eye(3), torque)
    return np.cross(a[None, :], q)


def _axis_permutations():
    """The 48 signed permutation matrices -- every map that takes the cubic
    lattice to itself and the origin to the origin. Cached on first use; it is a
    fixed table of 48 3x3 matrices."""
    global _AXIS_PERMUTATIONS
    if _AXIS_PERMUTATIONS is None:
        mats = []
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product((1.0, -1.0), repeat=3):
                m = np.zeros((3, 3))
                for row, col in enumerate(perm):
                    m[row, col] = signs[row]
                mats.append(m)
        _AXIS_PERMUTATIONS = np.array(mats)
    return _AXIS_PERMUTATIONS


_AXIS_PERMUTATIONS = None


def _exclude(n, controlled):
    mask = np.ones(n, dtype=bool)
    if controlled is not None:
        mask[controlled] = False
    return np.nonzero(mask)[0]


# --- concrete scenarios -------------------------------------------------------

class HexPatch(Scenario):
    """A small hexagonal patch of particles in the xy-plane: a centre site ringed
    by closed hexagonal shells, directors along +z.

    The smallest geometry that already shows tilt/splay physics -- pull the middle
    particle out of plane and its neighbours' directors splay to follow. The
    surrounding rings are held in place by soft forces (not a geometric
    imposition), standing in for continuation into a larger membrane, so they
    still move, jitter and respond dynamically.
    """

    name = "hex_patch"

    params = (
        structural("n_rings", 1, "closed hexagonal shells around the centre site"),
        structural("a", 1.0, "nearest-neighbour spacing"),
        structural("box", 6.0, "cubic container side"),
        structural("settle_steps", 300, "silent relaxation before control begins"),
        # Strong, but still forces -- the patch can dome, tilt and recover
        # realistically. The homing term is a safety net so no single outer
        # particle can be flung out of the box during violent play.
        structural("k_center", 7.0, "centre-of-mass centering stiffness"),
        structural("k_align", 7.0, "normal-up alignment torque strength"),
        structural("k_home", 0.5, "per-particle homing stiffness toward the origin"),
        # What the camera looks at and frames -- a rectangle in the control
        # plane, centred a little above the patch (see camera / fit_points).
        # Smaller half-extents = closer in.
        #
        # `view_center_x` slides that rectangle -- and the eye with it, so the
        # scene is still viewed head-on -- along the control plane's horizontal
        # axis. 0 is the patch's own centre, which is right for anything built
        # symmetrically about the origin; a scene whose subject sits off-centre
        # (see BeadAndPartner, whose two beads span 0 .. partner_x) uses it to
        # put that subject in the middle of the frame instead of the patch.
        structural("view_center_x", 0.0, "camera framing: centre, in x"),
        structural("view_center_z", 0.7, "camera framing: centre height, in z"),
        structural("view_half_width", 2.0, "camera framing: half-width, in x"),
        structural("view_half_height", 1.6, "camera framing: half-height, in z"),
    )

    def build(self, params, rng):
        pts2d = hex_ring_2d(int(params["n_rings"]), params["a"])
        pos = np.column_stack([pts2d, np.zeros(len(pts2d))])
        dirs = np.tile([0.0, 0.0, 1.0], (len(pos), 1))
        # Spokes from the centre plus the closed first ring. hex_ring_2d orders
        # each shell by angle, so the ring closes correctly.
        n1 = min(6, len(pos) - 1)
        bonds = [(0, k) for k in range(1, n1 + 1)]
        bonds += [(k, k % n1 + 1) for k in range(1, n1 + 1)]
        return ScenarioBuild(positions=pos, directors=dirs,
                             box=Box.cube(params["box"]), bonds=tuple(bonds))

    def create_commands(self, params, build, seed):
        return ["set group all dipole 0.0 0.0 1.0"]

    def post_control_settle(self, params):
        return [f"run {int(params['settle_steps'])}"]

    def housekeeping(self, positions, params, controlled=None, box=None):
        """Centring + normal-up alignment + homing on every particle except the
        controlled one, as per-frame momentum kicks the Langevin bath then damps
        (so the patch settles flat and centred instead of oscillating)."""
        sel = _exclude(len(positions), controlled)
        if len(sel) < 3:
            return None
        p = positions[sel]
        f = np.zeros((len(positions), 3))
        # Same force on each -> moves the centre of mass without distorting shape.
        f[sel] = -params["k_center"] * p.mean(axis=0)
        f[sel] += align_normal_forces(p, params["k_align"])
        f[sel] -= params["k_home"] * p
        return f

    def camera(self, box):
        """Aimed a little ABOVE the patch, not at it.

        Everything interesting here happens upward: the controlled particle is
        pulled out of the plane and the ring domes after it. Aiming at the patch
        itself spends half the frame on the empty space below it, so the target
        is lifted by `view_center_z` and the vertical framing budget goes where
        the motion is."""
        span = max(box.lengths)
        params = self.new_params()
        # Eye and target share `view_center_x`, so sliding the frame sideways
        # pans the camera rather than swinging it: the view axis stays parallel
        # to -y, and two beads either side of that centre stay at equal depth
        # and equal apparent size. Aiming an eye fixed at x = 0 at an off-centre
        # target would foreshorten them differently, which on a two-bead scene
        # reads as one bead being bigger than the other.
        cx = params["view_center_x"]
        return dict(eye=(cx, -0.9 * span, 0.6 * span),
                    target=(cx, 0.0, params["view_center_z"]),
                    up=(0.0, 0.0, 1.0), fov_deg=34.0)

    def fit_points(self, params, box):
        """Frame a rectangle in the CONTROL PLANE, not the container.

        Framing the box corners -- the obvious thing, and what this did -- puts
        the camera far too far back: the box's near face sits several units
        closer to the eye than the patch does, subtends a much larger angle, and
        the fit backs off until THAT fits. On the default 6-sigma box the seven
        beads came out filling about a fifth of the frame width, adrift in an
        empty cell.

        Framing a rectangle at y = 0 instead -- the plane the beads and the
        control net live in -- puts the patch where it belongs, at a bit over
        half the frame width. What that costs is the outer corners of the net and
        the box outline, which now fall outside the view; both are context for a
        scene whose subject is seven beads, and the beads win.
        """
        w, h = params["view_half_width"], params["view_half_height"]
        x, z = params["view_center_x"], params["view_center_z"]
        return np.array([(x - w, 0.0, z - h), (x + w, 0.0, z - h),
                         (x - w, 0.0, z + h), (x + w, 0.0, z + h)])


class BeadAndPartner(HexPatch):
    """One driven bead and one bead nailed down, so the pair potential can be felt
    one interaction at a time.

    The tutorial scene of `mesomem_bead` is a single bead in an empty box: it
    teaches the CONTROLS and, deliberately, no physics -- one bead has no
    neighbours, so every pair term in the force field is silent. This adds the
    smallest thing that makes them speak: a SECOND bead, held still, that the first
    can be driven toward, past, and around.

    WHY THE PARTNER IS FROZEN RATHER THAN HEAVY. What is being read here is the
    force field as a function of one separation and two orientations, and any motion
    of the partner turns that into a two-body problem whose answer moves while you
    are reading it. So it is genuinely fixed: it is left OUT of the integrator group
    (see `integrator_commands`), which pins its position and its director exactly
    and for free -- no restraining spring to fight, and no stiff spring's own
    timescale added to the deck. A `fix setforce` would hold the position and leave
    the director spinning, which is half the point missed: the tilt term is about
    the angle between a director and the line joining the pair, so the partner's
    director has to stay where it was put.

    WHAT IT IS FOR, in the order the terms come in as the beads approach:

      * far apart, nothing. The energy panels sit at zero and the net is just a net.
      * inside rc (2.5 sigma at the paper's values) the isotropic attraction turns
        on and the driven bead is PULLED -- the first force in this app that the
        hand did not apply.
      * inside wc (2.0) the tilt and splay terms join, and they depend on the
        angle, so twisting the driven bead's director now changes the force on it.
        This is the one scene where that is a clean statement rather than an average
        over a dozen neighbours.
      * inside sigma the 4-2 core repels, hard. The leash cannot push through it,
        which is worth feeling once.

    The partner sits on the control plane at the RIGHT-HAND EDGE of the net, at the
    height the driven bead starts at -- so the demo is "push right until something
    happens", the net's own edge marks how far there is to go, and the whole
    approach happens along one joystick axis.
    """

    name = "bead_and_partner"

    params = HexPatch.params + (
        # In the control plane, which for these playgrounds is the world xz-plane.
        # The default puts it at the net's right edge; the playground is what has to
        # keep the two agreeing (see mesomem_bead.py, which derives both from one
        # number).
        structural("partner_x", 3.0, "the fixed partner's x, in sigma"),
        structural("partner_z", 0.0, "the fixed partner's z, in sigma"),
        # Along +z, like the membrane's own beads: the driven bead then starts
        # side-on to it, which is the configuration the tilt term has most to say
        # about. Turn it to put the partner's director anywhere and read the pair
        # off against a fixed orientation.
        structural("partner_director", (0.0, 0.0, 1.0),
                   "the fixed partner's director (it never moves, so this is the "
                   "orientation the whole demo is read against)"),
    )

    def build(self, params, rng):
        """The patch, plus the partner LAST.

        Last, because the driven particle is chosen by `Control(atom="first")` and
        that has to be the one the hand is on. It also keeps the patch's own bond
        list -- which indexes into the patch's positions -- correct without
        rewriting it.
        """
        patch = super().build(params, rng)
        partner = np.array([[float(params["partner_x"]), 0.0,
                             float(params["partner_z"])]])
        director = np.asarray(params["partner_director"], dtype=float)
        director = director / max(np.linalg.norm(director), 1e-12)
        positions = np.vstack([patch.positions, partner])
        directors = np.vstack([patch.directors, director[None, :]])
        return ScenarioBuild(positions=positions, directors=directors,
                             box=patch.box, bonds=patch.bonds)

    def create_commands(self, params, build, seed):
        """Everything up, then the partner's own orientation over the top of it.

        By ID rather than by type: both beads are the same species, which is the
        whole point -- the interaction being felt is the membrane's own, not a
        second force field bolted on for the demo.
        """
        director = np.asarray(params["partner_director"], dtype=float)
        director = director / max(np.linalg.norm(director), 1e-12)
        pid = len(build.positions)
        return [
            "set group all dipole 0.0 0.0 1.0",
            f"set atom {pid} dipole {director[0]} {director[1]} {director[2]}",
        ]

    def group_commands(self, params, controlled_id):
        """`anchor` is the partner; `mobile` is everything else.

        The partner is the LAST id, which `build` guarantees. Named groups rather
        than a bare id on the fix so the integrator line below reads as what it is.
        """
        pid = self.n_particles(params)
        return [f"group anchor id {pid}",
                "group mobile subtract all anchor"]

    def integrator_commands(self, params):
        """The integrator, on `mobile` only -- which IS how the partner is fixed.

        An atom in no integrator group has no equation of motion: forces and
        torques still accumulate on it (and are still felt by everything else, which
        is what makes it a partner rather than a wall) but nothing integrates them,
        so its position and its director are exactly where they were put. That is
        cheaper and more honest than any restraint, which would move a little, ring
        at its own frequency, and add a stiffness to the deck that the physics being
        demonstrated does not contain.

        `nve/sphere update dipole` rather than plain `nve` because this scenario
        exists for MesoMem, whose beads carry directors that have to be integrated.
        Stated here rather than taken from the force field because installing ANY
        scenario integrator suppresses the force field's global one (see
        PlaygroundSystem._issue_setup), so this line has to be the whole answer.
        """
        return ["fix integrate mobile nve/sphere update dipole"]

    def n_particles(self, params):
        """How many particles this scenario builds -- the patch's sites plus one.

        Its own method because `group_commands` needs the partner's id before any
        build is in hand.
        """
        return len(hex_ring_2d(int(params["n_rings"]), params["a"])) + 1

    def thermostat_group(self):
        """The bath is `mobile`, not the mode's `bath`.

        GameMode's bath is "everything except the driven particle", which here
        includes the partner -- and a Langevin fix on an atom nothing integrates is
        merely pointless, but it also makes the panel's measured temperature an
        average over a particle that can never have any. `mobile` minus the driven
        one is what is left, and on this scenario that is nothing at all, which is
        the honest answer: there is no bath here, and the temperature slider says so
        by doing nothing (see mesomem_bead.py).
        """
        return "mobile"

    def housekeeping(self, positions, params, controlled=None, box=None):
        """None of the patch's soft corrections.

        They act on "everything except the driven particle", which here is the
        partner -- and the partner is fixed, so a centring force on it is a force on
        nothing. Returning None also keeps the per-frame gather out of the loop
        entirely (see PlaygroundSystem._apply_housekeeping).
        """
        return None


class HexSheet(Scenario):
    """A periodic hexagonal monolayer -- the paper's planar-stability test.

    Particles on a hexagonal lattice at spacing a, periodic in-plane, relaxed
    under Langevin dynamics with a barostat driving the lateral pressure to zero
    so the sheet equilibrates tension-free. The periodic cell means no artificial
    tether is needed -- the sheet holds itself flat.

    THE BAROSTAT KEEPS RUNNING, and that is not housekeeping -- it is what makes
    the planar membrane physical at all. Freezing the cell after the settle (which
    is what this scenario used to do) leaves the sheet at a FIXED PROJECTED AREA,
    and that area is only ever right for one temperature. The dial here spans
    0 to 0.5, and the tension-free cell measured on a 30x30 sheet is:

        T = 0.001   Lx = 0.977 of the built size
        T = 0.2     Lx = 1.103          -- +22% in AREA

    so running the built cell warm compresses the membrane by about a fifth. A
    laterally compressed membrane does not merely sit there being slightly wrong.
    The undulation free energy per mode is

        F_q = 1/2 A (kappa q^4 + gamma q^2) |h_q|^2

    and a compressed sheet has gamma < 0, so every mode below q^2 = |gamma|/kappa
    has NEGATIVE stiffness and grows instead of fluctuating, until the excess-area
    nonlinearity catches up. What is left is a static ripple at a selected
    wavelength that neither decays nor travels -- an unphysical standing wave, and
    the artefact this scenario was reported for. Measured, frozen cell at T = 0.2:
    the lateral pressure locks at +0.17 and stays there over 150 tau while the rms
    out-of-plane displacement climbs 0.23 -> 0.35 sigma and the amplitude piles
    into the lowest few modes. With the barostat left in, P -> 0.00 +/- 0.01 and
    the rms holds at 0.20.

    ABOVE `melt_temp` THE CELL INFLATES WITHOUT BOUND, and that is the honest
    answer rather than a failure of the barostat. A melted sheet has no cohesion
    left to hold at zero lateral tension, so it simply keeps expanding: at
    T = 0.35 the nematic order falls to 0.02 and the cell is still growing after
    60 tau. The frozen cell hid this by holding the box still -- but it melted
    just the same (S = 0.04 over the same run), so what the freezing bought was a
    tidier picture of an equally destroyed membrane.

    (The reference deck does the same thing with `fix nph/sphere x 0 0 Pdamp
    y 0 0 Pdamp couple xy ... update dipole`. This uses `press/berendsen` instead,
    for the reason RodOnSheet gives: it rides on top of the force field's own
    `nve/sphere update dipole` rather than replacing it, and it has no barostat
    mass of its own to ring against the user's hand.)
    """

    name = "hex_sheet"
    sim_time_per_frame = 0.1
    director_arrows = False     # hundreds of spikes are clutter and cost
    wrap_fade_fraction = 0.03   # slide across the seam instead of popping
    # The barostat below outlives the setup, so the cell is a different size every
    # frame and the runtime has to read it back: the drawn outline, the periodic
    # images, the minimum-imaging in the analysis pair list and the
    # `area_per_particle` observable all key off it.
    cell_is_live = True

    params = (
        structural("n_cols", 30, "particles per row (x)"),
        structural("n_rows", 30, "rows (y)"),
        structural("a", 0.8, "hexagonal spacing (the paper's benchmark value)"),
        # Kept shallow on purpose: an evaporated particle stays right by the
        # sheet, so the membrane's own attraction recaptures it.
        structural("z_half", 4.0, "out-of-plane half-height of the container"),
        structural("settle_steps", 1000, "barostat relaxation to zero lateral tension"),
        # How much of the (possibly tiled) membrane the camera frames, as a
        # multiple of the cell's in-plane half-extent. Pair it with
        # RenderStyle.periodic_images: framing 1.0 with a 3x3 tiling wastes the
        # tiling off-screen, framing 1.8 with no tiling wastes the frame.
        structural("view_span", 1.0, "camera framing: multiples of the cell half-width"),
        # How far past the cell's centre the camera aims, in multiples of its
        # in-plane half-width. 0 looks straight at the middle of the cell, which
        # leaves the near edge floating in the middle of the frame; aiming
        # further in drops that edge toward the bottom, which is what you want
        # when the cell is the FRONT of a tiled block and the images recede
        # behind it (see RenderStyle.periodic_images).
        structural("view_aim_ahead", 0.0, "camera framing: aim this far past the centre"),
        structural("baro_press", 0.0, "target lateral pressure (tension-free)"),
        # Quick, because this one has a JOB TO FINISH rather than a hand to keep
        # up with: it runs once, at build time, with nothing watching, and what it
        # is for is arriving at the tension-free cell before the user sees
        # anything. At 2.0 it did not -- it left the sheet at -0.055 and the rod
        # deck (which starts from a coarser lattice) at -0.082 and still moving,
        # so the demo began under a lateral tension that had nothing to do with
        # the physics being shown. See `baro_damp_run` for the other one.
        structural("baro_damp", 0.2, "settle barostat relaxation time"),
        # The relaxation time of the barostat that KEEPS RUNNING. Ten times
        # quicker than the settle's, and the reason is that this one has to track
        # a LIVE dial: the tension-free area at T = 0.2 is 22% above the one at
        # T = 0.001, so a temperature change is a real area change the cell has to
        # follow while the user watches. Measured on the 30x30 sheet, this reaches
        # zero pressure in ~150 frames (15 tau) with no overshoot, and the settled
        # cell then holds to within 0.1% -- fast enough to follow the dial, quiet
        # enough that the box is not visibly breathing.
        #
        # (`press/berendsen` dilates at a rate proportional to 1/(damp * modulus),
        # and `modulus` defaults to 10 in LJ units. 0.2 here is the same rate as
        # damp 2.0 with modulus 1.0 -- verified equal, and expressed in the damp so
        # there is one dial rather than two that multiply.)
        structural("baro_damp_run", 0.2,
                   "relaxation time of the barostat that keeps running"),
        structural("hold_steps", 200,
                   "silent settle after the running barostat is installed"),
        # Headroom for the barostat's expansion in the camera framing.
        # `fit_to_points` runs once at setup (see app._setup_viewport), so a cell
        # that then grows has nothing to re-fit it. Sized on the measured top of
        # the ORDERED range: the tension-free cell reaches 1.126 of the setup box
        # at T = 0.2 and 1.187 at T = 0.28, just under `melt_temp`. Past melting
        # the sheet is not a membrane any more and the cell inflates without
        # bound, which no fixed headroom covers and none should try to.
        #
        # It is close to free on this scenario, which is why it is generous rather
        # than exact: `view_span` frames the real cell and the periodic images
        # fill the rest of the frame anyway, so pulling back 20% shows more of the
        # tiling rather than emptier picture. The real cell, its outline and the
        # control net stay centred.
        structural("view_headroom", 1.20,
                   "camera framing: extra in-plane room for the barostat"),
        # WHERE THE MEMBRANE SITS IN THE CONTAINER, as a height. 0 is the middle,
        # which is right for a sheet that is pulled at from both sides.
        #
        # It is a lever for a scenario whose interesting direction is DOWN. The
        # rod's invagination goes one way -- the rod needs a few sigma of
        # clearance above the membrane to start out of contact, and everything
        # after that is depth -- so a membrane in the middle of its container
        # spends half the container on headroom nothing uses. Lifting the plane
        # spends that on depth instead, and it is free: the box does not get any
        # bigger, which matters because an over-deep box costs ~20% in the pair
        # loop for no extra pairs (measured, see mesomem_rod.py on `z_half`).
        #
        # The leash is centred on the ORIGIN, not on the membrane (see
        # modes.GameMode.constrain), so raising the plane also deepens the travel
        # the existing leash allows below it, and drops the membrane's image into
        # the top of the frame where a camera aimed at the origin is looking under
        # it. Both of those are the point.
        structural("plane_z", 0.0, "height of the membrane plane in the container"),
        structural("k_plane", 0.1, "pull toward the membrane's own z-plane"),
        structural("k_align", 10.0, "normal-up alignment rate"),
        structural("tracer_fraction", 0.3,
                   "where to place the highlighted diffusion tracer, as a "
                   "fraction in from the front-left corner (None -> no tracer)"),
    )

    def build(self, params, rng):
        n_cols, n_rows, a = int(params["n_cols"]), int(params["n_rows"]), params["a"]
        pts2d = hex_lattice_2d(n_cols, n_rows, a)
        pos = np.column_stack([pts2d, np.full(len(pts2d), float(params["plane_z"]))])
        dirs = np.tile([0.0, 0.0, 1.0], (len(pos), 1))
        # Sized exactly to the lattice so the sheet tiles seamlessly.
        lx = n_cols * a
        ly = n_rows * a * math.sqrt(3.0) / 2.0
        box = Box((-lx / 2.0, -ly / 2.0, -params["z_half"]),
                  (lx / 2.0, ly / 2.0, params["z_half"]),
                  periodic=(True, True, False))
        # Particles at a = 0.8 with diameter 1.0 overlap into a continuous sheet,
        # so no bond sticks (and none would tile across the periodic seam).
        return ScenarioBuild(positions=pos, directors=dirs, box=box,
                             brightness=self._tracer_brightness(pts2d, params))

    def _tracer_brightness(self, pts2d, params):
        """Brighten a small cluster -- one particle and its six nearest
        neighbours -- so it can be followed as it diffuses through the membrane.
        Chosen once from the initial lattice; because the runtime orders
        rendering by atom id, the highlight tracks the same particles as they
        wander."""
        frac = params["tracer_fraction"]
        if frac is None:
            return None
        lo, hi = pts2d.min(axis=0), pts2d.max(axis=0)
        target = lo + frac * (hi - lo)
        centre = int(np.argmin(np.linalg.norm(pts2d - target, axis=1)))
        d = np.linalg.norm(pts2d - pts2d[centre], axis=1)
        bright = np.ones(len(pts2d))
        bright[np.argsort(d)[1:7]] = 1.5
        bright[centre] = 2.1
        return bright

    def create_commands(self, params, build, seed):
        return ["set group all dipole 0.0 0.0 1.0"]

    def pre_control_settle(self, params, seed):
        """Langevin + a Berendsen barostat on x,y to reach the tension-free
        equilibrium spacing. The barostat is not an integrator (it rides on top
        of nve/sphere), so it only relaxes the in-plane spacing. Run before any
        control fixes exist so it relaxes a free sheet."""
        p, damp = params["baro_press"], params["baro_damp"]
        return [
            f"fix settle_bath all langevin 0.0 0.0 {self.langevin_damp} {seed} omega yes",
            f"fix settle_baro all press/berendsen x {p} {p} {damp} "
            f"y {p} {p} {damp} couple xy",
            f"run {int(params['settle_steps'])}",
        ]

    def settle_cleanup_commands(self):
        return ["unfix settle_baro", "unfix settle_bath"]

    def group_commands(self, params, controlled_id):
        """A `membrane` group: every bead the running barostat is allowed to
        rescale.

        In game mode that is everything except the driven bead. The barostat
        dilates coordinates toward the origin every step, and the driven bead's
        position is the one thing in the scene that is set in ABSOLUTE terms -- it
        is held on the control plane inside a leash centred on the origin, and the
        net drawn around it says exactly where it can go. Rescaling it too would
        drag it out from under the hand and out from under its own net.

        In sim mode there is no driven bead, so it is simply every bead; `type 1`
        rather than `all` for the same reason RodOnSheet uses it -- the two read
        the same on a single-species sheet, and it is the membrane that is meant.
        """
        if controlled_id is None:
            return ["group membrane type 1"]
        return ["group membrane subtract all controlled"]

    def post_control_settle(self, params):
        """Install the barostat that keeps running, and let the cell find itself.

        A SECOND fix rather than simply not unfixing the settle's: that one is
        `dilate all` and runs before any control fix exists (so that it relaxes a
        free sheet), and this one has to leave the driven bead alone and has its
        own, quicker relaxation time. The same split RodOnSheet makes, for the
        same two reasons.

        The class docstring has the measurements; the short version is that a
        frozen cell is a fixed projected area, a fixed projected area is the wrong
        one at every temperature but the settle's, and a laterally compressed
        membrane buckles into a standing ripple rather than undulating thermally.
        """
        p, damp = params["baro_press"], params["baro_damp_run"]
        return [
            f"fix baro membrane press/berendsen x {p} {p} {damp} "
            f"y {p} {p} {damp} couple xy dilate partial",
            f"run {int(params['hold_steps'])}",
        ]

    def housekeeping(self, positions, params, controlled=None, box=None):
        """Plane centring toward the membrane's own plane (`plane_z`, which is
        z = 0 unless a scenario lifted it), plus a rigid normal-up rotation of the
        whole sheet -- the same "smallest principal component upward" idea as the
        patch's ring torque, made size-independent by applying it as a rotation
        rate rather than a torque."""
        sel = _exclude(len(positions), controlled)
        if len(sel) < 3:
            return None
        p = positions[sel]
        f = np.zeros((len(positions), 3))
        f[sel, 2] += -params["k_plane"] * (p[:, 2] - float(params["plane_z"]))
        omega = align_normal_rate(p, params["k_align"])
        f[sel] += np.cross(omega[None, :], p - p.mean(axis=0))
        return f

    def camera(self, box):
        """The eye pulls back with `view_span`, not just the zoom.

        Re-framing alone cannot show a wider membrane: at 1.7x the cell the
        sheet's near edge reaches the camera's own position, and fitting THAT in
        sends the zoom to infinity. Scaling the distance keeps the geometry
        similar, so the picture is the same one seen from further away."""
        params = self.new_params()
        span = max(box.lengths[0], box.lengths[1]) * params["view_span"]
        aim = params["view_aim_ahead"] * 0.5 * box.lengths[1]
        return dict(eye=(0.0, -0.85 * span + aim, 0.6 * span),
                    target=(0.0, aim, 0.0), up=(0.0, 0.0, 1.0), fov_deg=34.0)

    def fit_points(self, params, box):
        """Frame the cell, or more of it when the renderer is drawing periodic
        images: `view_span` multiplies the in-plane extent the camera pulls back
        to cover, so a 3x3 tiling can actually be seen rather than sitting mostly
        off the edges of the frame. 1.0 frames the real cell exactly.

        `view_headroom` on top of it, because the cell this is handed is the one
        the barostat starts from rather than the one it will settle at, and the
        camera is fitted once (see app._setup_viewport)."""
        span = params["view_span"] * params["view_headroom"]
        aim = params["view_aim_ahead"] * 0.5 * box.lengths[1]
        corners = box.corners().astype(float)
        corners[:, :2] *= span
        corners[:, 1] += aim
        return np.vstack([corners, [(0.0, aim, 2.6), (0.0, aim, -2.2)]])


class RodOnSheet(HexSheet):
    """A HexSheet with one rigid rod (atom type 2) hanging above it, at constant
    lateral tension.

    Most of the membrane -- the periodic lattice, the plane-centring housekeeping,
    the tension-free settle -- is the sheet's, unchanged. What is added is one
    particle of a second type, placed on the control plane at a clearance the
    rod's own interaction cannot yet reach with its long axis lying IN that plane;
    a barostat that KEEPS RUNNING; and a camera in the membrane's own plane,
    because a wrap is a profile rather than a surface.

    THE BAROSTAT IS THE SHEET's NOW, and what is left here is its TIMESCALE.
    Covering a rod costs membrane area, and in a frozen periodic cell the only
    place that area can come from is stretching the lattice, so the membrane
    cannot invaginate the rod -- it just dents. Holding the lateral pressure at
    its target instead lets the projected area shrink as the wrap grows, which is
    the same thing the collaborator's reference deck does (`fix nph/sphere x .. y
    .. couple xy` running through all three of its rod phases, with the target
    ramped to set the tension). Zero pressure is the tension-free ensemble;
    `baro_press` is the dial for putting the membrane under tension instead, which
    is what suppresses wrapping.

    What this scenario does still choose for itself is `baro_damp_run`: the sheet
    wants a quick one, because there the barostat is tracking a temperature dial
    and should have finished before the user notices. Here it is tracking a wrap
    the user is actively pushing into the membrane, and how fast it gives up area
    IS how stiff the membrane feels under the hand (see mesomem_rod.py, which sets
    it). Same fix, an order of magnitude slower.

    It is a Berendsen barostat rather than the deck's Nose-Hoover one for the same
    reason the settle uses one: `press/berendsen` rides on top of whatever
    integrator is already there, so the force field keeps its own `nve/sphere
    update dipole` and the directors keep being integrated. It also has no
    oscillation of its own to fight the user's hand with.

    Two deliberate choices about where the rod starts:

      IT STARTS OUT OF CONTACT. `rod_height` is above the rod-membrane cutoff, so
        the barostat settle relaxes a membrane the rod is not yet touching, and
        the first thing the user does -- bringing it down until it grabs -- is the
        beginning of the demo rather than something that already happened.
      IT STARTS ON THE LEASH's plane, at y = 0 and inside its z half-extent. The
        leash is centred on the origin (see modes.GameMode.constrain), so a rod
        placed above it would be yanked down to the limit on the first frame.
        `verify_reach` says so out loud rather than leaving that to be discovered.
    """

    name = "rod_on_sheet"
    # The rod is a second species whose orientation is the whole point, and it is
    # the only particle whose director is worth a spike -- but the sheet has
    # thousands of them, and the renderer draws spikes for all or none. The rod's
    # own body (see MesoMemRod.glyph_spheres) shows which way it points instead.
    director_arrows = False
    # The cell changes size every step, so the runtime has to read it back rather
    # than trusting the geometry it asked for -- the drawn box outline, the
    # periodic seam handling and the minimum-imaging in the analysis pair list all
    # key off it. See PlaygroundSystem.step.
    cell_is_live = True

    params = HexSheet.params + (
        structural("rod_height", 3.5,
                   "the rod's starting height above the membrane plane"),
        structural("rod_axis", (1.0, 0.0, 0.0),
                   "the rod's initial long axis (must lie in the control plane)"),
        structural("view_elevation_deg", 6.0,
                   "camera elevation above the membrane plane (0 = exactly "
                   "edge-on, so the membrane is a line)"),
        # How much of the rod's OUT-OF-PLANE travel to keep in frame. None derives
        # it from the starting height, which is right only while that is the whole
        # of the travel; a playground whose leash reaches well past it has to say
        # so, or the rod is driven out of the bottom of the picture. See fit_points.
        structural("view_z_half", None,
                   "half-height to frame, in sigma (None -> the rod's starting "
                   "height plus a margin)"),
    )

    def camera(self, box):
        """Edge-on: the membrane is a LINE and the rod invaginates it in section.

        This is the whole reason this scenario has its own camera. The sheet looks
        down at its membrane from 35 degrees, which is right for watching a
        deformation travel across a surface. A wrap is not that -- it is a
        PROFILE. What there is to see is how far the rod has sunk below the
        surface and how far the membrane has climbed around it, and both of those
        are depths. Seen from above they are entirely along the view axis; seen
        edge-on they are the picture.

        It also happens to be the plane the rod is steered in: the control plane is
        the world xz-plane, so a camera looking along -y puts the two joystick
        axes exactly on screen-horizontal and screen-vertical. The rod goes where
        the stick goes.

        A few degrees up rather than exactly zero, so the membrane reads as a
        narrow band -- a surface seen almost edge-on -- rather than a single row of
        beads with no thickness to judge the rod against.

        This angle only works together with `RenderStyle.section_min`, which is
        declared alongside it in the playground. A monolayer is opaque: without
        the cut, the rows of beads between the camera and the rod sit at exactly
        the height the rod is being pushed to and hide the whole invagination, and
        no elevation fixes it -- raise the camera and the depths foreshorten away,
        lower it and the near rows are in the way. The cut removes them; this
        angle then shows what is left.

        The distance is measured from the CELL, not from the framed patch: this
        cell is 46 sigma deep and the camera has to stand well outside it. Close
        in, the nearest rows are a fraction of the distance the rod is and
        perspective blows them up into a wall of beads across the bottom of the
        frame; a long way back they are nearly the same size as the rod's own row.
        The app zooms to `fit_points` afterwards, so standing back costs nothing
        but the flatter perspective a section wants.
        """
        params = self.new_params()
        span = max(box.lengths[0], box.lengths[1]) * params["view_span"]
        el = math.radians(min(max(float(params["view_elevation_deg"]), 0.0), 85.0))
        d = 0.5 * box.lengths[1] + 2.5 * span
        return dict(eye=(0.0, -d * math.cos(el), d * math.sin(el)),
                    target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0), fov_deg=34.0)

    def build(self, params, rng):
        sheet = super().build(params, rng)
        axis = np.asarray(params["rod_axis"], dtype=float)
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        # ABOVE THE PLANE, which is what `rod_height` has always said it was and
        # was only ever the same as an absolute z while the plane sat at 0. A
        # scenario that lifts the membrane (see `plane_z`) lifts the rod with it,
        # so the clearance it starts at -- outside the rod-membrane cutoff, which
        # is the one thing that number has to get right -- is preserved.
        rod = np.array([[0.0, 0.0, float(params["plane_z"])
                         + float(params["rod_height"])]])
        positions = np.vstack([sheet.positions, rod])
        directors = np.vstack([sheet.directors, axis[None, :]])
        # Type 2 LAST, so `Control(atom="last")` names it -- and so the membrane
        # keeps the ids the sheet's tracer highlight was chosen from.
        types = np.concatenate([np.ones(len(sheet.positions), dtype=int), [2]])
        brightness = sheet.brightness
        if brightness is not None:
            brightness = np.append(brightness, 1.0)
        return ScenarioBuild(positions=positions, directors=directors,
                            types=types, box=sheet.box, brightness=brightness)

    def create_commands(self, params, build, seed):
        """The membrane's directors point along +z; the rod's axis lies in the
        control plane. Issued in that order, so the second overwrites the first
        for type 2 only."""
        axis = np.asarray(params["rod_axis"], dtype=float)
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        return [
            "set type 1 dipole 0.0 0.0 1.0",
            f"set type 2 dipole {axis[0]} {axis[1]} {axis[2]}",
        ]

    def group_commands(self, params, controlled_id):
        """A group for the membrane alone, by type.

        By TYPE rather than reusing the mode's `bath` (which is also "everything
        but the rod"): the barostat below has to exist in sim mode too, where
        there is no controlled particle and so no `bath`.
        """
        return ["group membrane type 1"]

    def post_control_settle(self, params):
        """Install the barostat that keeps running, and let the cell find itself.

        The same shape as HexSheet's -- which is where the argument for keeping a
        barostat running at all now lives -- but on the membrane group by TYPE
        (see `group_commands`) and at this scenario's own, much slower
        `baro_damp_run`.

        `dilate partial` on the membrane group is what keeps the rescaling off the
        rod: the pressure is still measured over the whole system (that is a
        global quantity and `press/berendsen` computes it that way whatever its
        group), but the coordinate rescaling touches membrane beads only, so the
        rod is not quietly dragged toward the origin every time the cell shrinks
        under it -- which would fight the leash and put the drawn control net
        somewhere other than where the rod can actually go.
        """
        p, damp = params["baro_press"], params["baro_damp_run"]
        return [
            f"fix baro membrane press/berendsen x {p} {p} {damp} "
            f"y {p} {p} {damp} couple xy dilate partial",
            f"run {int(params['hold_steps'])}",
        ]

    def fit_points(self, params, box):
        """The framed region, plus the rod's full travel.

        Not the sheet's version: that frames the CELL's corners, and this cell is
        several times the size of what the camera is looking at (`view_span` well
        below 1), so fitting its corners would pull the camera back until the rod
        was a speck. The framed patch is the in-plane extent `view_span` asks for,
        and the two z points keep the rod in shot over its whole travel.

        THE TRAVEL IS THE LEASH's, not the starting height's, and the two are not
        the same thing on a playground that means to push the rod several rod
        lengths into the membrane -- the rod starts a little above the surface and
        ends up well below it. This used to frame `rod_height + 1.5` either way,
        which silently put the bottom half of the drawn control net, and the rod
        once it was driven down there, outside the picture. `view_z_half` is what a
        playground says its leash reaches; the old expression is the default, for a
        scenario whose leash is about the size of its clearance.
        """
        span = float(params["view_span"])
        hx = 0.5 * box.lengths[0] * span
        hy = 0.5 * box.lengths[1] * span
        declared = params["view_z_half"]
        hz = (float(declared) if declared is not None
              else float(params["plane_z"]) + float(params["rod_height"]) + 1.5)
        return np.array([(x, y, 0.0) for x in (-hx, hx) for y in (-hy, hy)]
                        + [(0.0, 0.0, hz), (0.0, 0.0, -hz)])

    def verify_reach(self, control, rod_cutoff):
        """Complain if the leash cannot express the demo.

        Called from the playground's own test rather than at build time: these are
        statements about a Control and a force field the scenario does not own, and
        getting either wrong produces a demo that looks like it works and cannot
        actually touch the membrane -- which is exactly the kind of thing worth a
        test and not worth a runtime check on every build.
        """
        params = self.new_params()
        # TWO COORDINATES, and keeping them apart is the whole of this method now
        # that the membrane can be lifted off the container's midplane. `h` is the
        # rod's CLEARANCE above the membrane, which is what the cutoff is about;
        # `z` is where that puts it in the world, which is what the leash -- centred
        # on the ORIGIN, not on the membrane -- is about.
        h = float(params["rod_height"])
        plane = float(params["plane_z"])
        z = plane + h
        problems = []
        if z > control.leash[1]:
            problems.append(
                f"the rod starts at z {z} (plane {plane} + rod_height {h}), outside "
                f"the leash's z half-extent {control.leash[1]}: it would be clamped "
                f"down to the limit on the first frame")
        if h <= rod_cutoff:
            problems.append(
                f"rod_height {h} is inside the rod-membrane cutoff {rod_cutoff:.2f}: "
                f"the rod starts already in contact")
        if control.leash[1] - plane < rod_cutoff:
            problems.append(
                f"the leash reaches z {control.leash[1]}, only "
                f"{control.leash[1] - plane:.2f} above the membrane at {plane}, "
                f"which is inside the rod-membrane cutoff {rod_cutoff:.2f}: the rod "
                f"can never be lifted clear of the membrane")
        return problems


class RandomFill(Scenario):
    """N particles at random positions and orientations in a fully periodic cell
    -- the paper's spontaneous-assembly test.

    Under Langevin dynamics the disordered gas coarsens: small patches by
    t ~ 500 tau, coalescing into large planar membranes by t ~ 2000 tau. There is
    nothing to steer, so this is normally run in sim mode -- though game mode
    works too (it will pick a controllable particle), which is exactly the
    orthogonality the mode split buys.
    """

    name = "random_fill"
    timestep = 0.01
    sim_time_per_frame = 0.2
    # Weaker friction than the probing scenarios (larger damp = weaker drag) lets
    # particles diffuse and find each other, so assembly proceeds watchably.
    langevin_damp = 1.0
    neighbor_skin = 0.6
    director_arrows = False

    params = (
        structural("n", 1500, "particle count"),
        structural("box", 20.0,
                   "cubic cell side (phi = N*Vp/L^3 ~ 0.1 at the defaults)"),
        # 0.9 sigma sits just outside the 4-2 core's minimum at r = sigma, where
        # the pair force is small and attractive, so no two particles are dropped
        # inside each other's hard core and blow up the first step.
        structural("overlap", 0.9, "minimum centre-to-centre separation when seeding"),
        structural("maxtry", 200, "placement attempts per particle"),
        # --- two corrections that keep the run watchable, both OFF at 0 ------
        # Neither is part of the paper's experiment. Measured over 40k steps
        # against unforced controls: the field takes the membranes' common normal
        # to within 4-8 degrees of vertical (unforced, it wanders 35-64) while
        # the nematic order rises just as it does without it, so what is being
        # steered is the orientation, not the assembly. The centring only acts on
        # an axis the particles are actually concentrated along, so it does
        # nothing at all until something has formed.
        structural("k_upright", 0.6,
                   "field aligning directors with z, so sheets form flat (0 = off)"),
        structural("center_accel", 0.05,
                   "drift acceleration re-centring the aggregate (0 = off)"),
        structural("center_softness", 2.0,
                   "distance over which the centring eases off near the middle, sigma"),
        structural("center_concentration_min", 0.15,
                   "circular concentration below which an axis reads as uniform"),
        structural("center_concentration_full", 0.35,
                   "circular concentration at which centring reaches full strength"),
    )

    def housekeeping(self, positions, params, controlled=None, box=None):
        """Nudge whatever has assembled back to the middle of the cell.

        "The middle" needs care on a torus: an ordinary mean is meaningless
        there (a blob straddling the seam averages to the far side of the box).
        The circular mean is not -- map each coordinate onto the unit circle,
        theta = 2*pi*x/L, average the unit vectors, and read the angle back. That
        answer is translation-covariant, exactly as a centre of mass should be.
        The length R of the averaged vector comes free and is the useful part: it
        is the CONCENTRATION of the distribution along that axis, 1 for a tight
        blob and ~1/sqrt(N) for a uniform spread. So the correction is applied
        per axis and gated on R, which gives the right behaviour for free:
        a droplet is concentrated on all three axes and gets centred in all
        three, while a lamella is concentrated only along its normal and gets
        centred only in that direction -- exactly right, since it is uniform
        along the other two and has no centre there to find.
        Uniform on every particle, so it translates the system without shearing
        it, and eased off (tanh) near the target so it settles instead of
        oscillating. The Langevin bath turns the sustained acceleration into a
        slow terminal drift.
        """
        accel = params["center_accel"]
        if accel <= 0.0 or box is None:
            return None
        lo = np.asarray(box.lo, dtype=float)
        L = np.asarray(box.lengths, dtype=float)
        theta = 2.0 * np.pi * (positions - lo) / L
        c, sn = np.cos(theta).mean(axis=0), np.sin(theta).mean(axis=0)
        concentration = np.hypot(c, sn)
        mean = lo + L * (np.arctan2(sn, c) % (2.0 * np.pi)) / (2.0 * np.pi)
        d = np.asarray(box.center, dtype=float) - mean
        d -= L * np.round(d / L)                       # minimum image
        gate = np.array([_ramp(r, params["center_concentration_min"],
                               params["center_concentration_full"])
                         for r in concentration])
        a = accel * gate * np.tanh(d / max(params["center_softness"], 1e-6))
        if not np.any(a):
            return None
        f = np.tile(a, (len(positions), 1))
        if controlled is not None:
            f[controlled] = 0.0
        return f

    def director_housekeeping(self, positions, directors, params, controlled=None,
                              box=None):
        """A weak field that prefers membranes lying flat, normal along z.

        WHY A FIELD ON EACH DIRECTOR, AND NOT A TORQUE ON THE WHOLE SYSTEM.
        The obvious construction -- read the common normal off the directors as a
        tensor, and apply the one rigid rotation that brings it upright -- was
        tried first and is worse than useless. It rotates every director away
        from ITS OWN local membrane normal, which is precisely what the force
        field's tilt term exists to punish, so the membrane wins and the only
        result is energy pumped into the aggregates: measured over 60k steps
        against an unforced control, it did not flatten anything and pulled the
        nematic order back down from 0.52 to 0.11 as the sheets it was wrestling
        came apart.
        And that is the honest answer for a PERIODIC cell, not a bug. A membrane
        spanning one has to be commensurate with it -- it lies parallel to a pair
        of box faces -- so its available orientations are three discrete choices,
        not a continuum to be rotated through. Getting from one to another means
        dissolving and re-forming, which no gentle torque can drive.
        What CAN be biased is which one it forms in. Each director is pulled
        toward the nearer of +/-z, exactly as an external field aligns a nematic:
        in the disordered gas, where the particles are free to turn, that tilts
        the odds toward horizontal patches nucleating, and they then grow the way
        they always would. It is self-extinguishing -- the torque is proportional
        to sin(angle to z), so a membrane that IS flat feels nothing at all --
        which is why it needs no gate and never fights a finished sheet.
        """
        k = params["k_upright"]
        if k <= 0.0 or not len(directors):
            return None
        # Toward the NEARER pole: +n and -n are the same physical orientation, so
        # a director in the lower half turns toward -z, not the long way round.
        sense = np.where(directors[:, 2] < 0.0, -1.0, 1.0)
        up = np.zeros_like(directors)
        up[:, 2] = sense
        w = k * np.cross(directors, up)
        if controlled is not None:
            w[controlled] = 0.0
        return w

    def build(self, params, rng):
        """Positions placed HERE, in numpy, not by LAMMPS.

        This used to hand the job to `create_atoms 1 random N seed box overlap ...`
        and return an empty array, which is the obvious thing and was the single
        most expensive line in the whole remote demo: 47 seconds for the 50,000-bead
        cell, against 0.00 s for the same command with the overlap check removed. It
        is the entire cost of a remote build, and of a remote Reset, which is the
        same work over again. See state.random_points_min_separation for what
        replaces it (half a second for the same 50,000, drawing from the same
        distribution) and why the two are interchangeable.

        `maxtry` survives as the round budget rather than as a per-particle attempt
        count -- both are "how hard to try before admitting the cell is too full" --
        and unlike LAMMPS' version, failing here says so instead of quietly creating
        fewer atoms than asked for.
        """
        box = Box.cube(params["box"], (True, True, True))
        pos, _rounds = random_points_min_separation(
            int(params["n"]), box, float(params["overlap"]), rng,
            rounds=max(8, int(params["maxtry"]) // 4))
        return ScenarioBuild(positions=pos, directors=None, box=box)

    def outline(self, params):
        """The cell and the count, with nothing placed.

        This is the whole reason `Scenario.outline` exists. The composition here is
        one species and the count is a declared parameter, so both are known without
        touching a coordinate -- and the coordinates are the expensive part (half a
        second for 50,000, and it is the remote playgrounds that are that big and
        that are switched to and away from mid-talk).
        """
        box = Box.cube(params["box"], (True, True, True))
        return (ScenarioBuild(positions=np.zeros((0, 3)), directors=None, box=box),
                int(params["n"]))

    def create_commands(self, params, build, seed):
        # Random initial director orientations -- the disordered orientational
        # start the paper's self-assembly begins from.
        return [f"set group all dipole/random {seed} 1.0"]

    def camera(self, box):
        """A three-quarter view from a LONG lens, not a wide one.

        The distance here is the perspective strength, and it is the only thing
        that sets it -- the runtime zooms to fit (Camera3D.fit_to_points), so the
        declared `fov_deg` never survives and moving the eye out is compensated by
        a longer focal length rather than a smaller picture. Which means the one
        number that matters is how far away the eye is compared to how big the
        cell is.

        At the 1.44 cell-widths this used to sit at, the fit came out at a 75
        degree field: a wide-angle lens pressed up against a cube, with the near
        beads drawn three times the size of the far ones and the cell's faces
        visibly flaring outward. That reads as a fisheye photograph of a box
        rather than as a box. At 3.0 the fit lands near the 34 degrees this
        scenario asked for all along -- a normal lens, near beads about 1.6x the
        far ones, the cell's opposite faces near enough parallel to read as a
        cube, and enough perspective left that the turntable still tells you
        which way it is turning. Further out approaches an orthographic
        projection, where the depth stops doing any of the work of separating the
        aggregates and the box flattens into a wall of beads.
        """
        s = 3.0 * max(box.lengths)
        # The direction is unchanged: off to the right, well in front, above.
        d = np.array([0.6, -1.1, 0.7])
        eye = tuple(s * d / np.linalg.norm(d))
        return dict(eye=eye, target=(0.0, 0.0, 0.0),
                    up=(0.0, 0.0, 1.0), fov_deg=34.0)


class VesiclePolymer(Scenario):
    """A closed MesoMem vesicle with a melt of ring polymers sealed inside it.

    The collaborator's system (see polymer/ at the repo root): a spherical
    monolayer of directored beads -- the membrane force field's beads, radial
    directors, nothing else -- with a chromatin-like melt of self-avoiding ring
    polymers filling the lumen. The rings interact with each other and with the
    membrane through a purely repulsive core, so nothing sticks: what the polymer
    does to the vesicle it does by PUSHING, which is the only mechanism in the
    picture and is why it is worth watching.

    THE VESICLE. Beads at the face centres of a geodesic sphere (see
    state.icosphere_faces), which is the construction that gives a nearly uniform
    triangulation -- the membrane is a fluid and will re-space itself, but it
    cannot be handed two crowded poles to start from. Its RADIUS is not a free
    parameter: it follows from the bead count and the spacing, because those two
    are what the force field cares about. Ask for more beads at the same spacing
    and you get a bigger vesicle, which is the honest relation.

    Which also means `n_membrane` has a floor. MesoMem's tilt term prefers a FLAT
    membrane (c0 = 0), so closing one into a sphere costs bending energy that goes
    as 1/R^2, and a small enough vesicle is not a metastable object -- it roughens,
    and a chain pressed against it can find its way out. At the shipped size
    (18,000 beads, R ~ 35 sigma) nothing escapes over thousands of steps; a tenth
    of that radius is visibly leaky, which is what the small system in
    tests/test_vesicle_polymer.py is and why it asserts a fraction rather than a
    maximum.

    Also worth knowing: the packing on a geodesic sphere is not uniform, it varies
    by about half between a parent face's centre and one of the icosahedron's
    twelve five-fold vertices (see icosphere_spacing). The membrane relaxes that
    out within a few hundred steps, and `a` is a mean rather than a promise.

    THE POLYMER. Each ring is a Hamiltonian cycle on a small cubic lattice (see
    state.lattice_ring): a closed, compact, self-avoiding walk in which every bond
    is exactly one lattice step and no two beads are closer than one. That matters
    more than it sounds. The obvious construction -- a random walk -- produces
    overlapping beads, and a FENE bond plus a repulsive core applied to
    overlapping beads is an explosion on the first step; the usual fix is an
    offline soft-core push-off, which is exactly the pre-relaxed data file this
    scenario exists to not need. A lattice ring needs no relaxation at all: it is
    already a valid configuration of the potential it is about to be handed to.

    The rings are then laid out on a coarse grid inside the lumen, thinly enough
    to hit the reference system's volume fraction, and each is jittered off its
    lattice so the run does not begin from a crystal. They swell and interpenetrate
    within the first few hundred steps.

    HOW THE POLYMER REACHES LAMMPS. Not through `create_atoms`, which places
    points and knows nothing of bonds, but through a molecule template written at
    build time and instanced once (see `create_commands`). So `build` returns the
    positions of the WHOLE system -- which is what the remote client reads the
    composition off -- while `n_direct` stops the runtime's own upload at the
    membrane. The template's own coordinates are what LAMMPS actually places, up
    to the rigid rotation `create_atoms ... mol` applies; nothing draws the
    polymer half of `positions`.
    """

    name = "vesicle_polymer"
    timestep = 0.005
    sim_time_per_frame = 0.05
    # As RandomFill: weak friction, so the rings can actually explore. A ring melt
    # relaxes on its own slow timescale and a stiff bath would freeze it in the
    # lattice it was built on.
    langevin_damp = 1.0
    neighbor_skin = 0.6
    director_arrows = False     # tens of thousands of beads; spikes are cost and clutter

    params = (
        structural("n_membrane", 18000,
                   "target membrane beads (rounded to the nearest 20*nu^2 -- an "
                   "icosphere only comes in those sizes)"),
        structural("a", 0.85,
                   "membrane nearest-neighbour spacing. The sheet playgrounds' "
                   "relaxed value: the vesicle then starts near tension-free "
                   "rather than pre-compressed"),
        structural("box_factor", 1.12,
                   "cubic cell half-side, as a multiple of the vesicle radius"),
        structural("n_polymer", 32000,
                   "target polymer beads (rounded down to whole rings)"),
        structural("ring_side", 8,
                   "sites per side of a ring's cubic block, so ring_side^3 beads "
                   "per ring. Must be even (see state.lattice_ring)"),
        structural("bond_length", 1.0,
                   "polymer bond length at build, and so the ring lattice's step"),
        structural("fill_fraction", 0.74,
                   "how far out the polymer is laid, as a fraction of the vesicle "
                   "radius. Short of the membrane, so the first thing the run "
                   "shows is the melt expanding to meet it"),
        structural("jitter", 0.15,
                   "random per-bead displacement off the ring lattice, in bond "
                   "lengths -- enough that the melt does not begin as a crystal, "
                   "small enough to leave every bond well inside FENE's range"),
        structural("settle_steps", 400,
                   "silent relaxation before the run is handed over"),
        structural("view_span", 1.0,
                   "camera framing: multiples of the cell's half-width"),
    )

    # --- geometry -------------------------------------------------------------

    def subdivision(self, params):
        """The icosphere subdivision nearest the requested membrane count, and the
        count it actually gives (20*nu^2)."""
        nu = max(1, int(round(math.sqrt(max(params["n_membrane"], 20) / 20.0))))
        return nu, 20 * nu * nu

    def radius(self, params):
        """Vesicle radius, in sigma. Set by the bead count and the spacing -- see
        the class docstring."""
        nu, _ = self.subdivision(params)
        return float(params["a"]) / icosphere_spacing(nu)

    def ring_centres(self, params, radius):
        """Where the ring blocks go: a cubic grid inside the lumen, at the pitch
        that fits the requested number of them.

        The pitch comes from the target count rather than the other way round, so
        `n_polymer` means what it says. Solving it as a continuum -- how many cells
        of pitch p fit in the available sphere -- then counting the real grid and
        tightening if the estimate was optimistic, which it is at the corners.
        """
        side = int(params["ring_side"])
        b = float(params["bond_length"])
        per_ring = side ** 3
        want = max(1, int(params["n_polymer"]) // per_ring)
        block = (side - 1) * b
        half_diag = 0.5 * math.sqrt(3.0) * block
        # The centres have to sit far enough in that the whole block does.
        reach = float(params["fill_fraction"]) * radius - half_diag
        if reach <= 0.0:
            return np.zeros((0, 3)), want
        # A block may never be laid closer to its neighbour than one bond length,
        # or the two rings start out overlapping.
        min_pitch = block + b
        pitch = max(min_pitch,
                    ((4.0 / 3.0) * math.pi * reach ** 3 / want) ** (1.0 / 3.0))
        for _ in range(12):
            k = int(math.floor(reach / pitch)) + 1
            g = np.arange(-k, k + 1) * pitch
            grid = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
            inside = grid[np.linalg.norm(grid, axis=1) <= reach]
            if len(inside) >= want or pitch <= min_pitch:
                break
            pitch = max(min_pitch, pitch * 0.94)
        # Innermost first, so a target the lumen cannot hold gives a smaller,
        # centred melt rather than a hollow shell of rings.
        order = np.argsort(np.linalg.norm(inside, axis=1))
        return inside[order[:want]], want

    def build(self, params, rng):
        nu, n_mem = self.subdivision(params)
        radius = self.radius(params)
        _, normals = icosphere_faces(nu)
        # On the sphere exactly, not at the sub-triangle centroids, which sit a
        # few parts in ten thousand inside it. Same spacing, one fewer thing for
        # the membrane to relax out.
        membrane = normals * radius
        polymer = self._polymer(params, rng, radius)

        positions = np.vstack([membrane, polymer]) if len(polymer) else membrane
        directors = np.vstack([normals, np.zeros((len(polymer), 3))])
        types = np.concatenate([np.ones(n_mem, dtype=int),
                                np.full(len(polymer), 2, dtype=int)])
        half = float(params["box_factor"]) * radius
        return ScenarioBuild(
            positions=positions, directors=directors, types=types,
            # Free, not periodic: the vesicle is a finite object with vacuum
            # around it, and there is nothing here that a periodic image would
            # add except a second vesicle a fraction of a radius away.
            box=Box.cube(2.0 * half), n_direct=n_mem)

    def _polymer(self, params, rng, radius):
        """Every polymer bead's position, rings laid end to end in ring order."""
        centres, _ = self.ring_centres(params, radius)
        if not len(centres):
            return np.zeros((0, 3))
        side = int(params["ring_side"])
        b = float(params["bond_length"])
        ring = (lattice_ring(side, side, side).astype(float) - (side - 1) / 2.0) * b
        # Every ring is the SAME walk, and laid down unturned they all stack the
        # same way -- which is not just ugly but misleading, since a melt of
        # identically oriented rings is a lamellar phase and this is not one. So
        # each is turned by a random signed permutation of the axes: one of the 48
        # maps that take the cubic lattice to itself, so the block's footprint is
        # exactly unchanged and no ring can be swung into its neighbour, while the
        # serpentine inside it points somewhere else. A free rotation would look
        # better still and cannot be had: at the pitch these are laid on, a block's
        # corners sweep past its neighbour's.
        turns = _axis_permutations()
        blocks = np.empty((len(centres), len(ring), 3))
        for i, centre in enumerate(centres):
            blocks[i] = ring @ turns[rng.integers(len(turns))] + centre
        jitter = float(params["jitter"]) * b
        if jitter > 0.0:
            blocks = blocks + rng.uniform(-jitter, jitter, size=blocks.shape)
        return blocks.reshape(-1, 3)

    def ring_count(self, params):
        """How many rings this actually lays -- what `n_polymer` was rounded to."""
        return len(self.ring_centres(params, self.radius(params))[0])

    def particle_count(self, params):
        """Membrane plus polymer, without building anything."""
        _, n_mem = self.subdivision(params)
        return n_mem + self.ring_count(params) * int(params["ring_side"]) ** 3

    # --- LAMMPS side ----------------------------------------------------------

    def create_commands(self, params, build, seed):
        """The membrane's directors, and the polymer itself.

        The directors are set with atom-style variables rather than one `set` per
        bead: they are radial, so each one is its own position normalised, and
        that is a three-line expression LAMMPS evaluates over every atom at once
        instead of twenty thousand commands.

        The polymer arrives as a MOLECULE TEMPLATE -- the one route into LAMMPS
        that carries a topology, and the reason this scenario writes a file at
        all. All of the rings go in one template and are instanced with a single
        `create_atoms`: they are disconnected fragments of one object as far as
        LAMMPS is concerned, which costs nothing and means the melt is placed
        once rather than once per ring (`create_atoms ... mol` applies a random
        rigid rotation, and rotating each ring separately would swing its corners
        into its neighbours).
        """
        n_mem = build.n_uploaded
        cmds = [
            # r first, so the three components below are one divide each rather
            # than three square roots.
            "variable vp_r atom sqrt(x*x+y*y+z*z)",
            "variable vp_mx atom x/v_vp_r",
            "variable vp_my atom y/v_vp_r",
            "variable vp_mz atom z/v_vp_r",
            "set type 1 dipole v_vp_mx v_vp_my v_vp_mz",
        ]
        polymer = build.positions[n_mem:]
        if not len(polymer):
            return cmds
        path = self._write_template(params, polymer)
        return cmds + [
            f"molecule vp_polymer {path}",
            f"create_atoms 0 single 0.0 0.0 0.0 mol vp_polymer {int(seed)}",
        ]

    def _write_template(self, params, polymer):
        """Write the rings out as a LAMMPS molecule file and return its path.

        The per-particle attributes go in the file rather than being `set`
        afterwards for an ordering reason: the force field's own setup runs
        BEFORE this, when no polymer atom exists yet, so a `set type 2` there
        would land on nothing. A template that carries its own diameters and
        masses is self-sufficient whatever order it is instanced in.

        The directory is held on the scenario, so it outlives the command list
        being executed and is cleaned up when the process ends.
        """
        import os
        import tempfile

        if self._tmpdir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="lammps-live-poly-")
        path = os.path.join(self._tmpdir.name, "polymer.mol")
        side = int(params["ring_side"])
        per_ring = side ** 3
        n = len(polymer)
        n_rings = n // per_ring
        with open(path, "w") as f:
            f.write("ring-polymer melt, written by VesiclePolymer\n\n")
            # A ring of L beads has L bonds and L angles: the walk closes, so
            # the last bead is bonded to the first: there is no free end anywhere.
            f.write(f"{n} atoms\n{n} bonds\n{n} angles\n\n")
            f.write("Coords\n\n")
            for i, (x, y, z) in enumerate(polymer, 1):
                f.write(f"{i} {x:.5f} {y:.5f} {z:.5f}\n")
            f.write("\nTypes\n\n")
            f.write("".join(f"{i} 2\n" for i in range(1, n + 1)))
            f.write("\nDiameters\n\n")
            f.write("".join(f"{i} 1.0\n" for i in range(1, n + 1)))
            f.write("\nMasses\n\n")
            f.write("".join(f"{i} 1.0\n" for i in range(1, n + 1)))
            f.write("\nMolecules\n\n")
            f.write("".join(f"{i} {1 + (i - 1) // per_ring}\n"
                            for i in range(1, n + 1)))
            f.write("\nBonds\n\n")
            bid = 0
            for r in range(n_rings):
                base = r * per_ring
                for k in range(per_ring):
                    bid += 1
                    f.write(f"{bid} 1 {base + k + 1} "
                            f"{base + (k + 1) % per_ring + 1}\n")
            f.write("\nAngles\n\n")
            aid = 0
            for r in range(n_rings):
                base = r * per_ring
                for k in range(per_ring):
                    aid += 1
                    f.write(f"{aid} 1 {base + k + 1} "
                            f"{base + (k + 1) % per_ring + 1} "
                            f"{base + (k + 2) % per_ring + 1}\n")
        return path

    def group_commands(self, params, controlled_id):
        """The two species, by type. They are what the integrators below split on
        -- and they have to be groups rather than a single `all`, because only one
        of the two carries an orientation."""
        return ["group membrane type 1", "group polymer type 2"]

    def integrator_commands(self, params):
        """One integrator each, and the reason is the directors.

        `nve/sphere update dipole` integrates a particle's orientation, which the
        membrane needs and the polymer has nothing to integrate -- a polymer bead
        carries no director at all, and handing a zero-length one to that fix asks
        it to normalise nothing. Plain `nve` for the chains.
        """
        return [
            "fix vp_int_mem membrane nve/sphere update dipole",
            "fix vp_int_pol polymer nve",
        ]

    def post_control_settle(self, params):
        return [f"run {int(params['settle_steps'])}"]

    # --- rendering ------------------------------------------------------------

    def render_tints(self, params):
        """The polymer's own palette: a ramp along each ring's contour.

        The membrane keeps the MesoMem banding -- a mix of 0 leaves it alone --
        because that banding IS the physics on those beads, and the yellow
        equator tilting with the director is what a membrane playground is for.
        The polymer has no such thing to show (no director, no orientation), so
        the colour is free to carry the one property those beads DO have that
        nothing else in the picture shows: where along the chain each one sits.
        Following a strand through a melt by eye is otherwise impossible.

        A ramp per RING rather than across the whole melt, so every ring runs the
        full range and reads as a closed loop of its own rather than as one more
        sample of a global gradient.
        """
        _, n_mem = self.subdivision(params)
        n_rings = self.ring_count(params)
        per_ring = int(params["ring_side"]) ** 3
        tints = np.zeros((n_mem + n_rings * per_ring, 4))
        if not n_rings:
            return tints
        # Round-trip: the ramp has to come back to where it started, or a closed
        # ring shows a seam where its two ends meet.
        u = np.linspace(0.0, 1.0, per_ring, endpoint=False)
        t = 0.5 - 0.5 * np.cos(2.0 * np.pi * u)
        ramp = (np.array(POLYMER_COLD)[None, :] * (1.0 - t[:, None])
                + np.array(POLYMER_WARM)[None, :] * t[:, None])
        tints[n_mem:, :3] = np.tile(ramp, (n_rings, 1))
        tints[n_mem:, 3] = 1.0
        return tints

    def camera(self, box):
        """The assembly box's long lens, aimed at the vesicle. See
        RandomFill.camera for why the distance is what sets the perspective."""
        s = 2.6 * max(box.lengths)
        d = np.array([0.6, -1.1, 0.7])
        return dict(eye=tuple(s * d / np.linalg.norm(d)), target=(0.0, 0.0, 0.0),
                    up=(0.0, 0.0, 1.0), fov_deg=34.0)

    def fit_points(self, params, box):
        """Frame the VESICLE, not the cell.

        The box is a container with a margin of vacuum around a sphere, and
        framing its corners spends a third of the picture's width on that vacuum.
        Nor the sphere's BOUNDING BOX, which is the same mistake one step smaller:
        a cube's corners stand out by sqrt(3) past the sphere inscribed in it, so
        framing them leaves the vesicle filling 58% of the frame it was supposed
        to fill. The six axial extremes are the sphere's own silhouette from any
        angle, which is what there is to frame.
        """
        r = float(params["view_span"]) * self.radius(params)
        return np.array([(r, 0.0, 0.0), (-r, 0.0, 0.0), (0.0, r, 0.0),
                         (0.0, -r, 0.0), (0.0, 0.0, r), (0.0, 0.0, -r)])

    def __init__(self, **overrides):
        super().__init__(**overrides)
        # Holds the molecule template alive for as long as this scenario object
        # does -- which is the life of the process, since a playground file is a
        # module-level constant. Rebuilt on every Reset, into the same directory.
        self._tmpdir = None


class Composite(Scenario):
    """Several sub-scenarios' geometries placed at offsets in one shared cell.

    This is what makes "two patches colliding" a three-line playground instead of
    a new 800-line module. The first part's timing, housekeeping and rendering
    settings apply to the whole assembly.
    """

    name = "composite"

    def __init__(self, parts, box=None, periodic=(False, False, False),
                 margin=2.0):
        super().__init__()
        self.parts = tuple(parts)            # ((scenario, offset), ...)
        self._box_side = box
        self._periodic = periodic
        self._margin = margin
        # Each part is configured at its own call site -- compose(hex_patch(
        # n_rings=2), ...) -- so it keeps its own ParamSet rather than having its
        # parameters re-exported under a prefix here.
        self._part_params_cache = tuple(s.new_params() for s, _ in self.parts)
        base = self.parts[0][0]
        self.timestep = base.timestep
        self.sim_time_per_frame = base.sim_time_per_frame
        self.langevin_damp = base.langevin_damp
        self.director_arrows = base.director_arrows
        self.wrap_fade_fraction = base.wrap_fade_fraction

    def build(self, params, rng):
        chunks, dirs, bonds = [], [], []
        offset = 0
        for i, (scenario, at) in enumerate(self.parts):
            sub = scenario.build(self._part_params_cache[i], rng)
            chunks.append(sub.positions + np.asarray(at, dtype=float))
            dirs.append(sub.directors if sub.directors is not None
                        else np.tile([0.0, 0.0, 1.0], (len(sub.positions), 1)))
            bonds += [(a + offset, b + offset) for a, b in sub.bonds]
            offset += len(sub.positions)
        pos = np.vstack(chunks)
        if self._box_side is not None:
            box = Box.cube(self._box_side, self._periodic)
        else:
            # Snug box around everything, with a margin for the pull reach.
            half = np.abs(pos).max(axis=0) + self._margin
            box = Box(tuple(-half), tuple(half), self._periodic)
        return ScenarioBuild(positions=pos, directors=np.vstack(dirs), box=box,
                             bonds=tuple(bonds))

    def create_commands(self, params, build, seed):
        return ["set group all dipole 0.0 0.0 1.0"]

    def post_control_settle(self, params):
        return self.parts[0][0].post_control_settle(self._part_params_cache[0])

    def housekeeping(self, positions, params, controlled=None):
        return self.parts[0][0].housekeeping(
            positions, self._part_params_cache[0], controlled)


def compose(*parts, box=None, periodic=(False, False, False), margin=2.0):
    """compose(hex_patch(at=(-6, 0, 0)), hex_patch(at=(+6, 0, 0)))

    Each argument is a Scenario (placed at the origin) or a (Scenario, offset)
    pair in either order.
    """
    normalized = []
    for part in parts:
        if isinstance(part, Scenario):
            normalized.append((part, (0.0, 0.0, 0.0)))
        else:
            a, b = part
            normalized.append((a, b) if isinstance(a, Scenario) else (b, a))
    return Composite(normalized, box=box, periodic=periodic, margin=margin)


# --- convenience constructors used in playground files ------------------------
# Lower-case aliases so a playground file reads like a description of the setup.
# `at=` returns the (scenario, offset) pair compose() expects.

def _configured(cls, at, overrides):
    scenario = cls(**overrides)
    return (scenario, at) if at is not None else scenario


def hex_patch(at=None, **overrides):
    return _configured(HexPatch, at, overrides)


def bead_and_partner(at=None, **overrides):
    return _configured(BeadAndPartner, at, overrides)


def hex_sheet(at=None, **overrides):
    return _configured(HexSheet, at, overrides)


def rod_on_sheet(at=None, **overrides):
    return _configured(RodOnSheet, at, overrides)


def random_fill(at=None, **overrides):
    return _configured(RandomFill, at, overrides)


def vesicle_polymer(at=None, **overrides):
    return _configured(VesiclePolymer, at, overrides)
