"""Cutting the scene open with the joystick's thrust lever.

An opaque scene hides its own inside. A vesicle with a polymer coiled in it is
the extreme case -- the membrane is a closed, densely packed shell, so from the
outside the thing the playground exists to show is simply not visible -- but the
same is true of any packed box of beads, which is why this lives here and not in
one playground file.

The cut is a SLAB: keep only the beads within `THICKNESS_FRACTION` of the box's
width of a plane, and slide that plane through the box with the lever. The slab's
normal is the CARDINAL world axis the camera is closest to looking along when the
cut engages, so the face you are left looking at is a clean section through the
cell rather than an oblique one -- and it then STAYS that axis while the camera
moves round it (see `_cardinal` and REAIM_DEGREES).

    RenderStyle.section_min is the other cut in this codebase and a different
    thing: a fixed half-space in a fixed world axis, declared by a playground
    because that scene is ALWAYS drawn in section (the rod, edge-on). This one is
    live, camera-relative, and belongs to the person at the controls. They
    compose -- the renderer draws the intersection.

THE LEVER IS A POSITION, AND THE POSITION IS THE WHOLE ANSWER. It is an absolute
control with no detent, so the one thing it must never do is mean something
different from what it looks like it means. The two STOPS are therefore "off":

  * lever hard back, or hard forward -> NO slicing, the whole box on screen.
  * anywhere in between -> the slab is engaged, and the lever's position within
    that band is where the plane sits, near face to far.
  * the last `EDGE_FRACTION` of travel at each end is the transition: the slab
    widens smoothly back out to the whole box as the lever runs into its stop,
    so there is no frame where the scene snaps together.

Two stops rather than one because the lever has two and neither is privileged:
whichever end your hand is nearest, shoving it there gives you the whole scene
back. Nothing here is timed -- there is no idle timeout that opens the box up
under you (there was, and it made the lever a control you had to keep touching to
be believed). Cut in, let go, and it stays cut until you move it.

  * UNTOUCHED SINCE STARTUP -> no slicing, whatever the lever reads, because
    "whatever it reads" is wherever the last session happened to leave it. The
    first value seen is only recorded, never acted on. This is the one piece of
    state left: it is what stops a demo coming up sliced in half before anybody
    has touched anything.
"""
import math
from dataclasses import dataclass

import numpy as np

# The slab's thickness once fully engaged, as a fraction of the box's width along
# the cutting axis. 15% of a 100-sigma cell is fifteen bead diameters: a slab
# with some depth in it rather than a single sheet of beads, which is what makes
# the cut face read as a solid surface you are looking into. It started at 5%,
# which is a genuine section but leaves a closed membrane as a thin ring with
# nothing behind it.
THICKNESS_FRACTION = 0.15
# How long the slab takes to close from "whole box" to that thickness, and to
# open back up again. Short enough to feel like a response to the lever, long
# enough to read as one object opening rather than beads vanishing. It is a rate
# LIMIT on top of the lever, not a timer: a slow push is followed exactly, and
# only a shove is smoothed.
TRANSITION_SECONDS = 0.5
# How much of the lever's travel at EACH END means "no slicing", and over which
# the slab opens back out. Wide enough that shoving the lever to a stop without
# looking is reliably "off" -- the point of putting it at the stops -- and narrow
# enough that the remaining 70% of the travel still sweeps the plane across the
# whole box at a usable resolution.
EDGE_FRACTION = 0.15
# How much the lever has to move to count as touched. The device reports it as 7
# bits, so one notch is 1/127 ~ 0.008; this is a few notches -- past the last
# bit's dither, well inside a deliberate nudge.
TOUCH_EPSILON = 0.02
# The slab's half-thickness at the fully-open end of the transition, as a multiple
# of the box width. Not simply "infinity": the transition interpolates between
# this and the target, so it has to be a real number, and it has to be wide enough
# that nothing is cut at the open end -- including a periodically tiled scene,
# whose copies reach 1.5 box widths out from the centre (see
# RenderStyle.periodic_images).
OPEN_FACTOR = 3.0
# How far off the view direction the cutting axis is allowed to drift before it is
# re-picked, in degrees. 180 means NEVER: the axis is picked once, from the view,
# when the cut engages, and then held for as long as the slab is on screen
# (it is only forgotten once the lever has opened the box all the way back up).
#
# It used to be 55, which re-aimed the cut whenever the camera swung that far off
# it. On the orbiting cube scenes that meant the plane you had chosen to look at
# was replaced under you by the turntable -- the section you were studying turned
# into a different section with nobody touching the lever. A cut is something the
# viewer set; the camera moving round it should show it from another side, not
# change it. The mechanism is kept (a smaller value restores the old re-aiming).
REAIM_DEGREES = 180.0

_AXES = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
         np.array([0.0, 0.0, 1.0]))


def _smoothstep(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass(frozen=True)
class SlicePlane:
    """One frame's cut: keep the beads within `half` of `center` along `normal`.

    A value object rather than a live handle, so the renderer is handed a fact
    about this frame and cannot accidentally advance the animation by drawing.
    """

    normal: tuple
    center: float
    half: float

    def mask(self, points):
        """Boolean keep-mask over world points (N, 3), or None if it cuts nothing.

        None rather than an all-true array on purpose: the caller can then skip
        the whole indexing pass, which at 50k beads and a dozen per-bead channels
        is worth having on every frame the slice is not engaged.
        """
        if not np.isfinite(self.half):
            return None
        pts = np.asarray(points, dtype=float)
        if not len(pts):
            return None
        d = pts @ np.asarray(self.normal, dtype=float)
        keep = np.abs(d - self.center) <= self.half
        return None if keep.all() else keep


class ViewSlice:
    """The lever's state, and the slab it currently asks for.

    Device-free and geometry-free: it is fed a lever reading, a view direction
    and the box, and hands back a SlicePlane. That keeps the two questions worth
    testing -- when does it engage, and where does the plane land -- answerable
    without a joystick or a renderer (tests/test_view_slice.py).
    """

    def __init__(self, thickness_fraction=THICKNESS_FRACTION,
                 transition_seconds=TRANSITION_SECONDS,
                 edge_fraction=EDGE_FRACTION, touch_epsilon=TOUCH_EPSILON,
                 open_factor=OPEN_FACTOR, reaim_degrees=REAIM_DEGREES):
        self.thickness_fraction = thickness_fraction
        self.transition_seconds = transition_seconds
        self.edge_fraction = edge_fraction
        self.touch_epsilon = touch_epsilon
        self.open_factor = open_factor
        self.reaim_cos = math.cos(math.radians(reaim_degrees))
        # None until the first reading arrives -- which is exactly "the lever has
        # not been touched since this session started", since the device only
        # reports on change.
        self._lever = None
        self._touched = False
        # How much cut the lever's CURRENT position asks for, 0..1 -- a function
        # of that position and nothing else (see `demand`). `_progress` is what is
        # actually on screen, chasing it at the transition's rate.
        self._demand = 0.0
        self._progress = 0.0      # 0 = whole box, 1 = fully sliced
        self._axis = None         # the cutting normal, a signed cardinal axis
        self._plane = None

    # ---- the lever's position -> what it asks for ----------------------------

    def demand(self, lever):
        """How much cut a lever position asks for, 0 (whole box) .. 1 (full slab).

        1 across the middle of the travel, easing to 0 over the last
        `edge_fraction` at EACH end -- so both stops mean "no slicing" and the
        scene comes back smoothly as the lever runs into either of them.
        """
        if lever is None:
            return 0.0
        edge = self.edge_fraction
        if edge <= 0.0:
            return 1.0
        lever = min(max(float(lever), 0.0), 1.0)
        return _smoothstep(min(lever, 1.0 - lever) / edge)

    def sweep(self, lever):
        """Where along the box the plane sits, 0 (near face) .. 1 (far face).

        The middle band of the travel -- everything the two "off" ends leave --
        stretched over the whole box, so no part of the sweep is unreachable and
        the plane arrives at a face exactly as the slab starts opening back up.
        """
        if lever is None:
            return 0.0
        edge = self.edge_fraction
        span = max(1.0 - 2.0 * edge, 1e-6)
        return min(max((float(lever) - edge) / span, 0.0), 1.0)

    # ---- state ---------------------------------------------------------------

    @property
    def engaged(self):
        """Whether the lever's position is asking for a cut (before easing)."""
        return self._demand > 0.0

    @property
    def progress(self):
        """0 = whole box on screen, 1 = the slab at its full thinness."""
        return self._progress

    @property
    def plane(self):
        """This frame's SlicePlane, or None while the whole box is shown."""
        return self._plane

    def reset(self):
        """Forget the lever, as if the session had just started. Called on a
        playground switch: the box it was cutting is gone, and coming up sliced
        into a scene you have not touched the lever for is the same surprise the
        untouched-at-startup rule exists to avoid."""
        self._lever = None
        self._touched = False
        self._demand = 0.0
        self._progress = 0.0
        self._axis = None
        self._plane = None

    # ---- per frame -----------------------------------------------------------

    def update(self, lever, dt, forward=None, box_bounds=None):
        """One frame. `lever` is 0..1 (or None on a device with no lever),
        `forward` the camera's view direction, `box_bounds` the cell as
        (xlo, xhi, ylo, yhi, zlo, zhi). Returns this frame's plane, or None.

        Nothing here is timed: the lever's position is read every frame and
        `demand` turns it into how much cut it is asking for. The only history is
        whether it has been touched at all -- see the module docstring.
        """
        if lever is None:
            # No lever on this input device -- open back up rather than freezing
            # whatever the last joystick session left behind.
            self._touched = False
        elif self._lever is None:
            self._lever = float(lever)          # recorded, deliberately not acted on
        elif abs(float(lever) - self._lever) > self.touch_epsilon:
            self._lever = float(lever)
            self._touched = True
        self._demand = self.demand(self._lever) if self._touched else 0.0

        # The lever leads, the picture follows at a bounded rate: a slow push is
        # tracked exactly (the step is bigger than any one frame's change), and
        # only a shove is smoothed into the half second the transition is worth.
        step = dt / max(self.transition_seconds, 1e-6)
        if self._progress < self._demand:
            self._progress = min(self._demand, self._progress + step)
        elif self._progress > self._demand:
            self._progress = max(self._demand, self._progress - step)

        self._plane = self._resolve(forward, box_bounds)
        return self._plane

    # ---- geometry ------------------------------------------------------------

    def _resolve(self, forward, box_bounds):
        if self._progress <= 0.0 or box_bounds is None or forward is None:
            self._axis = None
            return None
        axis = self._cardinal(np.asarray(forward, dtype=float))
        self._axis = axis
        lo, hi = _extent(box_bounds, axis)
        span = max(hi - lo, 1e-9)
        # The lever's travel maps to the whole of the box along the cutting axis,
        # near face to far: pushed forward, the plane moves away from the eye,
        # which is the direction the axis was signed to point in. Only the middle
        # band of the travel carries the sweep -- the ends are "off" (see
        # `sweep`).
        center = lo + self.sweep(self._lever) * span
        target_half = 0.5 * self.thickness_fraction * span
        open_half = self.open_factor * span
        # Geometric, not linear, between the open and closed thicknesses: the slab
        # then thins at a constant PROPORTIONAL rate, which is what reads as a
        # steady closing. A linear ramp spends most of the half second in a slab
        # far wider than the cell and then snaps shut at the end.
        s = _smoothstep(self._progress)
        half = open_half * (target_half / open_half) ** s
        return SlicePlane(normal=tuple(axis), center=center, half=half)

    def _cardinal(self, forward):
        """The signed cardinal axis to cut along, pointing away from the eye.

        Sticky: the axis already in use is kept until the view has swung more
        than REAIM_DEGREES off it (see the constant), so an orbiting camera
        re-aims the cut once rather than flickering between two axes as it
        crosses the halfway angle.
        """
        n = np.linalg.norm(forward)
        if n < 1e-9:
            return self._axis if self._axis is not None else _AXES[1].copy()
        f = forward / n
        if self._axis is not None and float(self._axis @ f) >= self.reaim_cos:
            return self._axis
        k = int(np.argmax(np.abs(f)))
        return _AXES[k] * (1.0 if f[k] >= 0.0 else -1.0)


def _extent(box_bounds, axis):
    """(min, max) of the box's corners projected on `axis`."""
    xlo, xhi, ylo, yhi, zlo, zhi = (float(v) for v in box_bounds)
    lo = hi = None
    for x in (xlo, xhi):
        for y in (ylo, yhi):
            for z in (zlo, zhi):
                d = float(axis @ np.array([x, y, z]))
                lo = d if lo is None else min(lo, d)
                hi = d if hi is None else max(hi, d)
    return lo, hi
