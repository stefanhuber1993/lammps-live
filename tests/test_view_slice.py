"""The thrust lever's cut through the scene (lammps_live/view_slice.py).

What is worth pinning here is the state machine, not the arithmetic: the lever is
an absolute control with no detent, so *when* it slices is the whole design (see
the module docstring) and every rule in it is one that would otherwise be found
by a demo coming up cut in half.
"""
import numpy as np
import pytest

from lammps_live.view_slice import ViewSlice, _extent

# A 20-sigma cube at the origin, viewed down -y (the scenes' default angle).
BOX = (-10.0, 10.0, -10.0, 10.0, -10.0, 10.0)
FORWARD = np.array([0.0, 1.0, 0.0])


def step(vs, lever, seconds, dt=1.0 / 60.0, forward=FORWARD, box=BOX):
    """Run `seconds` of frames at a held lever position. Returns the last plane."""
    plane = vs.plane
    for _ in range(max(1, int(round(seconds / dt)))):
        plane = vs.update(lever, dt, forward=forward, box_bounds=box)
    return plane


def test_untouched_lever_never_slices():
    """Whatever it reads at startup, a lever nobody has moved cuts nothing."""
    vs = ViewSlice()
    assert step(vs, 0.9, 5.0) is None
    assert vs.progress == 0.0


def test_moving_the_lever_engages_and_fully_closes_within_the_transition():
    vs = ViewSlice()
    step(vs, 0.5, 0.1)                 # the first reading is recorded, not acted on
    assert vs.plane is None
    plane = step(vs, 0.6, vs.transition_seconds + 0.05)
    assert vs.progress == pytest.approx(1.0)
    # 15% of a 20-sigma box, as a half-thickness.
    assert plane.half == pytest.approx(0.5 * 0.15 * 20.0, rel=1e-6)


def test_noise_below_the_touch_epsilon_is_not_a_touch():
    vs = ViewSlice()
    step(vs, 0.5, 0.1)
    # One 7-bit notch of dither, held for longer than the transition.
    for i in range(120):
        vs.update(0.5 + (i % 2) * 0.008, 1 / 60.0, forward=FORWARD, box_bounds=BOX)
    assert vs.plane is None


def test_a_cut_stays_cut_for_as_long_as_the_lever_is_left_where_it_is():
    """There is no idle timeout: the lever is a position, and nobody should have
    to keep touching it to be believed."""
    vs = ViewSlice()
    step(vs, 0.5, 0.1)
    step(vs, 0.6, 1.0)
    assert vs.engaged and vs.progress == pytest.approx(1.0)
    plane = step(vs, 0.6, 30.0)             # half a minute of not touching it
    assert vs.engaged and vs.progress == pytest.approx(1.0)
    assert plane.half == pytest.approx(0.5 * 0.15 * 20.0, rel=1e-6)


def test_both_stops_mean_no_slicing_and_the_middle_means_slicing():
    """The lever's two ends are "off" -- whichever one your hand is nearest,
    shoving it there gives the whole scene back."""
    vs = ViewSlice()
    step(vs, 0.5, 0.1)
    assert vs.demand(0.0) == 0.0
    assert vs.demand(1.0) == 0.0
    assert vs.demand(0.5) == pytest.approx(1.0)
    assert vs.demand(vs.edge_fraction) == pytest.approx(1.0)
    assert vs.demand(1.0 - vs.edge_fraction) == pytest.approx(1.0)
    # ...and the ramp between is monotone, so running the lever into a stop opens
    # the box smoothly rather than snapping it together.
    ramp = [vs.demand(x * vs.edge_fraction) for x in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert ramp == sorted(ramp)
    assert 0.0 < ramp[2] < 1.0


def test_the_lever_at_a_stop_opens_the_box_back_up():
    vs = ViewSlice()
    step(vs, 0.5, 0.1)
    step(vs, 0.6, 1.0)
    assert vs.plane is not None
    step(vs, 1.0, vs.transition_seconds + 0.2)     # hard forward
    assert not vs.engaged
    assert vs.progress == 0.0 and vs.plane is None
    # Back into the band and the slice picks straight back up.
    step(vs, 0.4, vs.transition_seconds + 0.05)
    assert vs.engaged and vs.progress == pytest.approx(1.0)
    step(vs, 0.0, vs.transition_seconds + 0.2)     # hard back: also off
    assert not vs.engaged and vs.plane is None


def test_the_transition_is_monotone_and_cuts_nothing_at_the_open_end():
    """The half-thickness only ever shrinks while closing, and starts wide enough
    that the first frame of a transition cuts nothing -- including the periodic
    images, which reach 1.5 box widths out."""
    vs = ViewSlice()
    step(vs, 0.5, 0.1)
    halves = []
    for _ in range(int(vs.transition_seconds * 60) + 2):
        plane = vs.update(0.8, 1 / 60.0, forward=FORWARD, box_bounds=BOX)
        halves.append(plane.half)
    assert halves == sorted(halves, reverse=True)
    tiled = np.array([[0.0, y, 0.0] for y in np.linspace(-30.0, 30.0, 41)])
    # The first frame is one 60 Hz step in, so allow it to have started closing;
    # what must hold is that it is still wider than everything drawn.
    assert halves[0] > 30.0
    first = ViewSlice()
    step(first, 0.5, 0.1)
    assert first.update(0.8, 1e-9, forward=FORWARD, box_bounds=BOX).mask(tiled) is None


def test_the_lever_sweeps_the_plane_from_the_near_face_to_the_far_one():
    """The sweep is carried by the middle band of the travel -- the ends are the
    "off" positions -- so the plane reaches both faces while still cutting."""
    vs = ViewSlice()
    step(vs, 0.5, 0.1)
    edge = vs.edge_fraction
    near = step(vs, edge, 1.0)
    assert near.center == pytest.approx(-10.0)
    assert vs.engaged, "the near face is still a cut, not the off position"
    far = step(vs, 1.0 - edge, 1.0)
    assert far.center == pytest.approx(10.0)
    assert vs.engaged
    # The normal points AWAY from the eye, so pushing the lever forward pushes
    # the cut into the scene.
    assert np.dot(near.normal, FORWARD) > 0


def test_the_cut_axis_is_cardinal_and_faces_the_camera():
    vs = ViewSlice()
    step(vs, 0.5, 0.1)
    step(vs, 0.8, 1.0)                        # engaged, so there is a plane to read
    for forward, want in (((0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
                          ((-0.9, 0.2, 0.1), (-1.0, 0.0, 0.0)),
                          ((0.1, 0.1, 0.95), (0.0, 0.0, 1.0))):
        vs._axis = None                       # as if freshly engaged from this angle
        plane = vs.update(0.5, 1 / 60.0, forward=np.array(forward), box_bounds=BOX)
        assert plane.normal == want


def test_the_axis_is_held_while_the_camera_moves_round_it():
    """The cut is the viewer's: an orbiting camera shows the chosen section from
    another side, it does not swap it for a different one. Only opening the box
    all the way back up lets the next cut pick a new axis."""
    vs = ViewSlice()
    step(vs, 0.5, 0.1)
    step(vs, 0.8, 1.0)
    assert vs.plane.normal == (0.0, 1.0, 0.0)
    for deg in (50.0, 70.0, 90.0, 180.0):
        swung = np.array([np.sin(np.radians(deg)), np.cos(np.radians(deg)), 0.0])
        assert vs.update(0.8, 1 / 60.0, forward=swung,
                         box_bounds=BOX).normal == (0.0, 1.0, 0.0), deg
    # Lever to a stop: the box closes, and the next cut aims from the new view.
    side = np.array([1.0, 0.0, 0.0])
    for _ in range(120):
        vs.update(1.0, 1 / 60.0, forward=side, box_bounds=BOX)
    assert vs.plane is None
    for _ in range(120):
        vs.update(0.5, 1 / 60.0, forward=side, box_bounds=BOX)
    assert vs.plane.normal == (1.0, 0.0, 0.0)


def test_a_device_with_no_lever_opens_the_box_back_up():
    vs = ViewSlice()
    step(vs, 0.5, 0.1)
    step(vs, 0.8, 1.0)
    assert vs.plane is not None
    step(vs, None, vs.transition_seconds + 0.1)
    assert vs.plane is None


def test_reset_forgets_the_lever():
    vs = ViewSlice()
    step(vs, 0.5, 0.1)
    step(vs, 0.8, 1.0)
    vs.reset()
    # Same position as before the reset: a switch of playground must not carry
    # the cut into a box the lever has not been touched over.
    assert step(vs, 0.8, 2.0) is None


def test_mask_keeps_the_slab_and_nothing_else():
    vs = ViewSlice()
    step(vs, 0.5, 0.1)
    # 0.15 + 0.75 * 0.70 = 0.675 on the lever -> three quarters along the box,
    # i.e. a centre at -10 + 0.75 * 20 = +5 along +y, and a 1.5-sigma half-slab
    # (15% of the 20-sigma box, halved).
    plane = step(vs, vs.edge_fraction + 0.75 * (1.0 - 2 * vs.edge_fraction), 1.0)
    assert plane.center == pytest.approx(5.0)
    assert plane.half == pytest.approx(1.5)
    pts = np.array([[0.0, y, 0.0] for y in (-9.0, 0.0, 3.6, 5.0, 6.4, 9.0)])
    keep = plane.mask(pts)
    assert list(keep) == [False, False, True, True, True, False]


def test_extent_is_measured_over_the_whole_box():
    assert _extent(BOX, np.array([0.0, -1.0, 0.0])) == (-10.0, 10.0)
    assert _extent((0.0, 4.0, -1.0, 1.0, -1.0, 1.0),
                   np.array([1.0, 0.0, 0.0])) == (0.0, 4.0)
