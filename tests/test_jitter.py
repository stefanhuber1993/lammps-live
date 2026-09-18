"""The synthetic rattle: does it put back the right amount of motion?

The whole feature is a number -- how far a bead moves between two DRAWN frames --
and everything in jitter.py exists to make that number come out the same whether
the wire sent every frame or one in three. So the tests here drive it from a
SYNTHETIC diffusing population, where the true per-frame step is known exactly
rather than measured off a LAMMPS run, and check the drawn motion against it.

The rest is the behaviour that keeps it from being noticed in the wrong places:
it must go silent on a frozen or paused scene, it must never be visible to
anything that measures, and it must survive a periodic wall.
"""
import numpy as np
import pytest

from lammps_live.playground import jitter
from lammps_live.playground.jitter import RattleFill
from lammps_live.playground.state import Box, FrameState

STEP = 0.05          # per-component sigma of one 60 Hz frame of diffusion
RENDER_DT = 1.0 / 60


def _walk(n_frames, n_beads=4000, step=STEP, seed=0):
    """A true 60 Hz trajectory: an isotropic random walk, no drift."""
    rng = np.random.default_rng(seed)
    d = rng.normal(scale=step, size=(n_frames, n_beads, 3))
    return np.cumsum(d, axis=0)


def _state(pos, dirs=None, box=None):
    return FrameState(positions=pos, directors=dirs, types=None,
                      ids=np.arange(1, len(pos) + 1), box=box)


def _mean_step(frames):
    return float(np.linalg.norm(np.diff(frames, axis=0), axis=-1).mean())


def _steps_by_arrival(frames, k, offset, arrival):
    """Mean per-bead step over the drawn frames that did / did not receive one.

    `frames[i]` was drawn at global index `offset + i`, and a wire frame landed
    wherever that index is a multiple of k -- so the step INTO frame i is an
    arrival step when (offset + i) % k == 0.
    """
    d = np.linalg.norm(np.diff(frames, axis=0), axis=-1)
    idx = np.arange(1, len(frames)) + offset
    mask = (idx % k == 0)
    chosen = d[mask if arrival else ~mask]
    return float(chosen.mean())


def _quiet_step(frames, k, offset):
    return _steps_by_arrival(frames, k, offset, arrival=False)


def _arrival_step(frames, k, offset):
    return _steps_by_arrival(frames, k, offset, arrival=True)


def _drive(k, strength, n_frames=300, dirs=None, box=None, seed=0):
    """Play a 60 Hz walk through a wire that carries every kth frame.

    Returns (drawn frames, the true 60 Hz frames) over the same span.
    """
    truth = _walk(n_frames, seed=seed)
    fill = RattleFill(seed=seed + 1)
    drawn, held = [], None
    for i, frame in enumerate(truth):
        if i % k == 0:                       # a frame arrived
            held = frame
            fill.observe(_state(held, dirs, box))
        state = _state(held, dirs, box)
        out = fill.apply(state, strength, RENDER_DT, share=1.0 / k)
        drawn.append(np.asarray(out.positions).copy())
    return np.array(drawn), truth


@pytest.mark.parametrize("k", [2, 3, 6])
def test_the_frames_between_arrivals_move_like_a_full_wire(k):
    """The claim the whole file rests on, stated over the right population.

    The frames that would otherwise be FROZEN -- the k-1 in k where no new state
    arrived -- must carry a full frame's worth of motion. That is the quantity
    `_sigma_for` targets, and it is what catches its three separate factors: the
    median-length to per-component-sigma conversion, the sqrt(share) for
    diffusive growth, and the OU's own per-step variance. Getting any one wrong
    moves this ratio by tens of percent; getting the first wrong is the 54% error
    the first draft of this shipped with.

    Note what is NOT asserted here: the mean over ALL drawn frames, which comes
    out ~1.4x the truth and correctly so -- see the arrival-pulse test below.
    """
    drawn, truth = _drive(k, strength=1.0)
    quiet = _quiet_step(drawn[120:], k, offset=120)
    ratio = quiet / _mean_step(truth[120:])
    assert 0.8 < ratio < 1.25, f"k={k}: quiet frames move {ratio:.2f}x the truth"


@pytest.mark.parametrize("k", [2, 3, 6])
def test_the_arrival_frame_still_pulses(k):
    """The honest limit of a zero-latency scheme, asserted so it stays known.

    The real motion is not MISSING from a slow wire -- it is LUMPY. A wire frame
    carries k frames' worth of displacement and lands it all on one drawn frame,
    so filling the quiet frames to full motion necessarily leaves that one moving
    more than the rest. Averaged over everything the picture therefore carries
    ~1.4x the true motion, and that excess IS the residual stutter.

    Removing it means easing toward each arriving frame rather than snapping to
    it, which is latency, which is the thing this design refuses. So the test says
    the pulse is there and roughly how big, rather than pretending otherwise.
    """
    drawn, _truth = _drive(k, strength=1.0)
    quiet = _quiet_step(drawn[120:], k, offset=120)
    arrival = _arrival_step(drawn[120:], k, offset=120)
    assert arrival > 1.5 * quiet, "the arrival pulse has quietly gone away"


def test_the_slider_is_proportional():
    """Liveliness is a multiplier on the amplitude, so the motion it produces
    should track it -- otherwise the number on the panel means nothing."""
    base = _quiet_step(_drive(3, strength=1.0)[0][120:], 3, 120)
    half = _quiet_step(_drive(3, strength=0.5)[0][120:], 3, 120)
    off = _quiet_step(_drive(3, strength=0.0)[0][120:], 3, 120)
    # On a quiet frame the fill is ALL the motion there is, so here the slider
    # really is proportional -- which is the cleanest place to assert it.
    assert off == pytest.approx(0.0, abs=1e-12)
    assert half == pytest.approx(0.5 * base, rel=0.15)


def test_a_frozen_scene_does_not_shimmer():
    """The estimator's real job. A simulation at temperature zero, or paused,
    sends identical frames -- and a scene that is genuinely still must be drawn
    still, or the demo shows thermal motion that is not there."""
    fill = RattleFill(seed=3)
    pos = np.random.default_rng(0).uniform(0, 10, size=(500, 3))
    for _ in range(40):
        fill.observe(_state(pos))
        out = fill.apply(_state(pos), 1.0, RENDER_DT, share=1 / 3)
        assert np.array_equal(np.asarray(out.positions), pos)


def test_it_goes_quiet_when_the_motion_does():
    """Not just exactly-zero: a scene that slows down must calm down with it."""
    fill = RattleFill(seed=4)
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 10, size=(2000, 3))
    moving = []
    for _ in range(60):
        pos = pos + rng.normal(scale=0.15, size=pos.shape)
        fill.observe(_state(pos))
        moving.append(np.asarray(fill.apply(_state(pos), 1.0, RENDER_DT, 1 / 3)
                                 .positions) - pos)
    still = []
    for _ in range(60):                       # the motion stops; frames repeat
        fill.observe(_state(pos))
        still.append(np.asarray(fill.apply(_state(pos), 1.0, RENDER_DT, 1 / 3)
                                .positions) - pos)
    assert np.abs(still[-1]).max() < 0.1 * np.abs(moving[-1]).max()


def test_the_wobble_is_correlated_not_static():
    """An OU, not white noise. Uncorrelated per-frame offsets are what makes a
    synthetic rattle read as television static instead of as temperature."""
    drawn, _ = _drive(3, strength=1.0, n_frames=400)
    off = drawn[200:] - drawn[200:].mean(axis=0)
    a = off[:-1].ravel()
    b = off[1:].ravel()
    rho = float(np.corrcoef(a, b)[0, 1])
    assert rho > 0.3, f"offsets decorrelate in one frame (rho={rho:.2f})"


def test_a_periodic_wall_is_not_a_teleport():
    """Two ways this could go wrong at a seam, and both would be loud: a bead
    that wrapped between frames must not be read as a box-length step by the
    amplitude estimator, and a bead nudged past the wall must come back inside."""
    box = Box.cube(10.0, (True, True, True))
    fill = RattleFill(seed=5)
    rng = np.random.default_rng(2)
    pos = rng.uniform(0.0, 10.0, size=(3000, 3))
    for _ in range(30):
        pos = box.wrap(pos + rng.normal(scale=0.2, size=pos.shape))
        fill.observe(_state(pos, box=box))
    # The estimator saw a scene stepping ~0.2 per axis, not one stepping 10.
    assert fill._step < 1.0
    out = np.asarray(fill.apply(_state(pos, box=box), 2.0, RENDER_DT, 1 / 3)
                     .positions)
    assert out.min() >= box.lo[0] and out.max() <= box.hi[0]


def test_directors_stay_unit_vectors():
    """The rattle perturbs directors additively and renormalizes. A director that
    came back short or long would go straight into the shader's lighting."""
    rng = np.random.default_rng(6)
    fill = RattleFill(seed=7)
    pos = rng.uniform(0, 10, size=(1000, 3))
    dirs = rng.normal(size=(1000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    for _ in range(30):
        pos = pos + rng.normal(scale=0.1, size=pos.shape)
        dirs += rng.normal(scale=0.05, size=dirs.shape)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        fill.observe(_state(pos, dirs))
    out = fill.apply(_state(pos, dirs), 1.0, RENDER_DT, 1 / 3)
    assert np.allclose(np.linalg.norm(out.directors, axis=1), 1.0, atol=1e-6)


def test_a_frame_rate_change_is_not_a_temperature_change():
    """The reason `_sigma_for` inverts the OU's own step variance rather than
    setting sigma directly: the same scene drawn at 30 fps must carry the same
    motion PER SECOND as at 60, not the same motion per frame."""
    def motion_per_second(render_dt):
        truth = _walk(600, seed=11)
        fill = RattleFill(seed=12)
        drawn, held, t = [], None, 0.0
        for i, frame in enumerate(truth):
            if i % 3 == 0:
                held = frame
                fill.observe(_state(held))
            # only draw on the frames this refresh rate would have drawn
            if i * (1.0 / 60) >= t:
                t += render_dt
                drawn.append(np.asarray(
                    fill.apply(_state(held), 1.0, render_dt, share=render_dt * 20)
                    .positions).copy())
        drawn = np.array(drawn[40:])
        return _mean_step(drawn) / render_dt

    fast = motion_per_second(1.0 / 60)
    slow = motion_per_second(1.0 / 30)
    assert 0.6 < slow / fast < 1.6, f"30 Hz moves {slow / fast:.2f}x per second"


def test_the_defaults_are_the_ones_the_slider_offers():
    """The slider is built from these, so a typo here is a silently different
    demo rather than an error."""
    # Zero is a legal default and is the one shipping now -- the effect is off
    # until someone drags the slider up. What this still catches is a default
    # past the top of the slider's own range.
    assert 0.0 <= jitter.DEFAULT_JITTER <= jitter.JITTER_MAX
    assert jitter.TAU_WALL > 1.0 / 60      # correlated over more than one frame


def test_the_rattle_never_reaches_what_measures():
    """The invariant this file shares with smoothing.py, and the one whose failure
    would be worst: a cosmetic filter that wrote through to the state the
    observables and the energy panels read would put a made-up wobble into
    nematic_S and never look like a bug.

    `apply` returns a NEW state and must not touch the arrays it was handed --
    which is what an in-place `+=` on the positions would quietly break.
    """
    rng = np.random.default_rng(9)
    fill = RattleFill(seed=10)
    pos = rng.uniform(0, 10, size=(800, 3))
    dirs = rng.normal(size=(800, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    for _ in range(20):
        pos = pos + rng.normal(scale=0.1, size=pos.shape)
        fill.observe(_state(pos, dirs))

    before_pos, before_dirs = pos.copy(), dirs.copy()
    state = _state(pos, dirs)
    out = fill.apply(state, 2.0, RENDER_DT, share=1 / 3)

    assert np.array_equal(pos, before_pos), "the source positions were written to"
    assert np.array_equal(dirs, before_dirs), "the source directors were written to"
    assert np.array_equal(np.asarray(state.positions), before_pos)
    # ...and it did actually do something, or the assertions above are vacuous.
    assert not np.array_equal(np.asarray(out.positions), before_pos)
