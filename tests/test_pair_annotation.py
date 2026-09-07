"""The two-bead scene's term-by-term readout: the numbers, and that they are the
force field's own.

The annotation is a teaching instrument, so being WRONG here is worse than being
absent -- a number written between two beads is read as authoritative. Its energies
come straight from `ForceField.energy_terms` (already cross-checked against LAMMPS
by verify.py), but the forces and torques are differenced off that expression, and
nothing else in the app does that. So the load-bearing tests below are the two that
compare the difference quotients against what the C++ pair style actually applied:
`test_term_forces_sum_to_the_measured_force` and
`test_term_torques_sum_to_the_measured_torque`.
"""
import math
import os

import numpy as np
import pytest

from lammps_live.forcefields.mesomem import ISO, SPLAY, TILT, MesoMem
from lammps_live.playground.observables import get as get_observable
from lammps_live.playground.pair_probe import (
    FD_STEP, READOUT_EPS, PairAnnotation, PairTerm, probe_pair)
from lammps_live.playground.state import FrameState

# The paper's standard conditions, as the playground runs them.
PARAMS = {"k_tilt": 12.0, "k_splay": 1.0, "zeta": 5.0, "rc": 2.5, "wc": 2.0,
          "splay_symmetry": 0.0}


def pair_state(r, n_driven=(0.0, 0.0, 1.0), n_partner=(0.0, 0.0, 1.0)):
    """The scene's geometry: the partner at +x, the driven bead `r` to its left,
    both directors as given. Same layout as BeadAndPartner builds."""
    return FrameState(positions=np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]]),
                      directors=np.array([n_driven, n_partner], dtype=float))


def probe(r, **kw):
    return probe_pair(MesoMem(), pair_state(r, **kw), PARAMS,
                      plane_normal=(0.0, 1.0, 0.0), torque_scale=2.5)


def term(ann, label):
    return next(t for t in ann.terms if t.label == label)


# --- the geometry it reports ---------------------------------------------------

def test_the_angle_is_the_same_number_the_observable_reports():
    """The connector's header and the HUD line must not be able to disagree: they
    are the same angle, and the annotation says so by matching the observable's
    convention (measured against the vector TO the partner, 0..180)."""
    observable = get_observable("pair_director_angle")
    for director in ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.6, 0.0, -0.8)):
        state = pair_state(1.4, n_driven=director)
        ann = probe(1.4, n_driven=director)
        assert ann.angle_deg == pytest.approx(
            observable(state, None, PARAMS), abs=1e-9)


def test_a_broadside_pair_is_pure_van_der_waals():
    """Both directors along +z with the bond along x: (n . rhat) = 0 and
    (n_i . n_j) = 1, so tilt and splay are exactly zero -- energy, force AND
    torque -- however close the beads get. This is the scene's resting state, and
    the whole point of it is that the two orientational terms sit at zero until
    the user twists."""
    for r in (0.9, 1.2, 1.6, 1.99):
        ann = probe(r)
        for label in (TILT, SPLAY):
            t = term(ann, label)
            assert t.energy == 0.0
            assert t.radial_force == 0.0
            assert t.twist == 0.0
        assert term(ann, ISO).energy != 0.0


def test_the_van_der_waals_term_never_produces_a_torque():
    """It depends on r alone, so its torque is identically zero at every angle --
    the contrast the torque column exists to show."""
    for angle in (0.0, 20.0, 45.0, 70.0, 90.0, 135.0):
        a = math.radians(angle)
        ann = probe(1.2, n_driven=(math.sin(a), 0.0, math.cos(a)))
        assert term(ann, ISO).torque == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
        assert term(ann, ISO).twist == pytest.approx(0.0, abs=1e-9)


def test_out_of_range_is_flat_zero_and_says_so():
    ann = probe(2.6)
    assert not ann.in_range
    assert all(t.energy == 0.0 and t.radial_force == 0.0 for t in ann.terms)
    ann = probe(2.4)
    assert ann.in_range


def test_the_shells_follow_the_live_cutoffs():
    """The rings are drawn from the parameters, so dragging rc moves them -- which
    is the only reason a ring is worth drawing rather than printing.

    wc is deliberately NOT among them: w(r) vanishes there with an essential
    singularity, so just inside it the orientational energies are e^-40 and a ring
    at that radius promised the audience something that then did not happen for
    another 0.2 sigma. Rings drawn at each term's measured onset were tried in its
    place and were worse -- they moved as the director turned -- so what is left is
    the two radii that never move, and the faded rows are what say which term is
    asleep (see Renderer._draw_pair_shells).
    """
    assert probe(1.4).shells == (("sigma", 1.0, 0), ("rc", 2.5, 0))
    widened = probe_pair(MesoMem(), pair_state(1.4), dict(PARAMS, rc=2.9))
    assert widened.shells == (("sigma", 1.0, 0), ("rc", 2.9, 0))


def test_a_missing_partner_is_no_annotation():
    """A scene mid-rebuild, or a remote playground before its first frame."""
    one = FrameState(positions=np.zeros((1, 3)), directors=np.zeros((1, 3)))
    assert probe_pair(MesoMem(), one, PARAMS) is None
    # ...and two particles on top of each other are not a pair either.
    stacked = FrameState(positions=np.zeros((2, 3)),
                         directors=np.array([[0.0, 0.0, 1.0]] * 2))
    assert probe_pair(MesoMem(), stacked, PARAMS) is None


def test_a_force_field_with_no_decomposition_declines():
    class Featureless(MesoMem):
        energy_terms_labels = ()

    assert probe_pair(Featureless(), pair_state(1.4), PARAMS) is None


# --- the difference quotients, against closed form ----------------------------

def iso_force(r, rc=2.5, zeta=5.0):
    """-dU/dr of the attractive branch in closed form, as an independent check on
    the central difference (the branch the hand spends all its time in)."""
    g = 0.5 * math.pi * (r - 1.0) / (rc - 1.0)
    dg = 0.5 * math.pi / (rc - 1.0)
    return -(2.0 * zeta * math.cos(g) ** (2.0 * zeta - 1.0) * math.sin(g) * dg)


@pytest.mark.parametrize("r", [1.1, 1.4, 1.6, 1.8, 2.0, 2.3])
def test_the_radial_force_matches_the_closed_form(r):
    """The tolerance is the central difference's own truncation error at
    FD_STEP -- second order, so a few times 1e-6 on a force of order 1. Two orders
    below the second decimal the callout prints, which is the standard the numbers
    actually have to meet."""
    assert term(probe(r), ISO).radial_force == pytest.approx(iso_force(r),
                                                             abs=1e-4)


def test_attraction_pulls_and_the_core_pushes():
    """The sign convention the display leans on: negative pulls the pair together,
    positive shoves it apart. Sigma is the turning point, and it is where the
    energy is exactly -1 -- the force field's two units in one configuration."""
    assert term(probe(1.4), ISO).radial_force < -1.0        # the strongest pull
    assert term(probe(0.9), ISO).radial_force > 0.0         # the 4-2 core
    # Loosest of the four, and the reason is in the arithmetic: sigma is where the
    # 4-2 core meets the attractive branch. They agree there in value AND slope, but
    # not in curvature, so a central difference straddling the join is first-order
    # in FD_STEP rather than second -- 1e-3 rather than 1e-6. It rounds to +0.00 on
    # the display either way, which is what the reading has to get right.
    assert term(probe(1.0), ISO).radial_force == pytest.approx(0.0, abs=2e-3)
    assert term(probe(1.0), ISO).energy == pytest.approx(-1.0, abs=1e-12)


def test_the_tilt_torque_restores_toward_broadside():
    """Tilt penalises (n . rhat), so a director tipped toward its neighbour is
    twisted back -- and the sign has to be the restoring one, or the arc drawn
    beside the number points the wrong way round."""
    a = math.radians(45.0)
    ann = probe(1.2, n_driven=(math.sin(a), 0.0, math.cos(a)))
    tilt = term(ann, TILT)
    assert tilt.energy > 0.0
    # Tipped toward +x about +y, so the restoring torque is about -y.
    assert tilt.twist < 0.0
    # ...and tipping the other way flips it, with the same magnitude.
    other = term(probe(1.2, n_driven=(-math.sin(a), 0.0, math.cos(a))), TILT)
    assert other.twist == pytest.approx(-tilt.twist, rel=1e-6)


def test_the_twist_is_the_component_about_the_plane_normal():
    """One number for the display, and it is the component about the axis the twist
    control turns the director about. With no plane declared there is no such axis
    and the magnitude is reported instead."""
    a = math.radians(45.0)
    n = (math.sin(a), 0.0, math.cos(a))
    ann = probe(1.2, n_driven=n)
    tilt = term(ann, TILT)
    assert tilt.twist == pytest.approx(tilt.torque[1], abs=1e-12)
    free = probe_pair(MesoMem(), pair_state(1.2, n_driven=n), PARAMS)
    assert term(free, TILT).twist == pytest.approx(
        float(np.linalg.norm(tilt.torque)), rel=1e-9)


def test_contact_leaves_the_radial_force_out_rather_than_dividing_by_zero():
    """A step through r = 0 is not a separation. The energies are still reported."""
    ann = probe(0.5 * FD_STEP)
    assert all(t.radial_force == 0.0 for t in ann.terms)
    assert term(ann, ISO).energy != 0.0


# --- against the running pair style -------------------------------------------

pytest.importorskip("lammps")


@pytest.fixture(scope="module")
def bead():
    from lammps_live.playground import registry

    system = registry.build("mesomem_bead")
    yield system
    system.close()


def place(system, r, angle_deg):
    """Put the driven bead `r` from the partner with its director at `angle_deg`
    from +z, and recompute forces without integrating."""
    partner = np.array(system.lmp.numpy.extract_atom("x")[:2], dtype=float)[1]
    pos, mu = system.lmp.numpy.extract_atom("x"), system.lmp.numpy.extract_atom("mu")
    pos[0][0], pos[0][1], pos[0][2] = partner[0] - r, partner[1], partner[2]
    a = math.radians(angle_deg)
    mu[0][0], mu[0][1], mu[0][2] = math.sin(a), 0.0, math.cos(a)
    # At rest, so that what the mode reports is the PAIR STYLE's force and nothing
    # else: `interaction_force` reconstructs it as "total minus the forces we
    # applied ourselves", and the viscous damping is one of those -- a bead left
    # drifting at 1e-3 sigma/tau carries a 4e-3 damping term into the comparison,
    # which is larger than everything this test is trying to resolve.
    v = system.lmp.numpy.extract_atom("v")
    v[0][0] = v[0][1] = v[0][2] = 0.0
    system.lmp.command("run 0 post no")
    # The frame state is cached once per frame; nothing here has stepped, so say
    # so explicitly rather than reading the state from before the placement.
    system._frame += 1
    return system.get_pair_annotation()


@pytest.mark.parametrize("r,angle", [(1.8, 90.0), (1.6, 45.0), (1.4, 20.0),
                                     (1.2, 45.0), (1.0, 70.0), (0.95, 0.0)])
def test_term_forces_sum_to_the_measured_force(bead, r, angle):
    """The per-term radial forces must add up to the radial force the C++ pair
    style applied. This is what makes the force column a reading rather than a
    plausible-looking number: the terms are differenced off the Python energy
    expression, and the total is measured on the atom."""
    ann = place(bead, r, angle)
    # The mode reports the reaction on the control plane's two axes (x, z) with
    # the drive and the damping already taken out -- and the bond is along x, so
    # its x component IS the radial force, measured inward.
    measured = -float(bead.get_interaction_force()[0])
    assert ann.total_radial_force == pytest.approx(measured, abs=2e-3)


@pytest.mark.parametrize("r,angle", [(1.6, 20.0), (1.4, 45.0), (1.2, 70.0),
                                     (1.0, 45.0)])
def test_term_torques_sum_to_the_measured_torque(bead, r, angle):
    """...and the same for the torque column, WITH ONE CORRECTION, which is a bug
    in the pair style rather than in this.

    The C++ applies the splay torque with the sign of +d(U_splay)/d(n_i . n_j)
    rather than minus it (pair_membrane_sillano_v2.cpp, section G: `splay_pref *
    ni_x_nj`, where splay_pref = ks * dev * dsym and dev = n_i.n_j - 1 <= 0). A
    torque is minus the energy gradient, so as written the splay term twists
    neighbouring directors APART instead of into alignment -- it is anti-restoring.
    The tilt torque beside it has the correct sign, and the ENERGY expression is
    verified against LAMMPS to machine precision (verify.py), so the discrepancy
    is in that one term's torque and nowhere else.

    The correction is written out here rather than hidden, so that this test FAILS
    LOUDLY if the C++ sign is ever fixed -- at which point the right change is to
    delete the correction, not to widen the tolerance.
    """
    ann = place(bead, r, angle)
    measured = float(np.array(bead.lmp.numpy.extract_atom("torque")[:2],
                              dtype=float)[0][1])
    splay = term(ann, SPLAY).twist
    assert ann.total_twist - 2.0 * splay == pytest.approx(measured, abs=3e-3)
    # The uncorrected sum differs by exactly that, which is the statement above.
    assert ann.total_twist - measured == pytest.approx(2.0 * splay, abs=3e-3)


def test_the_annotation_replaces_the_energy_panels(bead):
    """Both panels go quiet on a pair-annotated playground: they and the connector
    would be three copies of one reading."""
    assert bead.get_pair_annotation() is not None
    assert bead.get_potential_terms() is None
    assert bead.get_total_potential_terms() is None


def test_a_playground_that_did_not_ask_for_it_keeps_its_panels():
    from lammps_live.playground import registry

    patch = registry.build("mesomem_patch")
    try:
        assert not patch.playground.pair_annotation
        assert patch.get_pair_annotation() is None
        assert patch.get_potential_terms() is not None
        assert patch.get_total_potential_terms() is not None
    finally:
        patch.close()


def test_the_scene_carries_its_own_feedback_calibration():
    """One pair's forces are a tenth of a membrane's, so the shared reduced-unit
    profile left the whole interaction under the stiffness threshold. The arrow
    scale rides on the same knee, so this is also what makes the red reaction arrow
    and the connector's force glyphs read at this scene's scale."""
    from lammps_live.playground.spec import REDUCED_UNIT_FEEDBACK
    from lammps_live.playgrounds.mesomem_bead import PLAYGROUND

    profile = PLAYGROUND.force_feedback
    assert profile is not REDUCED_UNIT_FEEDBACK
    assert profile.ff_knee < REDUCED_UNIT_FEEDBACK.ff_knee
    assert profile.stiffness_threshold < REDUCED_UNIT_FEEDBACK.stiffness_threshold
    # First contact -- 0.2 at r = 1.8 -- has to firm the spring up at all.
    from lammps_live.forcefeedback import contact_fraction
    assert contact_fraction(0.2, 0.0, profile) > 0.15
    assert contact_fraction(0.2, 0.0, REDUCED_UNIT_FEEDBACK) == 0.0


# --- and that the renderer draws it -------------------------------------------

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


@pytest.fixture(scope="module")
def renderer():
    import pygame

    from lammps_live.ui.renderer import Renderer

    pygame.display.init()
    pygame.font.init()
    yield Renderer((900, 700))
    pygame.display.quit()


ANNOTATION = PairAnnotation(
    i=0, j=1, r=1.4, angle_deg=45.0,
    terms=(PairTerm(ISO, -0.41, -1.89, (0.0, 0.0, 0.0), 0.0),
           PairTerm(TILT, 0.23, 1.37, (0.0, -0.45, 0.0), -0.45),
           PairTerm(SPLAY, 0.01, 0.03, (0.0, -0.02, 0.0), -0.02)),
    shells=(("sigma", 1.0, 0), ("rc", 2.5, 0)),
    plane_normal=(0.0, 1.0, 0.0), torque_scale=2.5)


def scene(renderer):
    """A camera, two beads on the control plane, and their projection."""
    from lammps_live.ui.camera import Camera3D

    camera = Camera3D(eye=(0.0, -8.0, 3.0), target=(0.7, 0.0, 0.0),
                      up=(0.0, 0.0, 1.0), fov_deg=45.0,
                      viewport_w=renderer.sim_width,
                      viewport_h=renderer.window_size[1])
    pts = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    screen, _, scale = camera.project(pts)
    return camera, pts, screen, np.maximum(0.5 * scale, 2.0)


def drawn_pixels(renderer, **kw):
    """How many pixels the annotation changed, drawn over a flat ground."""
    import pygame

    from lammps_live.playgrounds.mesomem_bead import PLAYGROUND, STYLE

    renderer._scene_style = STYLE
    renderer.screen.fill(STYLE.background)
    before = pygame.image.tostring(renderer.screen, "RGB")
    camera, pts, screen, radii = scene(renderer)
    renderer._draw_pair_annotation(camera, ANNOTATION, pts, screen, radii,
                                   _Spec(PLAYGROUND.force_feedback), **kw)
    after = pygame.image.tostring(renderer.screen, "RGB")
    return sum(1 for a, b in zip(before, after) if a != b)


class _Spec:
    """The two fields the annotation reads off the SystemSpec."""

    def __init__(self, force_feedback):
        self.force_feedback = force_feedback


def test_the_renderer_draws_the_annotation(renderer):
    assert drawn_pixels(renderer, shown=None) > 2000


def test_a_bead_cut_away_takes_the_annotation_with_it(renderer):
    """The view slice can remove either bead. A table of numbers about a pair that
    is not on screen, with a leader line pointing at where it would have been, is
    worse than nothing -- and the same two numbers are still on the HUD."""
    assert drawn_pixels(renderer, shown=np.array([True, False])) == 0
    assert drawn_pixels(renderer, shown=np.array([False, True])) == 0
    assert drawn_pixels(renderer, shown=np.array([True, True])) > 2000


def test_the_fade_threshold_is_the_one_the_callout_rounds_at():
    """One constant, two readers. The callout prints two decimals and fades a row
    whose every column reads "+0.00"; if the renderer kept its own copy of that
    threshold, a row could be faded while showing a digit."""
    from lammps_live.ui.renderer import Renderer

    assert Renderer._pair_number(READOUT_EPS * 0.99) == "+0.00"
    assert Renderer._pair_number(READOUT_EPS * 1.01) != "+0.00"
    assert Renderer._pair_number(-READOUT_EPS * 0.99) == "+0.00"
