"""Scenario, parameter and geometry tests -- all pure numpy, no LAMMPS.

Being able to test the geometry and the parameter rules without starting a
simulation is one of the concrete wins of splitting them out of the system class:
none of this was reachable before without building a LAMMPS instance.
"""
import math

import numpy as np
import pytest

from lammps_live.playground.params import Param, ParamSet, Tier, structural
from lammps_live.playground.scenario import (
    BeadAndPartner, HexPatch, HexSheet, RandomFill, align_normal_rate, compose,
    hex_patch,
)
from lammps_live.playground.smoothing import TrajectorySmoother
from lammps_live.playground.state import (
    Box, FrameState, PlacementFailed, build_pairs, hex_lattice_2d, hex_ring_2d,
    principal_normal, random_points_min_separation,
)


# --- geometry -----------------------------------------------------------------

def test_hex_ring_is_centre_first_and_angularly_ordered():
    """Scenarios build bond lists from these indices (spokes 0-k, then the closed
    ring k -> k+1), so centre-first and angle-ordered is a contract."""
    pts = hex_ring_2d(1, a=1.0)
    assert len(pts) == 7
    assert np.allclose(pts[0], (0.0, 0.0))
    # Ring at unit distance, angles 0, 60, ... 300 degrees in order.
    ring = pts[1:]
    assert np.allclose(np.linalg.norm(ring, axis=1), 1.0)
    angles = np.degrees(np.arctan2(ring[:, 1], ring[:, 0])) % 360.0
    assert np.allclose(angles, [0, 60, 120, 180, 240, 300], atol=1e-9)


def test_hex_ring_shell_counts():
    """Shell k of a triangular lattice holds 6k sites."""
    for n_rings in (1, 2, 3):
        pts = hex_ring_2d(n_rings, a=1.0)
        assert len(pts) == 1 + sum(6 * k for k in range(1, n_rings + 1))


def test_hex_lattice_is_centred_and_tiles_its_cell():
    a, n_cols, n_rows = 0.8, 30, 30
    pts = hex_lattice_2d(n_cols, n_rows, a)
    assert len(pts) == n_cols * n_rows
    assert np.allclose(pts.mean(axis=0), 0.0)
    # The cell the sheet scenario derives from these counts must contain them.
    lx, ly = n_cols * a, n_rows * a * math.sqrt(3.0) / 2.0
    assert np.ptp(pts[:, 0]) < lx
    assert np.ptp(pts[:, 1]) < ly


def test_principal_normal_of_a_tilted_plane():
    """The normal-up housekeeping and the tilt observable both rest on this."""
    rng = np.random.default_rng(0)
    pts2d = rng.uniform(-1, 1, size=(200, 2))
    # A plane tilted 30 degrees about x: z = tan(30) * y.
    t = math.radians(30.0)
    pts = np.column_stack([pts2d[:, 0], pts2d[:, 1], math.tan(t) * pts2d[:, 1]])
    n = principal_normal(pts)
    assert n[2] > 0.0                                  # sign-fixed upward
    assert np.degrees(math.acos(min(1.0, n[2]))) == pytest.approx(30.0, abs=1e-6)


def test_align_rate_vanishes_for_a_flat_cloud():
    rng = np.random.default_rng(1)
    flat = np.column_stack([rng.uniform(-1, 1, 100), rng.uniform(-1, 1, 100),
                            np.zeros(100)])
    assert np.allclose(align_normal_rate(flat, 10.0), 0.0, atol=1e-9)


# --- scenarios ----------------------------------------------------------------

def test_hex_patch_build():
    s = HexPatch(n_rings=1, a=1.0, box=6.0)
    build = s.build(s.new_params(), np.random.default_rng(0))
    assert len(build.positions) == 7
    assert np.allclose(build.positions[:, 2], 0.0)         # lies in the xy plane
    assert np.allclose(build.directors, (0.0, 0.0, 1.0))   # directors along +z
    assert build.box.periodic == (False, False, False)
    # 6 spokes + a closed 6-segment ring.
    assert len(build.bonds) == 12


def test_hex_sheet_box_matches_the_lattice():
    """The cell must be sized exactly to the lattice, or the periodic sheet does
    not tile seamlessly -- which is the whole reason it holds itself flat."""
    s = HexSheet(n_cols=10, n_rows=10, a=0.8, z_half=4.0)
    build = s.build(s.new_params(), np.random.default_rng(0))
    assert len(build.positions) == 100
    assert build.box.periodic == (True, True, False)
    assert build.box.lengths[0] == pytest.approx(10 * 0.8)
    assert build.box.lengths[1] == pytest.approx(10 * 0.8 * math.sqrt(3) / 2)


def test_sheet_tracer_marks_one_cluster_of_seven():
    s = HexSheet(n_cols=20, n_rows=20, a=0.8)
    build = s.build(s.new_params(), np.random.default_rng(0))
    b = build.brightness
    assert b is not None
    assert np.count_nonzero(b > 1.0) == 7      # a centre plus six neighbours
    assert b.max() == pytest.approx(2.1)


def test_random_fill_places_the_particles_itself_and_keeps_them_apart():
    """Placement moved out of LAMMPS, and the reason is the clock: `create_atoms
    random ... overlap` is 47 s for the 50,000-bead remote cell and was the whole
    cost of a remote build and of a remote Reset. What must not change is the
    guarantee -- no two particles inside `overlap`, or the first step blows up on a
    pair dropped inside its own hard core.
    """
    s = RandomFill(n=400, box=20.0, overlap=0.9)
    params = s.new_params()
    build = s.build(params, np.random.default_rng(0))
    assert len(build.positions) == 400
    assert build.box.periodic == (True, True, True)
    assert s.atom_creation_commands(params, seed=1234) is None, \
        "the runtime uploads these positions; LAMMPS must not place its own too"
    lo, hi = np.asarray(build.box.lo), np.asarray(build.box.hi)
    assert np.all(build.positions >= lo) and np.all(build.positions <= hi)
    # Minimum-imaged, because the cell is periodic and two points either side of a
    # face are as close as they look.
    d = build.positions[:, None, :] - build.positions[None, :, :]
    lengths = np.asarray(build.box.lengths)
    d -= lengths * np.round(d / lengths)
    r = np.linalg.norm(d, axis=-1)
    r[np.diag_indices(len(r))] = np.inf
    assert r.min() >= 0.9 - 1e-9, f"closest pair is {r.min():.4f} apart"


def test_random_fill_is_reproducible_from_the_seed():
    """Same generator, same box -- a playground with a declared seed has to come up
    the same way twice, and that is now this method's promise rather than LAMMPS'."""
    s = RandomFill(n=200, box=15.0)
    params = s.new_params()
    a = s.build(params, np.random.default_rng(4)).positions
    b = s.build(params, np.random.default_rng(4)).positions
    c = s.build(params, np.random.default_rng(5)).positions
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_random_fill_outline_answers_without_placing_anything():
    """The remote client asks for the cell and the count on every switch to a
    remote playground, and does not want the coordinates -- see Scenario.outline."""
    s = RandomFill(n=50_000, box=86.8)
    params = s.new_params()
    build, natoms = s.outline(params)
    assert natoms == 50_000
    assert len(build.positions) == 0
    assert build.box.lengths[0] == pytest.approx(86.8)


def test_random_fill_says_so_when_the_cell_is_too_full():
    """LAMMPS' own version gives up quietly and creates fewer atoms than asked for.
    A cell that cannot hold what a playground declares is a mistake in the
    playground, and it should read as one."""
    s = RandomFill(n=4000, box=6.0, overlap=1.0)
    with pytest.raises(PlacementFailed) as excinfo:
        s.build(s.new_params(), np.random.default_rng(0))
    assert "too full" in str(excinfo.value)


# --- the two-bead tutorial ----------------------------------------------------

def test_bead_and_partner_puts_the_fixed_one_last():
    """`Control(atom="first")` names the DRIVEN bead, so the partner has to be last
    -- get it the wrong way round and the hand is holding the thing that is supposed
    to be nailed down."""
    s = BeadAndPartner(n_rings=0, a=1.0, partner_x=3.0, partner_z=0.0)
    params = s.new_params()
    build = s.build(params, np.random.default_rng(0))
    assert len(build.positions) == 2
    assert np.allclose(build.positions[0], (0.0, 0.0, 0.0))
    assert np.allclose(build.positions[-1], (3.0, 0.0, 0.0))
    # On the control plane (y = 0), or the leash flattens it there on the first
    # frame and the declared position is a lie.
    assert build.positions[-1][1] == 0.0
    assert s.n_particles(params) == len(build.positions)


def test_the_partner_is_outside_no_integrator_which_is_how_it_is_fixed():
    """Position AND rotation. A `fix setforce` would hold the first and leave the
    director spinning, which is half the point missed: the tilt term is about the
    angle between a director and the bond, so the partner's must stay put."""
    s = BeadAndPartner(n_rings=0, a=1.0)
    params = s.new_params()
    groups = s.group_commands(params, controlled_id=1)
    assert groups == ["group anchor id 2", "group mobile subtract all anchor"]
    integrators = s.integrator_commands(params)
    assert integrators == ["fix integrate mobile nve/sphere update dipole"]
    # It must install one, or the force field's global `all` integrator takes over
    # and the partner moves after all (see PlaygroundSystem._issue_setup).
    assert integrators, "no scenario integrator means the global one integrates all"
    # And the bath must not reach past `mobile` either.
    assert s.thermostat_group() == "mobile"


def test_the_partners_director_is_written_after_the_others():
    """Both commands touch the partner; the second has to win."""
    s = BeadAndPartner(n_rings=0, a=1.0, partner_director=(1.0, 0.0, 0.0))
    params = s.new_params()
    build = s.build(params, np.random.default_rng(0))
    cmds = s.create_commands(params, build, seed=7)
    assert cmds[0] == "set group all dipole 0.0 0.0 1.0"
    assert cmds[1].startswith("set atom 2 dipole 1.0 0.0 0.0")
    assert np.allclose(build.directors[-1], (1.0, 0.0, 0.0))


def test_bead_and_partner_has_no_housekeeping_to_apply():
    """The patch's corrections act on everything but the driven particle, which here
    is the fixed partner -- a force on something that cannot move."""
    s = BeadAndPartner(n_rings=0, a=1.0)
    params = s.new_params()
    pts = s.build(params, np.random.default_rng(0)).positions
    assert s.housekeeping(pts, params, controlled=0) is None


def test_housekeeping_excludes_the_controlled_particle():
    """Its position IS the user's input; a correction force there would fight it."""
    s = HexPatch()
    params = s.new_params()
    pts = s.build(params, np.random.default_rng(0)).positions
    f = s.housekeeping(pts, params, controlled=0)
    assert np.allclose(f[0], 0.0)
    assert not np.allclose(f[1:], 0.0)


def test_compose_stacks_geometry_and_reindexes_bonds():
    scenario = compose(hex_patch(at=(-6, 0, 0)), hex_patch(at=(+6, 0, 0)))
    build = scenario.build(scenario.new_params(), np.random.default_rng(0))
    assert len(build.positions) == 14
    # Each patch centred on its offset.
    assert build.positions[0][0] == pytest.approx(-6.0)
    assert build.positions[7][0] == pytest.approx(+6.0)
    # The second patch's bonds must point into its own atoms, not the first's.
    assert (7, 8) in build.bonds
    assert max(max(b) for b in build.bonds) == 13


def test_wall_commands_cover_only_non_periodic_faces():
    s = HexSheet()
    walls = s.wall_commands(Box.centered(10, 10, 8, periodic=(True, True, False)))
    assert len(walls) == 1
    assert "zlo EDGE" in walls[0] and "zhi EDGE" in walls[0]
    assert "xlo" not in walls[0]
    # A fully periodic cell needs none.
    assert s.wall_commands(Box.cube(10, (True, True, True))) == []


def test_scenario_kwargs_split_into_params_and_attributes():
    s = HexPatch(n_rings=2, timestep=0.001, sim_time_per_frame=0.02)
    assert s.timestep == 0.001              # attribute, not a parameter
    assert s.sim_time_per_frame == 0.02
    assert s.new_params()["n_rings"] == 2    # parameter


# --- parameters ---------------------------------------------------------------

def test_clamp_is_applied_wherever_the_value_is_read():
    """The wc <= rc rule was a helper each old system had to remember to call.
    Declared on the Param, it applies to every read."""
    params = ParamSet.build([
        Param("rc", 2.5, vmin=0.0, vmax=3.0),
        Param("wc", 2.0, vmin=0.0, vmax=3.0, clamp=lambda v, p: min(v, p["rc"])),
    ])
    assert params["wc"] == pytest.approx(2.0)
    params.set("rc", 1.2)
    assert params["wc"] == pytest.approx(1.2)      # clamped by the new rc
    assert params.raw("wc") == pytest.approx(2.0)  # raw value preserved
    assert params.as_dict()["wc"] == pytest.approx(1.2)


def test_set_reports_whether_the_value_changed():
    """The app pushes every slider every frame, so a cheap no-op matters."""
    params = ParamSet.build([Param("k", 1.0, vmin=0.0, vmax=5.0)])
    assert params.set("k", 2.0) is True
    assert params.set("k", 2.0) is False
    assert params.set("nonexistent", 1.0) is False


def test_unknown_override_raises_rather_than_being_ignored():
    """A typo in a preset would otherwise look like the preset having no effect."""
    with pytest.raises(KeyError, match="unknown parameter"):
        ParamSet.build([Param("k", 1.0)], {"kk": 2.0})


def test_structural_params_generate_no_sliders():
    params = ParamSet.build([
        structural("n_beads", 100),
        Param("k", 1.0, "k", 0.0, 5.0),
        Param("rc", 2.5, "rc", 0.0, 3.0, tier=Tier.HOT_RESTYLE),
    ])
    keys = [s.key for s in params.slider_specs()]
    assert keys == ["k", "rc"]          # n_beads is file-time only


def test_slider_range_override():
    params = ParamSet.build([Param("k_splay", 1.0, "k_splay", 0.0, 3.0)])
    wide = params.slider_specs({"k_splay": (0.0, 40.0)})
    assert (wide[0].vmin, wide[0].vmax) == (0.0, 40.0)
    assert params.spec("k_splay").vmax == 3.0     # declaration untouched


# --- pair list ----------------------------------------------------------------

def test_pair_list_respects_minimum_image():
    """Two particles either side of a periodic seam are neighbours."""
    box = Box((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), periodic=(True, True, True))
    pos = np.array([[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]])
    pairs = build_pairs(pos, cutoff=2.0, box=box)
    assert len(pairs) == 1
    assert pairs.r[0] == pytest.approx(1.0)
    # Non-periodic, the same two are 9 apart and not neighbours.
    assert len(build_pairs(pos, cutoff=2.0, box=None)) == 0


def test_pair_list_tolerates_coordinates_a_hair_below_the_lower_bound():
    """LAMMPS reports x = -9e-16 for an atom nominally at 0. The modulo sends that
    to exactly L, which cKDTree rejects outright -- a crash the 2D deposition port
    hit for real."""
    box = Box((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), periodic=(True, False, True))
    pos = np.array([[-9.3e-16, 1.0, 0.0], [0.5, 1.0, 0.0]])
    pairs = build_pairs(pos, cutoff=2.0, box=box)
    assert len(pairs) == 1


def test_touching_selects_a_particles_pairs():
    pos = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [8.0, 0, 0]])
    pairs = build_pairs(pos, cutoff=1.5, box=None)
    assert pairs.touching(0).sum() == 1     # only (0,1)
    assert pairs.touching(1).sum() == 2     # (0,1) and (1,2)
    assert pairs.touching(3).sum() == 0     # isolated


# --- the assembly box's whole-system corrections -------------------------------
# Both are gated on the structure being there, and both have to survive the
# periodic wrap -- which is exactly the part that silently goes wrong.

def _random_fill():
    return RandomFill(), RandomFill().new_params()


def test_centring_finds_a_blob_straddling_the_periodic_seam():
    """The whole point of the circular mean: a cluster sitting ON the seam has
    its parts at both ends of the axis, and an ordinary mean would put its
    "centre" at the far side of the box and push it the wrong way."""
    scen, params = _random_fill()
    box = Box((-10.0, -10.0, -10.0), (10.0, 10.0, 10.0), (True, True, True))
    rng = np.random.default_rng(0)
    # A tight blob centred on x = +10 == -10, i.e. split across the seam.
    blob = rng.normal(scale=0.8, size=(400, 3))
    pts = blob.copy()
    pts[:, 0] = (blob[:, 0] + 10.0 + 10.0) % 20.0 - 10.0
    f = scen.housekeeping(pts, params, box=box)
    assert f is not None
    # It is already at the box's x-seam, which is as far from the centre as an
    # axis goes, so the push must be along x and must be the LARGEST component.
    assert abs(f[0][0]) > abs(f[0][1]) and abs(f[0][0]) > abs(f[0][2])
    # And every particle gets the same push: a translation, not a shear.
    assert np.allclose(f, f[0])


def test_centring_pushes_toward_the_middle_and_eases_off_at_it():
    scen, params = _random_fill()
    box = Box((-10.0,) * 3, (10.0,) * 3, (True, True, True))
    rng = np.random.default_rng(1)
    blob = rng.normal(scale=0.8, size=(400, 3))
    off = scen.housekeeping(blob + np.array([5.0, 0.0, 0.0]), params, box=box)
    assert off[0][0] < 0.0                      # sitting at +x, pushed back to -x
    centred = scen.housekeeping(blob, params, box=box)
    assert abs(centred[0][0]) < 0.1 * abs(off[0][0])


def test_centring_ignores_an_axis_the_particles_fill_uniformly():
    """A lamella is concentrated along its normal and uniform in-plane; there is
    no centre to find in-plane, and trying to invent one would shove the sheet
    around forever."""
    scen, params = _random_fill()
    box = Box((-10.0,) * 3, (10.0,) * 3, (True, True, True))
    rng = np.random.default_rng(2)
    pts = np.column_stack([rng.uniform(-10, 10, 800),
                           rng.uniform(-10, 10, 800),
                           rng.normal(scale=0.5, size=800) + 4.0])
    f = scen.housekeeping(pts, params, box=box)
    assert abs(f[0][2]) > 0.0                   # centred along the normal
    assert abs(f[0][0]) < 1e-3 and abs(f[0][1]) < 1e-3   # left alone in-plane


def test_centring_is_off_for_a_gas():
    scen, params = _random_fill()
    box = Box((-10.0,) * 3, (10.0,) * 3, (True, True, True))
    rng = np.random.default_rng(3)
    pts = rng.uniform(-10, 10, size=(1500, 3))
    assert scen.housekeeping(pts, params, box=box) is None


def test_upright_field_turns_every_director_toward_the_nearer_pole():
    """+n and -n are one orientation, so a director in the lower half must turn
    toward -z rather than the long way round to +z."""
    scen, params = _random_fill()
    n = np.array([[0.0, 0.6, 0.8], [0.0, 0.6, -0.8], [1.0, 0.0, 0.0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    w = scen.director_housekeeping(np.zeros_like(n), n, params)
    for ni, wi in zip(n, w):
        turned = ni + np.cross(wi, ni) * 0.01
        turned /= np.linalg.norm(turned)
        assert abs(turned[2]) > abs(ni[2])       # closer to vertical, either sign


def test_upright_field_is_self_extinguishing():
    """A membrane that is already flat has its directors along z and must feel
    nothing -- that is what lets the field run without a gate."""
    scen, params = _random_fill()
    n = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    w = scen.director_housekeeping(np.zeros_like(n), n, params)
    assert np.allclose(w, 0.0, atol=1e-12)


def test_upright_field_off_at_zero():
    scen = RandomFill()
    params = scen.new_params({"k_upright": 0.0})
    rng = np.random.default_rng(5)
    n = rng.normal(size=(64, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    assert scen.director_housekeeping(np.zeros_like(n), n, params) is None


# --- visual trajectory smoothing ----------------------------------------------

def _state(pos, dirs=None, box=None):
    return FrameState(positions=np.asarray(pos, dtype=float), directors=dirs,
                      box=box or Box.cube(10.0, periodic=(True, True, True)))


def test_smoothing_off_returns_the_state_untouched():
    sm = TrajectorySmoother()
    st = _state([[1.0, 2.0, 3.0]])
    assert sm.apply(st, tau=0.0, dt=0.1) is st
    assert not sm.active


def test_smoothing_first_frame_seeds_rather_than_lagging():
    """A filter that started from zero would drag the whole scene in from the
    origin on the frame it was switched on."""
    sm = TrajectorySmoother()
    st = _state([[1.0, 2.0, 3.0]])
    assert sm.apply(st, tau=1.0, dt=0.1) is st
    assert sm.active


def test_smoothing_lags_toward_the_live_position_at_the_declared_rate():
    sm = TrajectorySmoother()
    tau, dt = 1.0, 0.5
    sm.apply(_state([[0.0, 0.0, 0.0]]), tau, dt)          # seed at the origin
    out = sm.apply(_state([[1.0, 0.0, 0.0]]), tau, dt)
    assert out.positions[0][0] == pytest.approx(1.0 - math.exp(-dt / tau))


def test_smoothing_weight_is_per_sim_time_not_per_frame():
    """Two half-length frames must land where one full-length frame does, or the
    amount of smoothing would depend on the frame rate."""
    one, two = TrajectorySmoother(), TrajectorySmoother()
    target = _state([[1.0, 0.0, 0.0]])
    one.apply(_state([[0.0, 0.0, 0.0]]), tau=1.0, dt=0.4)
    a = one.apply(target, tau=1.0, dt=0.4).positions[0][0]
    two.apply(_state([[0.0, 0.0, 0.0]]), tau=1.0, dt=0.2)
    two.apply(target, tau=1.0, dt=0.2)
    b = two.apply(target, tau=1.0, dt=0.2).positions[0][0]
    assert b == pytest.approx(a, abs=0.02)


def test_smoothing_follows_a_particle_across_a_periodic_seam():
    """THE failure mode: averaging x = +4.9 with x = -4.9 across a periodic wall
    puts the drawn bead in the middle of the box, streaking through everything."""
    box = Box.cube(10.0, periodic=(True, True, True))
    sm = TrajectorySmoother()
    sm.apply(_state([[4.9, 0.0, 0.0]], box=box), tau=1.0, dt=0.5)
    out = sm.apply(_state([[-4.9, 0.0, 0.0]], box=box), tau=1.0, dt=0.5)
    x = out.positions[0][0]
    # It must stay near the seam (either side) and never near the middle.
    assert abs(abs(x) - 5.0) < 0.5, x
    # And inside the cell, so the renderer's wrap/ghost logic still holds.
    assert -5.0 <= x < 5.0


def test_smoothing_keeps_the_controlled_particle_exact():
    """The particle the user is dragging is signal, not jitter -- and the puller
    marker drawn over it comes from the unsmoothed physics."""
    sm = TrajectorySmoother()
    sm.apply(_state([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), tau=1.0, dt=0.5)
    out = sm.apply(_state([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), tau=1.0, dt=0.5,
                   keep_exact=1)
    assert out.positions[1][0] == pytest.approx(1.0)
    assert out.positions[0][0] < 0.9


def test_smoothing_normalizes_smoothed_directors():
    sm = TrajectorySmoother()
    a = np.array([[0.0, 0.0, 1.0]])
    b = np.array([[1.0, 0.0, 0.0]])
    sm.apply(_state([[0.0, 0.0, 0.0]], dirs=a), tau=1.0, dt=0.5)
    out = sm.apply(_state([[0.0, 0.0, 0.0]], dirs=b), tau=1.0, dt=0.5)
    assert np.linalg.norm(out.directors[0]) == pytest.approx(1.0)
    assert 0.0 < out.directors[0][0] < 1.0     # partway from a toward b


def test_smoothing_drops_a_nan_frame_instead_of_poisoning_the_history():
    """An unstable simulation hands back NaN; one NaN in the running average would
    make every later frame NaN too, long after a Reset."""
    sm = TrajectorySmoother()
    sm.apply(_state([[0.0, 0.0, 0.0]]), tau=1.0, dt=0.5)
    bad = _state([[np.nan, 0.0, 0.0]])
    assert sm.apply(bad, tau=1.0, dt=0.5) is bad
    assert not sm.active
    good = _state([[1.0, 0.0, 0.0]])
    assert np.isfinite(sm.apply(good, tau=1.0, dt=0.5).positions).all()
