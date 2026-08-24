"""The rod-on-membrane playground: geometry, commands, energy and drawn body.

The energy test here is the one that matters. `rod_lj`'s potential is a 12-6 LJ at
the distance from a bead to the closest point on the rod's AXIS SEGMENT, with
sigma_eff = r_mem + r_rod -- so the Python expression that drives the energy
panels has to reproduce a closest-point-on-segment clamp, not a centre-to-centre
distance. `reference_adhesion` below is that formula written out as a scalar loop,
transcribed from the C++ (`pair_rod_lj.cpp`'s compute()); if the vectorized
version ever drifts from it, this fails. The vectorized one is separately checked
against LAMMPS itself in test_runtime.py.

Everything except the last two tests is pure numpy -- no LAMMPS instance.
"""
import math

import numpy as np
import pytest

from lammps_live.forcefields.mesomem import ISO, SPLAY, TILT
from lammps_live.forcefields.mesomem_rod import ADHESION, R_MEM, MesoMemRod
from lammps_live.playground import registry
from lammps_live.playground.observables import analysis_pairs
from lammps_live.playground.observables import get as get_observable
from lammps_live.playground.scenario import RodOnSheet
from lammps_live.playground.state import Box, FrameState, build_pairs
from lammps_live.playground.verify import verify_system


def reference_adhesion(x_rod, x_bead, axis, length, sigma_eff, eps, cut):
    """Scalar transcription of pair_rod_lj.cpp's compute(), for one pair."""
    a = x_rod - 0.5 * length * axis
    ab = length * axis
    t = float((x_bead - a) @ ab) / (float(ab @ ab) + 1.0e-30)
    t = min(max(t, 0.0), 1.0)
    closest = a + t * ab
    r = float(np.linalg.norm(closest - x_bead))
    if r >= cut or r < 1e-15:
        return 0.0
    sr6 = (sigma_eff / r) ** 6
    return 4.0 * eps * (sr6 * sr6 - sr6)


def rod_state(rod_pos, bead_positions, axis=(1.0, 0.0, 0.0), box=None):
    """A FrameState of `bead_positions` (type 1) plus one rod (type 2, last)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    beads = np.asarray(bead_positions, dtype=float).reshape(-1, 3)
    positions = np.vstack([beads, np.asarray(rod_pos, dtype=float)[None, :]])
    directors = np.vstack([np.tile([0.0, 0.0, 1.0], (len(beads), 1)), axis[None, :]])
    types = np.concatenate([np.ones(len(beads), dtype=int), [2]])
    return FrameState(positions=positions, directors=directors, types=types,
                      ids=np.arange(1, len(positions) + 1), box=box)


# --- the scenario -------------------------------------------------------------

def test_rod_is_the_last_particle_and_lies_in_the_control_plane():
    """`Control(atom="last")` names the rod, so the rod has to be last -- and its
    axis has to be in the control plane, or the first frame's constraint flattens
    it into one and the declared starting orientation is a lie."""
    scenario = RodOnSheet(n_cols=6, n_rows=6, rod_height=3.5,
                          rod_axis=(1.0, 0.0, 0.0), tracer_fraction=None)
    build = scenario.build(scenario.new_params(), np.random.default_rng(0))

    assert build.types[-1] == 2
    assert (build.types[:-1] == 1).all()
    assert len(build.positions) == 6 * 6 + 1
    assert build.positions[-1] == pytest.approx([0.0, 0.0, 3.5])
    # In the xz control plane means no y component.
    assert build.directors[-1] == pytest.approx([1.0, 0.0, 0.0])
    # The membrane's own directors are untouched.
    assert build.directors[:-1] == pytest.approx(np.tile([0, 0, 1], (36, 1)))


def test_rod_axis_is_normalized_and_can_stand_on_end():
    scenario = RodOnSheet(n_cols=4, n_rows=4, rod_axis=(0.0, 0.0, 3.0),
                          tracer_fraction=None)
    build = scenario.build(scenario.new_params(), np.random.default_rng(0))
    assert build.directors[-1] == pytest.approx([0.0, 0.0, 1.0])


def test_the_rod_gets_its_own_dipole_command_after_the_membranes():
    """Order is load-bearing: the first command sets every director to +z, and the
    second has to come after it to give the rod its axis back."""
    scenario = RodOnSheet(rod_axis=(1.0, 0.0, 0.0))
    cmds = scenario.create_commands(scenario.new_params(), None, seed=1)
    assert cmds[0] == "set type 1 dipole 0.0 0.0 1.0"
    assert cmds[1].startswith("set type 2 dipole 1.0 0.0")


def test_the_shipped_playground_can_actually_reach_the_membrane():
    """The three numbers that have to agree -- where the rod starts, how far the
    leash reaches, and how far the rod's interaction reaches -- live in three
    different declarations, and getting them wrong produces a demo that looks
    fine and cannot touch anything. See RodOnSheet.verify_reach."""
    playground = registry.load("mesomem_rod")
    force_field = MesoMemRod(**playground.force_field_options)
    params = force_field.new_params(playground.resolved_params())
    problems = playground.scenario.verify_reach(playground.effective_control(),
                                               force_field.rod_cutoff(params))
    assert problems == [], "; ".join(problems)


def test_verify_reach_catches_a_rod_placed_outside_its_leash():
    from lammps_live.playground.spec import Control
    scenario = RodOnSheet(rod_height=9.0)
    problems = scenario.verify_reach(Control(plane="xz", leash=(7.0, 5.0)), 3.24)
    assert any("outside the leash" in p for p in problems)


def test_lifting_the_membrane_buys_depth_and_is_checked_against_the_leash():
    """`plane_z` moves the membrane up the container so the invagination has
    somewhere to go.

    The leash and the frame are centred on the ORIGIN rather than on the membrane,
    so raising the plane converts headroom the rod never uses into depth below it
    -- for free, since the container does not get any deeper. What it must not do
    is push the rod's own starting height out through the top of the leash, and
    that is the thing `verify_reach` is here to catch.
    """
    from lammps_live.playground.spec import Control
    control = Control(plane="xz", leash=(14.0, 10.0))

    lifted = RodOnSheet(plane_z=5.0, rod_height=3.5)
    build = lifted.build(lifted.new_params(), np.random.default_rng(0))
    # The membrane is up there, and the rod is above it by its own clearance --
    # `rod_height` is a clearance, not an absolute z.
    assert build.positions[:-1, 2] == pytest.approx(5.0)
    assert build.positions[-1, 2] == pytest.approx(8.5)
    # The container is unchanged: this is a reallocation, not a bigger box.
    flat = RodOnSheet(plane_z=0.0, rod_height=3.5)
    assert build.box.lengths[2] == flat.build(
        flat.new_params(), np.random.default_rng(0)).box.lengths[2]
    # Which is the point: 15 sigma of travel below the membrane instead of 10.
    assert lifted.verify_reach(control, 3.24) == []

    # Lift it too far and the rod starts outside the leash's ceiling.
    too_high = RodOnSheet(plane_z=8.0, rod_height=3.5)
    assert any("outside the leash" in p for p in too_high.verify_reach(control, 3.24))
    # Lift it to the ceiling and the rod can never be pulled clear of the membrane.
    at_ceiling = RodOnSheet(plane_z=8.0, rod_height=3.5)
    assert any("lifted clear" in p for p in at_ceiling.verify_reach(control, 3.24))


# --- the LAMMPS commands ------------------------------------------------------

def test_pair_style_is_hybrid_and_every_type_pair_is_assigned():
    """pair hybrid requires all three type pairs to be set, and the membrane's
    inherited `pair_coeff 1 1` line has to grow a sub-style name."""
    ff = MesoMemRod()
    params = ff.new_params()
    cmds = ff.pair_commands(params)

    assert cmds[0].startswith("pair_style hybrid mesomem ")
    assert " rod_lj " in cmds[0]
    coeffs = [c for c in cmds if c.startswith("pair_coeff")]
    assert [c.split()[1:4] for c in coeffs] == [
        ["1", "1", "mesomem"], ["1", "2", "rod_lj"], ["2", "2", "rod_lj"]]
    # sigma_pair is the MEMBRANE's radius; the style adds the rod's own.
    assert coeffs[1].split()[4] == str(R_MEM)


def test_the_rods_geometry_travels_on_the_particle():
    """q is the length AND the rod/point marker, radius is the half-thickness, and
    the mass has to be per-atom (the `mass` command does not touch rmass under
    atom_style sphere)."""
    ff = MesoMemRod(rod_mass=6.0)
    params = ff.new_params({"rod_length": 5.0, "rod_radius": 1.5})
    cmds = ff.setup_commands(params)

    assert "set type 1 charge 0.0" in cmds
    assert "set type 2 charge 5.0" in cmds
    assert "set type 2 diameter 3.0" in cmds
    assert "set type 2 mass 6.0" in cmds
    # The membrane's diameter is set by TYPE, so the rod's cannot depend on which
    # of the two commands happens to be issued last.
    assert not any(c.startswith("set group all diameter") for c in cmds)


def test_a_shape_change_writes_the_particle_before_restyling():
    """rod_lj derives its neighbour-cutoff extension from the largest q it can
    find, so the new length has to be on the particle before `pair_style` is
    re-declared."""
    ff = MesoMemRod()
    params = ff.new_params()
    params.set("rod_length", 9.0)
    cmds = ff.live_commands(params, "rod_length")

    charge = next(i for i, c in enumerate(cmds) if c.startswith("set type 2 charge"))
    style = next(i for i, c in enumerate(cmds) if c.startswith("pair_style"))
    assert charge < style
    assert "set type 2 charge 9.0" in cmds


def test_eps_rod_is_a_coefficient_change_only():
    ff = MesoMemRod()
    params = ff.new_params()
    params.set("eps_rod", 5.0)
    cmds = ff.live_commands(params, "eps_rod")
    assert not any(c.startswith("pair_style") for c in cmds)
    assert not any(c.startswith("set ") for c in cmds)
    assert any("rod_lj 0.5 5.0" in c for c in cmds)


def test_the_rods_reach_is_its_cutoff_plus_half_its_length():
    """A bead a half-length along the rod is still in contact at the rod's cutoff,
    because that cutoff is measured from the AXIS."""
    ff = MesoMemRod()
    params = ff.new_params({"rod_length": 5.0, "rod_radius": 1.5, "rc": 2.5})
    assert ff.rod_reach(params) == pytest.approx(ff.rod_cutoff(params) + 2.5)


# --- the energy decomposition -------------------------------------------------

def test_adhesion_matches_the_scalar_reference_along_the_whole_rod():
    """Beads placed level with the rod's middle, its end and past its tip -- which
    is the case the segment clamp exists for, and the one a centre-to-centre
    distance gets wrong."""
    ff = MesoMemRod()
    params = ff.new_params({"rod_length": 5.0, "rod_radius": 1.5, "eps_rod": 3.0})
    sigma_eff = R_MEM + 1.5
    cut = ff.rod_cutoff(params)

    rod = np.array([0.0, 0.0, 0.0])
    beads = np.array([
        [0.0, 0.0, 2.3],       # beside the middle, in the attractive well
        [0.0, 0.0, 1.9],       # beside the middle, inside the repulsive core
        [2.5, 0.0, 2.3],       # level with the tip
        [4.0, 0.0, 1.0],       # past the tip: the clamp is what makes this right
        [3.4, 1.1, 2.0],       # off-axis past the tip
    ])
    state = rod_state(rod, beads, axis=(1.0, 0.0, 0.0))
    pairs = analysis_pairs(ff, state, params)
    terms = ff.energy_terms(state, pairs, params)

    expected = sum(reference_adhesion(rod, b, np.array([1.0, 0.0, 0.0]), 5.0,
                                      sigma_eff, 3.0, cut) for b in beads)
    assert terms[ADHESION].sum() == pytest.approx(expected, rel=1e-12)
    assert abs(expected) > 1.0, "the test configuration has to actually interact"


def test_the_rod_gets_no_membrane_energy_and_the_membrane_no_adhesion():
    """`pair_style hybrid` routes 1-1 to the membrane and 1-2 to the rod, and the
    Python expression has to split the same way. Evaluated on a rod pair, the tilt
    term would read the rod's AXIS as a membrane director and invent an energy."""
    ff = MesoMemRod()
    params = ff.new_params({"rod_length": 5.0, "rod_radius": 1.5, "eps_rod": 3.0})
    # Two beads at a normal membrane spacing, and a rod just above them.
    beads = np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]])
    state = rod_state([0.45, 0.0, 2.2], beads, axis=(1.0, 0.0, 0.0))
    pairs = analysis_pairs(ff, state, params)
    terms = ff.energy_terms(state, pairs, params)

    rod = 2
    on_rod = (pairs.a == rod) | (pairs.b == rod)
    assert on_rod.sum() == 2                      # both beads see the rod
    for label in (ISO, TILT, SPLAY):
        assert terms[label][on_rod] == pytest.approx(0.0)
    assert terms[ADHESION][~on_rod] == pytest.approx(0.0)
    # And the membrane pair still carries its own energy, unchanged by the rod.
    assert terms[ISO][~on_rod].sum() != 0.0


def test_a_single_species_state_reports_no_adhesion_rather_than_failing():
    """The verifier and the remote client can hand this force field a frame with
    no types at all; an observable or a term that is not applicable must report
    nothing, not raise."""
    ff = MesoMemRod()
    params = ff.new_params()
    positions = np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]])
    state = FrameState(positions=positions,
                       directors=np.tile([0.0, 0.0, 1.0], (2, 1)))
    pairs = build_pairs(positions, ff.interaction_cutoff(params))
    terms = ff.energy_terms(state, pairs, params)
    assert terms[ADHESION] == pytest.approx(np.zeros(len(pairs)))
    assert terms[ISO].sum() != 0.0


def test_adhesion_uses_the_minimum_image_across_a_periodic_seam():
    """The membrane is periodic, so a bead on the far side of the seam is a
    NEIGHBOUR of a rod near the edge."""
    ff = MesoMemRod()
    params = ff.new_params({"rod_length": 2.0, "rod_radius": 1.5, "eps_rod": 3.0})
    box = Box((-10.0, -10.0, -5.0), (10.0, 10.0, 5.0), periodic=(True, True, False))
    # Rod just inside the +x face; bead just inside the -x face, 1.5 away through
    # the seam and 18.5 away the long way round.
    state = rod_state([9.8, 0.0, 0.0], [[-9.9, 0.0, 2.2]], axis=(0.0, 1.0, 0.0),
                      box=box)
    pairs = analysis_pairs(ff, state, params)
    assert len(pairs) == 1
    terms = ff.energy_terms(state, pairs, params)
    expected = reference_adhesion(np.array([9.8, 0.0, 0.0]),
                                  np.array([10.1, 0.0, 2.2]),
                                  np.array([0.0, 1.0, 0.0]), 2.0,
                                  R_MEM + 1.5, 3.0, ff.rod_cutoff(params))
    assert terms[ADHESION].sum() == pytest.approx(expected, rel=1e-12)
    assert abs(expected) > 0.1


# --- the drawn body -----------------------------------------------------------

def test_the_drawn_body_spans_the_rod_and_names_its_owner():
    """One particle, drawn as a capsule. The spheres have to reach the axis'
    ends (the surface then sits a radius outside that, which is the surface the
    membrane actually touches) and overlap enough not to look beaded."""
    ff = MesoMemRod()
    params = ff.new_params({"rod_length": 5.0, "rod_radius": 1.5})
    axis = np.array([0.0, 0.0, 1.0])
    state = rod_state([1.0, 2.0, 3.0], [[0.0, 0.0, 0.0]], axis=axis)

    centers, radii, dirs, owners = ff.glyph_spheres(state, params)
    along = (centers - np.array([1.0, 2.0, 3.0])) @ axis
    assert along.min() == pytest.approx(-2.5)
    assert along.max() == pytest.approx(+2.5)
    # Consecutive spheres closer together than a radius -> a smooth silhouette.
    assert np.diff(np.sort(along)).max() < 1.5
    assert radii == pytest.approx(np.full(len(centers), 1.5))
    # The drawn director is ACROSS the body, not along it: the renderer bands each
    # sphere about its own director, so along-the-axis makes the capsule look like
    # a stack of coins, while across lines the bands up into one stripe.
    assert np.abs(dirs @ axis).max() < 1e-12
    assert np.linalg.norm(dirs, axis=1) == pytest.approx(np.ones(len(dirs)))
    assert dirs.std(axis=0) == pytest.approx(np.zeros(3))   # all the same
    # Every sphere belongs to the rod, which is the last particle here.
    assert (owners == len(state.positions) - 1).all()
    # Off the axis, the spheres are exactly on it.
    assert np.linalg.norm(centers - (np.array([1.0, 2.0, 3.0])
                                     + along[:, None] * axis), axis=1).max() < 1e-12


def test_a_membrane_only_force_field_draws_no_body():
    from lammps_live.forcefields.mesomem import MesoMem
    state = rod_state([0.0, 0.0, 3.0], [[0.0, 0.0, 0.0]])
    assert MesoMem().glyph_spheres(state, MesoMem().new_params()) is None
    # And so does the rod force field, on a frame that has no rod in it.
    ff = MesoMemRod()
    no_rod = FrameState(positions=np.zeros((2, 3)),
                        directors=np.tile([0.0, 0.0, 1.0], (2, 1)))
    assert ff.glyph_spheres(no_rod, ff.new_params()) is None


# --- the observables ----------------------------------------------------------

def test_rod_observables_read_the_geometry_they_claim_to():
    params = MesoMemRod().new_params({"rod_length": 5.0, "rod_radius": 1.5})
    # A flat membrane in z = 0, and a rod 2.2 above it at 30 degrees.
    rng = np.random.default_rng(0)
    beads = np.column_stack([rng.uniform(-6, 6, 200), rng.uniform(-6, 6, 200),
                             np.zeros(200)])
    tilt = math.radians(30.0)
    state = rod_state([0.0, 0.0, 2.2], beads,
                      axis=(math.cos(tilt), 0.0, math.sin(tilt)))

    assert get_observable("rod_height")(state, None, params) == pytest.approx(2.2)
    assert get_observable("rod_tilt_deg")(state, None, params) == pytest.approx(30.0)
    n = get_observable("rod_contacts")(state, None, params)
    # 1.25 * (0.5 + 1.5) = 2.5 from the axis, over a segment of length 5 in a
    # membrane at ~1.4 beads per unit area: tens of beads, not zero and not all.
    assert 0 < n < 200


def test_rod_observables_are_quiet_on_a_membrane_without_a_rod():
    params = MesoMemRod().new_params()
    state = FrameState(positions=np.zeros((3, 3)),
                       directors=np.tile([0.0, 0.0, 1.0], (3, 1)))
    for name in ("rod_height", "rod_contacts", "rod_tilt_deg"):
        assert math.isnan(get_observable(name)(state, None, params))


# --- with LAMMPS --------------------------------------------------------------

def test_the_rod_engages_the_membrane_and_the_energy_still_checks_out():
    """The whole thing, end to end: drive the rod down until it grabs, then check
    that the membrane pushed back, that beads are touching it, and that the Python
    energy expression still equals what the compiled pair style computed -- now
    with the adhesion term actually carrying something."""
    system = registry.build("mesomem_rod")
    try:
        before = system.analysis.values()
        assert before["rod_contacts"] == 0.0, "the rod must start out of contact"

        for _ in range(30):
            system.set_input_force(0.0, -25.0)
            system.step(20)

        after = system.analysis.values()
        assert after["rod_contacts"] > 20.0
        assert after["rod_height"] < before["rod_height"] - 1.0
        # The membrane pushes back, on the control plane's out-of-plane axis.
        assert system.get_interaction_force()[1] > 1.0

        result = verify_system(system, tolerance=1e-9)
        assert result.ok, result.report("mesomem_rod, rod in contact")
        adhesion = dict(result.terms)[ADHESION]
        assert adhesion < -10.0, f"adhesion should be strongly negative, got {adhesion}"

        # And the drawn body follows the rod that moved.
        centers, _radii, _dirs, owners = system.get_glyph_spheres()
        rod_z = system.current_state().positions[-1][2]
        assert centers[:, 2] == pytest.approx(np.full(len(centers), rod_z),
                                              abs=0.2)
        assert (owners == system.natoms - 1).all()
    finally:
        system.close()


def test_the_rod_can_be_resized_live_without_a_rebuild():
    """rod_length and rod_radius are sliders, not structural parameters: the
    membrane you have already deformed has to stay deformed while the thing
    deforming it changes size.

    Ramped, the way a slider is actually dragged. Growing a buried rod inflates
    it inside the membrane, so the beads it displaces have to be given somewhere
    to go -- jumped straight to the top of the dial it blows the simulation up,
    which the app survives (see the test below) but which is not what dragging a
    slider does.
    """
    system = registry.build("mesomem_rod")
    try:
        for _ in range(20):
            system.set_input_force(0.0, -25.0)
            system.step(20)
        natoms = system.natoms
        contacts = system.analysis.values()["rod_contacts"]

        for radius in np.linspace(1.5, 2.5, 12)[1:]:
            system.set_extra_param("rod_radius", float(radius))
            system.set_input_force(0.0, -5.0)
            system.step(20)
        for length in np.linspace(5.0, 8.0, 12)[1:]:
            system.set_extra_param("rod_length", float(length))
            system.step(20)

        assert system.natoms == natoms          # no rebuild happened
        assert not system.unstable
        # A bigger rod touches more of the membrane, and is drawn bigger.
        assert system.analysis.values()["rod_contacts"] > contacts
        centers, radii, _dirs, _owners = system.get_glyph_spheres()
        assert radii == pytest.approx(np.full(len(radii), 2.5))
        along = np.linalg.norm(centers - centers.mean(axis=0), axis=1).max()
        assert along == pytest.approx(4.0, abs=1e-6)     # half of L = 8
    finally:
        system.close()


def test_inflating_a_buried_rod_degrades_to_a_message_not_a_crash():
    """The one violent thing this playground's sliders can do: jump the radius to
    the top of its range while the rod is inside the membrane, which overlaps the
    rod with dozens of beads at once. Exploring a parameter space means being able
    to reach settings that destroy the simulation; what must not happen is losing
    the session over it."""
    system = registry.build("mesomem_rod")
    try:
        for _ in range(20):
            system.set_input_force(0.0, -25.0)
            system.step(20)
        system.set_extra_param("rod_radius", 4.0)
        for _ in range(10):
            system.step(20)         # must not raise, whatever LAMMPS makes of it
        if system.unstable:
            assert system.get_hud_lines()[0].startswith("SIMULATION UNSTABLE")
            system.reset()
            assert not system.unstable
        system.step(20)
        assert np.isfinite(system.get_positions_3d()[1]).all()
    finally:
        system.close()


# --- the renderer seam --------------------------------------------------------

def test_the_drawn_body_wears_the_rods_own_colour():
    """`_append_bodies` is the seam between "one particle" and "one capsule".

    Two things have to hold or the body reads as a separate object. It takes its
    colour channels from the OWNER, off the REAL per-bead arrays -- the expanded
    ones the tiling produced are a different set of rows, so indexing those would
    paint the body some other bead's colour. And the owner particle itself comes
    back too, at the ordinary bead radius: the caller held it out of the tiling so
    that its copies would not be drawn as bodiless beads, so this is the only
    place it gets drawn at all.
    """
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from lammps_live.ui.renderer import Renderer

    renderer = Renderer.__new__(Renderer)      # no window: just the one helper
    n, k, rod = 6, 4, 2
    real = dict(
        pts=np.arange(3 * n, dtype=float).reshape(n, 3),
        dips=np.tile([0.0, 0.0, 1.0], (n, 1)),
        bright=np.full(n, 0.75),
        energy=np.arange(n, dtype=float) * -3.0,
        tint=np.tile(np.arange(n, dtype=float)[:, None], (1, 3)),
    )
    # What the tiling produced: the OTHER five beads, twice over, and nothing
    # that lines up with the real array's row numbering.
    m = 2 * (n - 1)
    tiled = (np.zeros((m, 3)), np.zeros((m, 3)), np.ones(m), np.zeros(m),
             np.zeros((m, 3)), np.ones(m), np.full(m, 0.5, dtype=np.float32),
             np.zeros(m, dtype=np.float32))

    centers = np.linspace(-2.0, 2.0, k)[:, None] * np.array([1.0, 0.0, 0.0])
    glyphs = (centers, np.full(k, 1.5), np.tile([1.0, 0.0, 0.0], (k, 1)),
              np.full(k, rod))
    out = renderer._append_bodies(glyphs, np.array([rod]), real, 0.5, *tiled)
    gpts, gdips, gbright, genergy, gtint, gfade, gradii, gmaterial = out

    assert len(gpts) == m + 1 + k
    assert np.array_equal(gpts[:m], tiled[0])       # the tiled beads, untouched
    # The owner particle, then its body.
    assert gpts[m] == pytest.approx(real["pts"][rod])
    assert gradii[m] == pytest.approx(0.5)
    assert gpts[m + 1:] == pytest.approx(centers)
    assert gradii[m + 1:] == pytest.approx(1.5)
    assert gdips[m + 1:] == pytest.approx(np.tile([1.0, 0.0, 0.0], (k, 1)))
    assert gbright[m + 1:] == pytest.approx(1.0)
    # Owner's colour, from the REAL arrays -- not from the tiled rows.
    assert genergy[m:] == pytest.approx(real["energy"][rod])
    assert gtint[m:] == pytest.approx(np.tile(real["tint"][rod], (1 + k, 1)))
    assert gfade[m:] == pytest.approx(1.0)          # drawn once, never faded
    assert gmaterial == pytest.approx(0.0)          # no declared body material


def test_a_declared_body_material_takes_the_whole_object_off_the_colourings():
    """`RenderStyle.body_material` marks the owner particle AND every sphere of its
    body, and hands each of them the OWNER's position in the tint channel.

    Both halves matter. Marking the body but not the particle inside it would
    leave a bead of a different colour showing through the ends of the capsule.
    And the tint channel stops being a colour here and becomes the anchor the
    shader samples the material's noise about -- the owner's position, the same
    for every sphere of one body, which is what keeps the texture ON the rod as it
    is steered rather than the rod sliding through a fixed field of it.
    """
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from lammps_live.ui.renderer import BODY_MATERIALS, Renderer

    renderer = Renderer.__new__(Renderer)
    n, k, rod = 6, 4, 2
    real = dict(
        pts=np.arange(3 * n, dtype=float).reshape(n, 3),
        dips=np.tile([0.0, 0.0, 1.0], (n, 1)),
        bright=np.ones(n),
        energy=np.arange(n, dtype=float) * -3.0,
        tint=np.tile(np.arange(n, dtype=float)[:, None], (1, 4)),
    )
    empty = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0),
             np.zeros((0, 4)), None, np.zeros(0, dtype=np.float32),
             np.zeros(0, dtype=np.float32))
    centers = np.linspace(-2.0, 2.0, k)[:, None] * np.array([1.0, 0.0, 0.0])
    glyphs = (centers, np.full(k, 1.5), np.tile([1.0, 0.0, 0.0], (k, 1)),
              np.full(k, rod))

    material = BODY_MATERIALS["bacterium"]
    out = renderer._append_bodies(glyphs, np.array([rod]), real, 0.5, *empty,
                                  material)
    gtint, gmaterial = out[4], out[7]
    assert len(gmaterial) == 1 + k
    assert gmaterial == pytest.approx(material), "the particle is marked too"
    # The anchor, not a colour: the owner's world position, on every row.
    assert gtint[:, :3] == pytest.approx(np.tile(real["pts"][rod], (1 + k, 1)))
    assert gtint[:, 3] == pytest.approx(0.0)


# --- constant lateral pressure -------------------------------------------------

def test_the_barostat_keeps_running_and_dilates_only_the_membrane():
    """The difference between denting and invaginating.

    Covering a rod costs membrane area. In a frozen periodic cell the only place
    that can come from is stretching the lattice, so the membrane cannot wrap --
    which is why this scenario, unlike HexSheet, leaves a barostat installed. It
    dilates the membrane group alone so the rod is not dragged toward the origin
    every time the cell shrinks under it, which would fight the leash.
    """
    scenario = RodOnSheet(baro_press=0.0, baro_damp_run=20.0, hold_steps=50)
    params = scenario.new_params()

    assert scenario.group_commands(params, controlled_id=1) == \
        ["group membrane type 1"]
    cmds = scenario.post_control_settle(params)
    baro = next(c for c in cmds if "press/berendsen" in c)
    assert baro.startswith("fix baro membrane press/berendsen")
    assert "couple xy" in baro and "dilate partial" in baro
    assert "x 0.0 0.0 20.0" in baro and "y 0.0 0.0 20.0" in baro
    # And it is NOT unfixed afterwards -- that is the whole point.
    assert not any(c.startswith("unfix baro") for c in cmds)
    assert not any("unfix baro" in c for c in scenario.settle_cleanup_commands())
    # A live cell has to be declared, or every consumer of the box reads a stale
    # one (see PlaygroundSystem.step).
    assert scenario.cell_is_live is True


def test_the_view_draws_every_bead_and_looks_over_the_near_rows():
    """No fixed cut, and a camera raised far enough that there did not need to be
    one.

    These two are one decision. The scene used to remove the near half of the cell
    (`section_min`) so an almost-edge-on camera could see the rod through an opaque
    monolayer; the foreground beads are membrane and are now drawn, which is only
    survivable because the camera looks DOWN at the wrap rather than along it. A
    cut when one is wanted is the viewer's thrust lever (view_slice.py), not this
    file's -- so if `section_min` ever comes back here, the elevation below has to
    come back down with it.
    """
    playground = registry.load("mesomem_rod")
    scenario = playground.scenario
    params = scenario.new_params()
    box = scenario.build(params, np.random.default_rng(0)).box

    cam = scenario.camera(box)
    eye = np.array(cam["eye"], dtype=float)
    elevation = math.degrees(math.asin(eye[2] / np.linalg.norm(eye)))
    # High enough to see over a monolayer, low enough that a depth still reads as
    # a depth rather than foreshortening into the view axis.
    assert 15.0 < elevation < 40.0
    # Standing outside the cell, or the near rows of membrane are behind the eye.
    assert abs(eye[1]) > 0.5 * box.lengths[1]

    style = playground.render_style
    assert style.section_min is None, "the near beads are the membrane -- draw them"
    # And the tiling is off: at this size the copies would be off-screen geometry,
    # each carrying another rod.
    assert tuple(style.periodic_images) == (0, 0, 0)


def test_there_is_room_for_the_rod_to_stand_up_inside_the_wrap():
    """The end of the sequence the playground is for -- engulfed sideways, a neck,
    then the rod upright in the pit -- is a geometric claim about the leash and the
    container, and it is cheap to state.

    A rod driven to the bottom of its leash and then turned end-on sweeps half a
    length past that depth. If the container floor is not below THAT, the demo
    cannot show the last stage: the rod hits the wall instead of turning.
    """
    playground = registry.load("mesomem_rod")
    control = playground.effective_control()
    params = playground.scenario.new_params()
    reach = control.leash[1] + 0.5 * playground.params["rod_length"]
    assert params["z_half"] > reach, (
        f"the leash reaches {control.leash[1]} down and the rod is "
        f"{playground.params['rod_length']} long, so it sweeps to {reach}; the "
        f"container floor is at {params['z_half']}")
    # And the net has to be wider than it is deep, or a wrapped rod cannot be
    # dragged sideways far enough to see what the invagination does with it.
    assert control.leash[0] > control.leash[1]


def test_the_cell_shrinks_as_the_wrap_grows_and_the_runtime_notices():
    """The area the membrane gives up to a wrap, end to end: the barostat lets the
    cell shrink, and the runtime re-reads it so everything downstream (the pair
    list's minimum-imaging, the drawn box) is talking about the cell that exists.
    """
    system = registry.build("mesomem_rod")
    try:
        before = system.box.lengths[0]
        for _ in range(60):
            system.set_input_force(0.0, -30.0)
            system.step(6)

        def shrink():
            return (before - system.box.lengths[0]) / before

        for _ in range(200):
            system.set_input_force(0.0, 0.0)
            system.step(6)
        early = shrink()
        for _ in range(400):
            system.set_input_force(0.0, 0.0)
            system.step(6)
        late = shrink()

        # IT KEEPS GIVING UP AREA AS THE WRAP GROWS, which is the claim -- and it
        # is the claim because a fixed offset would also satisfy "the cell got
        # smaller". Measured: 0.0004 at the first mark and 0.0012 at the second.
        #
        # The absolute numbers are an order of magnitude smaller than they used to
        # be, and that is the point of the change that shrank them: this deck used
        # to start on a lattice 26% too large in area, so the cell was contracting
        # under its own unrelaxed tension the whole time and most of what this test
        # measured had nothing to do with the rod. Idle, with the rod left hanging
        # out of contact, the same number of steps now moves the cell by 0.0002.
        assert late > 2.0 * early, f"the cell stopped following the wrap: {early} -> {late}"
        assert 0.0008 < late < 0.10, "a couple of per cent, not a collapse"
        # The frame state carries the live cell, not the one that was asked for.
        assert system.current_state().box.lengths[0] == pytest.approx(
            system.box.lengths[0])
        assert system.analysis.values()["rod_contacts"] > 50
    finally:
        system.close()


def test_a_frozen_cell_scenario_is_not_paying_for_the_refresh():
    """`cell_is_live` is opt-in, so a scenario whose cell really is frozen keeps
    its box exactly as it built it.

    HexSheet is deliberately NOT in this list any more: its barostat outlives the
    setup too, because a planar membrane held at a fixed projected area is
    laterally compressed at every temperature but the settle's and buckles into a
    standing ripple (see HexSheet's docstring, and
    test_runtime.test_the_sheet_relaxes_to_zero_lateral_tension)."""
    from lammps_live.playground.scenario import HexPatch, RandomFill
    for cls in (HexPatch, RandomFill):
        assert cls.cell_is_live is False


# --- the rod's own pairs ------------------------------------------------------

def test_the_rod_finds_its_own_long_ranged_pairs_without_widening_the_list():
    """The rod reaches twice as far as the membrane does, and the analysis cutoff
    is global -- so widening it to cover the rod finds every membrane pair at the
    long range too. The rod names its own instead.

    The thing that must hold: the union is exactly the rod's pairs within reach,
    with no pair counted twice (which would double its adhesion energy).
    """
    from lammps_live.playground.observables import analysis_pairs
    ff = MesoMemRod()
    params = ff.new_params({"rod_length": 5.0, "rod_radius": 1.5})

    # The membrane's own cutoff, not the rod's reach.
    assert ff.interaction_cutoff(params) == pytest.approx(params["rc"])
    assert ff.rod_reach(params) > 2.0 * ff.interaction_cutoff(params)

    rng = np.random.default_rng(1)
    beads = np.column_stack([rng.uniform(-8, 8, 400), rng.uniform(-8, 8, 400),
                             rng.normal(0, 0.1, 400)])
    state = rod_state([0.0, 0.0, 1.6], beads, axis=(1.0, 0.0, 0.0))
    pairs = analysis_pairs(ff, state, params)

    rod = len(state.positions) - 1
    on_rod = np.flatnonzero((pairs.a == rod) | (pairs.b == rod))
    partners = np.where(pairs.a[on_rod] == rod, pairs.b[on_rod], pairs.a[on_rod])
    assert len(partners) == len(set(partners.tolist())), "a pair was counted twice"

    # Exactly the beads within the rod's reach of its centre, and no others.
    d = np.linalg.norm(beads - state.positions[rod], axis=1)
    assert set(partners.tolist()) == set(np.flatnonzero(d < ff.rod_reach(params)).tolist())
    # Which is more than the membrane's own cutoff would have found.
    assert len(partners) > np.count_nonzero(d < ff.interaction_cutoff(params))


def test_a_membrane_only_force_field_names_no_extra_pairs():
    from lammps_live.forcefields.mesomem import MesoMem
    ff = MesoMem()
    state = rod_state([0.0, 0.0, 3.0], [[0.0, 0.0, 0.0]])
    assert ff.extended_pairs(state, None, ff.new_params()) is None


# --- the section cut ----------------------------------------------------------

def test_the_section_cut_drops_the_near_half_and_keeps_the_body_whole():
    """The cut is applied to the beads and NOT to the bodies: half a rod is a
    rendering artifact, not a more informative picture."""
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from lammps_live.render_style import DEFAULT_STYLE

    style = DEFAULT_STYLE.varied(section_axis=(0.0, 1.0, 0.0), section_min=0.0)
    ys = np.array([-5.0, -0.1, 0.0, 3.0])
    keep = (np.column_stack([np.zeros(4), ys, np.zeros(4)])
            @ np.asarray(style.section_axis)) >= style.section_min
    assert keep.tolist() == [False, False, True, True]
    # A style that asks for no cut keeps everything, which is every other scene.
    assert DEFAULT_STYLE.section_min is None
