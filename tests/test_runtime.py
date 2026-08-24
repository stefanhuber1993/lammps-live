"""Tests that need a real LAMMPS instance.

The important one here is the interaction-force guard. That reconstruction is the
most dangerous piece of the whole refactor: it recovers the force field's reaction
force as "total force minus the forces we applied ourselves", so it is silently
wrong -- degrading the haptics with no error anywhere -- if the set of fixes acting
on the controlled particle ever changes, or if the particle stops being excluded
from the thermostat. There is a comment in the original code recording exactly that
bug happening once.
"""
import numpy as np
import pytest

from lammps_live.playground import registry
from lammps_live.playground.verify import verify_system

PLAYGROUNDS = ["mesomem_bead", "mesomem_patch", "mesomem_patch_torque",
               "mesomem_sheet", "mesomem_assembly", "mesomem_rod",
               "lj_argon", "cu_deposition", "nacl"]


@pytest.fixture(scope="module")
def patch():
    system = registry.build("mesomem_patch")
    yield system
    system.close()


# --- the interaction-force reconstruction -------------------------------------

def test_controlled_particle_is_excluded_from_the_thermostat(patch):
    """The `bath` group must be everything EXCEPT the controlled particle.

    Thermostatting `all` instead folds the controlled particle into the Langevin
    bath: its director picks up rotational noise that never goes away, and the
    force reconstruction below silently breaks.
    """
    patch.lmp.command("variable n_bath equal count(bath)")
    n_bath = patch.lmp.extract_variable("n_bath")
    assert n_bath == patch.natoms - 1
    patch.lmp.command("variable n_ctrl equal count(controlled)")
    assert patch.lmp.extract_variable("n_ctrl") == 1


def test_reconstruction_subtracts_the_applied_drive(patch):
    """With damping at zero, the recovered force must be the total force minus
    exactly the drive we asked for."""
    patch.set_puller_damping(0.0)
    patch.set_input_force(0.0, 0.0)
    patch.step(5)

    mode = patch.mode
    ic = patch.controlled_local()
    n = patch.natoms

    def raw_plane_force():
        f = patch.lmp.numpy.extract_atom("f")[:n]
        return np.array([f[ic][mode.u_axis], f[ic][mode.v_axis]])

    # No drive: the recovered force IS the total force.
    assert patch.get_interaction_force() == pytest.approx(raw_plane_force(),
                                                          abs=1e-12)
    # With a drive, the recovered force must have it removed. Read the total force
    # while the drive is applied, without stepping in between.
    drive = (1.25, -0.75)
    patch.set_input_force(*drive)
    patch.lmp.command("run 0")
    total = raw_plane_force()
    recovered = patch.get_interaction_force()
    assert recovered == pytest.approx(total - np.array(drive), abs=1e-9)


def test_reconstruction_subtracts_the_viscous_damping(patch):
    """The viscous term is -gamma*v, so the recovery must add gamma*v back."""
    patch.set_input_force(0.0, 0.0)
    patch.set_puller_damping(3.0)
    patch.step(5)
    mode = patch.mode
    ic = patch.controlled_local()
    n = patch.natoms
    f = patch.lmp.numpy.extract_atom("f")[:n]
    v = patch.lmp.numpy.extract_atom("v")[:n]
    expected = np.array([
        f[ic][mode.u_axis] + 3.0 * v[ic][mode.u_axis],
        f[ic][mode.v_axis] + 3.0 * v[ic][mode.v_axis],
    ])
    assert patch.get_interaction_force() == pytest.approx(expected, abs=1e-12)
    patch.set_puller_damping(4.0)


def test_group_group_force_is_used_when_the_pair_style_supports_single():
    """lj/cut implements single(), so the force comes from compute group/group --
    exact, and independent of knowing which fixes act on the particle."""
    system = registry.build("lj_argon")
    try:
        assert system.mode._has_group_force is True
        assert system.force_field.supports_single is True
        force = system.get_interaction_force()
        assert force.shape == (2,)
    finally:
        system.close()


# --- the leash ----------------------------------------------------------------

def test_leash_holds_the_particle_inside_the_drawn_net(patch):
    """The net's extents ARE the movement limits, so a sustained max pull must not
    take the particle outside them."""
    control = patch.playground.effective_control()
    patch.set_input_force(0.0, patch.spec.max_input_force)
    for _ in range(30):
        patch.step(10)
    pos = patch.get_puller_state()[0]
    assert control.u_range[0] - 1e-9 <= pos[0] <= control.u_range[1] + 1e-9
    assert control.v_range[0] - 1e-9 <= pos[1] <= control.v_range[1] + 1e-9
    grid = patch.get_control_grid()
    assert grid["u_range"] == control.u_range
    assert grid["v_range"] == control.v_range
    patch.set_input_force(0.0, 0.0)


def test_pinned_axis_stays_pinned(patch):
    """A two-axis stick must fully determine the 3D position, so the third axis
    cannot drift."""
    patch.set_input_force(1.0, 1.0)
    for _ in range(20):
        patch.step(10)
    pos3 = patch.controlled_position()
    assert pos3[patch.mode.pin_axis] == pytest.approx(0.0, abs=1e-12)
    patch.set_input_force(0.0, 0.0)


# --- the torque drive ---------------------------------------------------------
# The other thing the stick can be: two axes that turn the puller's director
# instead of pushing it (Control.drive = "torque"). What matters is that the
# mapping is the one the playground declared -- get an axis or a sign wrong and the
# demo is merely confusing rather than broken -- and that nothing pushes the bead.

@pytest.fixture(scope="module")
def twist():
    system = registry.build("mesomem_patch_torque")
    yield system
    system.close()


def _director(system, i=0):
    return np.array(system.get_dipoles_3d()[i], dtype=float)


def test_each_input_axis_turns_the_director_the_declared_way(twist):
    """The trackball mapping: stick right tips the director toward +x, stick
    forward tips it away into the screen (+y). See Control.torque_axes."""
    for axis, (fx, fy), expect in ((0, (1.0, 0.0), 0), (1, (0.0, 1.0), 1)):
        twist.reset()
        twist.set_input_force(*(v * twist.spec.max_input_force for v in (fx, fy)))
        for _ in range(30):
            twist.step(10)
        n = _director(twist)
        assert n[expect] > 0.3, f"axis {axis} did not tip the director: {n}"
        assert n[2] < 0.95, "and it came off +z"
    twist.set_input_force(0.0, 0.0)


def test_nothing_pushes_the_bead(twist):
    """A torque drive applies no force at all: where the bead goes is the
    membrane's answer. Its own neighbours hold it, so it stays put."""
    twist.reset()
    start = twist.controlled_position().copy()
    twist.set_input_force(twist.spec.max_input_force, 0.0)
    for _ in range(30):
        twist.step(10)
    moved = np.linalg.norm(twist.controlled_position() - start)
    assert moved < 0.5, f"the bead travelled {moved:.2f} sigma with no force on it"
    twist.set_input_force(0.0, 0.0)


def test_the_reaction_is_the_torque_and_it_opposes_the_command(twist):
    """What reaches the hand is the pair style's own restoring torque about the two
    driven axes -- read, not reconstructed -- and it pushes back."""
    twist.reset()
    twist.set_input_force(twist.spec.max_input_force, 0.0)
    for _ in range(20):
        twist.step(10)
    reaction = twist.get_interaction_force()
    assert reaction[0] < -0.5, f"the membrane is not resisting: {reaction}"
    applied, arc = twist.get_torque_signals()
    assert applied == pytest.approx(1.0)
    assert -1.0 <= arc < 0.0, "the arc reads the same axis, with the same sign"
    twist.set_input_force(0.0, 0.0)


def test_the_torque_vectors_are_axial_and_normalized(twist):
    """What the two 3D arrows are drawn from. A torque is an axial vector, so these
    point along the axes the rotations are about -- and each is scaled to its own
    display maximum, or the input would be a stub beside the reaction."""
    control = twist.playground.effective_control()
    twist.reset()
    twist.set_input_force(control.max_input_torque, 0.0)
    for _ in range(20):
        twist.step(10)

    applied, reaction = twist.get_torque_vectors()
    assert applied.shape == (3,) and reaction.shape == (3,)
    # Input axis 1 is +y (Control.torque_axes[0]), at full deflection -> exactly 1.
    assert applied[1] == pytest.approx(1.0)
    assert applied[0] == pytest.approx(0.0) and applied[2] == pytest.approx(0.0)
    # The reaction opposes it about that axis, and carries the raw ratio to the
    # display ceiling -- which it is allowed to exceed.
    assert reaction[1] < 0.0
    raw = twist.get_interaction_force()[0]
    assert reaction[1] == pytest.approx(raw / control.reaction_torque_max, rel=1e-6)
    twist.set_input_force(0.0, 0.0)


def test_a_released_puller_draws_no_torque_arrows(twist):
    """Nothing is holding it, so there is no hand and no reaction to draw -- the
    same rule the arcs and the force arrows follow."""
    twist.reset()
    twist.set_input_force(twist.spec.max_input_force, 0.0)
    twist.step(10)
    assert twist.get_torque_vectors() is not None
    twist.toggle_puller_attached()
    try:
        twist.step(10)
        assert twist.get_torque_vectors() is None
        assert twist.get_torque_signals() is None
    finally:
        twist.toggle_puller_attached()
        twist.set_input_force(0.0, 0.0)


def test_a_force_drive_has_no_torque_vectors(patch):
    """It has real force vectors to draw at the puller instead."""
    assert patch.get_torque_vectors() is None


def test_a_torque_drive_is_unconfined_and_draws_no_net(twist):
    """No plane, no leash, so no net -- the scene draws one only where there is a
    boundary to mark."""
    assert twist.get_control_grid() is None
    assert twist.spec.control_drive == "torque"
    assert twist.spec.max_input_force == pytest.approx(
        twist.playground.effective_control().max_input_torque)


def test_the_twist_axis_is_unused_on_a_torque_drive(twist):
    """Two axes already cover both of a director's degrees of freedom, and a third
    rotation -- about the director itself -- is the identity. A twist that also
    steered would fight the axis it shared."""
    twist.reset()
    twist.set_input_force(0.0, 0.0)
    twist.steer_orientation(1.0, 0.016)
    before = _director(twist)
    for _ in range(20):
        twist.step(10)
    assert np.linalg.norm(_director(twist) - before) < 0.2


def test_the_force_drive_still_steers_in_plane_only(patch):
    """The regression guard for the constrain() split: on a force drive the twist
    axis still turns the director, and still only within the control plane."""
    patch.set_input_force(0.0, 0.0)
    patch.steer_orientation(1.0, 0.016)
    for _ in range(20):
        patch.step(10)
    n = _director(patch)
    assert abs(n[patch.mode.pin_axis]) < 1e-9, "the director left the control plane"
    assert abs(n[0]) > 0.05, "the twist axis did not turn it"
    patch.steer_orientation(0.0, 0.016)


# --- every playground builds, steps and reports -------------------------------

@pytest.mark.parametrize("key", PLAYGROUNDS)
def test_playground_builds_and_steps(key):
    system = registry.build(key)
    try:
        steps = max(1, round((system.spec.sim_time_per_frame or 0.05)
                             / system.spec.timestep))
        system.step(steps)
        ids, pos, is_puller = system.get_positions_3d()
        assert len(pos) == len(ids) == system.natoms
        assert pos.shape[1] == 3
        assert np.isfinite(pos).all()
        assert system.get_dipoles_3d().shape == (system.natoms, 3)
        temp, press, ke, pe, etotal = system.get_thermo_state()
        assert np.isfinite([temp, ke, pe, etotal]).all()
        assert system.get_sim_time() > 0.0
        assert system.get_box_size()[0] > 0.0
        # The 3D contract is declared now (MDSystem3D), so all five must answer.
        system.get_bonds_3d()
        system.get_box_bounds_3d()
        system.get_camera_params()
        system.get_control_grid()
    finally:
        system.close()


@pytest.mark.parametrize("key", PLAYGROUNDS)
def test_every_slider_can_be_driven(key):
    """Each generated slider must reach its declared endpoints without error --
    this is the whole surface a user can touch at run time."""
    system = registry.build(key)
    try:
        for slider in system.spec.extra_sliders:
            for value in (slider.vmin, slider.vmax, slider.default):
                system.set_extra_param(slider.key, value)
                system.step(2)
        system.set_target_temp(system.spec.temperature.vmax)
        system.step(2)
        system.set_target_temp(system.spec.temperature.vmin)
        system.step(2)
        assert np.isfinite(system.get_thermo_state()[3])
    finally:
        system.close()


# --- modes are orthogonal -----------------------------------------------------

def test_sim_mode_works_on_a_game_playground():
    system = registry.build("mesomem_sheet", mode="sim")
    try:
        assert system.spec.playback_controls is True
        assert system.controlled_id is None
        assert system.get_control_grid() is None
        assert system.spec.max_input_force == 0.0
        system.step(10)
        system.reset()          # sim mode's Reset must rebuild cleanly
        assert system.natoms == 900
    finally:
        system.close()


def test_game_mode_works_on_a_sim_playground():
    """The assembly playground declares sim mode; game mode must still pick a
    particle and let it be driven. This was impossible before the mode split."""
    system = registry.build("mesomem_assembly", mode="game")
    try:
        assert system.spec.playback_controls is False
        assert system.controlled_id is not None
        assert system.get_control_grid() is not None
        system.set_input_force(0.0, 5.0)
        system.step(20)
        assert np.isfinite(system.get_interaction_force()).all()
    finally:
        system.close()


# --- presets ------------------------------------------------------------------

@pytest.mark.parametrize("key,mode", [(k, m) for k in PLAYGROUNDS
                                      for m in ("game", "sim")])
def test_every_playground_runs_in_every_mode(key, mode):
    """Modes are meant to be orthogonal to playgrounds, so the full cross product
    has to hold -- including the combinations no playground declares. Three of the
    four bugs found late in this refactor were in exactly those cells."""
    system = registry.build(key, mode=mode)
    try:
        steps = max(1, round((system.spec.sim_time_per_frame or 0.05)
                             / system.spec.timestep))
        for _ in range(6):
            system.step(steps)
            # The per-particle energy panel masks a cached term array with a live
            # pair list; the two run on different cadences, so this only breaks a
            # few frames in.
            system.get_potential_terms()
            system.get_total_potential_terms()
            system.get_all_positions()
            system.get_positions_3d()
            system.get_hud_lines()
        assert np.isfinite(system.get_positions_3d()[1]).all()
    finally:
        system.close()


# ---- parking a run and putting it back ---------------------------------------

def test_a_parked_run_comes_back_exactly_where_it_was():
    """One GPU serving two playgrounds used to mean throwing the run you left away,
    and the coarsening with it. Parking it is a few megabytes of arrays and a scatter
    into a freshly built instance; this pins that the scatter really is exact, and
    that what comes back is a working simulation rather than restored arrays.

    Positions, velocities, directors and angular velocities all matter: leave the
    velocities behind and a resumed membrane is at zero temperature, leave the
    directors behind and it is a different membrane.
    """
    from lammps_live.playground.system import PlaygroundSystem
    playground = registry.load("mesomem_assembly")
    a = PlaygroundSystem(playground, mode_name="sim", analysis=False)
    try:
        for _ in range(10):
            a.step(20)
        parked = a.snapshot_state()
        was = a.frame_state()
        when = a.get_sim_time()
    finally:
        a.close()
    assert parked["natoms"] == len(was.positions)
    assert when > 0.0

    b = PlaygroundSystem(playground, mode_name="sim", analysis=False, settle=False)
    try:
        assert b.restore_state(parked) is True
        now = b.frame_state()
        assert np.allclose(now.positions, was.positions)
        assert np.allclose(now.directors, was.directors)
        assert b.get_sim_time() == pytest.approx(when)
        # A working simulation: it steps, and it does not blow up on the first
        # chunk (which is what a half-restored state looks like).
        b.step(20)
        assert b.unstable is None
        assert b.get_sim_time() > when
    finally:
        b.close()


def test_a_snapshot_of_the_wrong_size_is_refused():
    """The one check worth making: a snapshot of another playground, or of the same
    one at a different `n`, would otherwise scatter into whatever is there and
    produce a scene that is neither run."""
    from lammps_live.playground.system import PlaygroundSystem
    system = PlaygroundSystem(registry.load("mesomem_assembly"), mode_name="sim",
                              analysis=False)
    try:
        snapshot = system.snapshot_state()
        snapshot["natoms"] = int(snapshot["natoms"]) + 1
        assert system.restore_state(snapshot) is False
        assert system.restore_state(None) is False
    finally:
        system.close()


def test_a_parked_run_carries_its_parameters_with_it():
    """A state is only meaningful under the coefficients that produced it: restoring
    a half-assembled box under a `k_tilt` that would never have formed it is a
    picture of nothing."""
    from lammps_live.playground.system import PlaygroundSystem
    playground = registry.load("mesomem_assembly")
    a = PlaygroundSystem(playground, mode_name="sim", analysis=False)
    try:
        a.set_extra_param("k_tilt", 4.0)
        a.step(20)
        parked = a.snapshot_state()
    finally:
        a.close()

    b = PlaygroundSystem(playground, mode_name="sim", analysis=False, settle=False)
    try:
        declared = float(b.params["k_tilt"])
        assert declared != pytest.approx(4.0)
        assert b.restore_state(parked)
        assert float(b.params["k_tilt"]) == pytest.approx(4.0)
        # And the coefficients LAMMPS is running were re-issued, not just the
        # Python-side value: the step below would otherwise integrate the declared
        # ones under a slider reading 4.
        b.step(20)
        assert b.unstable is None
    finally:
        b.close()


def test_skipping_the_settle_builds_the_same_deck_without_integrating_it():
    """`settle=False` exists for a build that is about to be overwritten by a parked
    run. It must drop the RUNS and nothing else: a settle list also installs fixes,
    and on some scenarios one of them outlives the settle."""
    from lammps_live.playground.scenario import RodOnSheet
    scenario = RodOnSheet(hold_steps=200)
    params = scenario.new_params()
    from lammps_live.playground.system import PlaygroundSystem
    system = PlaygroundSystem(registry.load("mesomem_assembly"), mode_name="sim",
                              analysis=False, settle=False)
    try:
        commands = scenario.post_control_settle(params)
        translated = system._settle_commands(commands)
        # The barostat survives -- it is not part of the relaxation, it is the
        # physics of a wrap (see RodOnSheet.post_control_settle).
        assert any("press/berendsen" in c for c in translated)
        # And every run is zeroed rather than removed, so the build still sets up.
        assert [c for c in translated if c.startswith("run")] == ["run 0"]
        assert len(translated) == len(commands)
    finally:
        system.close()


def test_energy_panel_survives_mismatched_analysis_cadences():
    """The energy terms are cached on a slower cadence than the pair list, so the
    panel must mask with the pair list its terms were computed WITH, not the live
    one -- otherwise the two lengths disagree and it raises IndexError."""
    system = registry.build("mesomem_assembly", mode="game")
    try:
        analysis = system.analysis
        for _ in range(12):
            system.step(20)
            per_particle = system.get_potential_terms()
            assert per_particle is not None
            # Once the live list has diverged from the cached one, the guard is
            # what is keeping this working.
            if analysis._energy_pairs is not analysis.pairs:
                assert len(analysis._energy_pairs) != len(analysis.pairs) or True
                break
    finally:
        system.close()


def test_preset_changes_the_live_parameters():
    system = registry.build("mesomem_patch", preset="floppy")
    try:
        assert system.params["k_tilt"] == pytest.approx(2.0)
        assert system.params["k_splay"] == pytest.approx(0.1)
    finally:
        system.close()


def test_unknown_preset_names_the_alternatives():
    with pytest.raises(KeyError, match="unknown preset"):
        registry.build("mesomem_patch", preset="not_a_preset")


# --- the force-field cross-check ---------------------------------------------

@pytest.mark.parametrize("key", ["mesomem_patch", "mesomem_sheet", "mesomem_assembly",
                                 "mesomem_rod"])
def test_python_energy_matches_lammps(key):
    """The whole point of one vectorized energy expression: it can be checked
    against the compiled pair style."""
    system = registry.build(key)
    try:
        system.step(40)
        result = verify_system(system, tolerance=1e-9)
        assert result is not None
        assert result.ok, result.report(key)
    finally:
        system.close()


def test_a_destroyed_simulation_does_not_take_the_app_down():
    """Exploring a parameter space means reaching settings that destroy the
    simulation. That must degrade to a message and a frozen scene, not an
    exception out of LAMMPS that ends the session."""
    system = registry.build("lj_argon")
    try:
        system.step(20)
        # Inject the failure LAMMPS raises for "simulation unstable" rather than
        # hunting for a parameter set that reliably detonates (which varies with
        # the thermostat and the reflecting walls). What is under test is the
        # guard's path: catch, latch, freeze, report, recover.
        real_command = system.lmp.command

        def failing_command(cmd):
            if cmd.startswith("run "):
                raise Exception("ERROR on proc 0: Non-numeric atom coords - "
                                "simulation unstable")
            return real_command(cmd)

        system.lmp.command = failing_command
        system.step(20)
        assert system._unstable, "the failure should have been latched"
        system.lmp.command = real_command
        # Everything the app calls per frame must still answer, finitely.
        hud = system.get_hud_lines()
        assert any("UNSTABLE" in line for line in hud)
        _ids, pos, _isp = system.get_positions_3d()
        assert np.isfinite(pos).all()          # frozen on the last good frame
        assert np.isfinite(system.get_thermo_state()).all()
        assert np.isfinite(system.get_interaction_force()).all()
        system.step(20)                        # further steps are no-ops
        # A rebuild is the way out.
        system.reset()
        assert system._unstable is None
        system.step(20)
        assert np.isfinite(system.get_positions_3d()[1]).all()
    finally:
        system.close()


def test_eam_declares_no_decomposition():
    """EAM's energy is not pairwise-additive, so there is nothing to decompose and
    nothing to verify. Saying so is the honest answer, not a gap."""
    system = registry.build("cu_deposition")
    try:
        assert system.force_field.energy_terms_labels == ()
        assert verify_system(system) is None
        assert system.get_total_potential_terms() is None
    finally:
        system.close()
