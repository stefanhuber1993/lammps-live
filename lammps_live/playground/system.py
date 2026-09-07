"""PlaygroundSystem: the single MDSystem implementation the app talks to.

It composes a ForceField, a Scenario, a Mode and an Analysis into the interface
the existing control loop and renderer already speak, so nothing downstream had
to be rewritten -- the app was already push-based and spec-driven with no
per-system branching, which is the part of the old design that was right.

Everything system-specific now lives in the four composed objects. What is left
here is the LAMMPS deck assembly (in one place instead of three near-identical
copies), the id-to-local-index bookkeeping LAMMPS forces on us, and the plumbing
of readouts into the shapes the renderer wants.
"""
import dataclasses
import random

import numpy as np
from lammps import lammps

from ..mdsystem import MDSystem3D, SliderSpec, SystemSpec
from . import forcefield as ff_registry
from .modes import GameMode, SimMode, select_controlled
from .faults import Fault
from . import jitter
from .observables import Analysis
from .pair_probe import probe_pair
from .rdf import InPlaneRDF, RadialRDF3D
from .scenario import Scenario
from .clustering import ClusterTracker, contact_cutoff
from .smoothing import TrajectorySmoother
from .state import Box, FrameState, normalize_rows


def make_mode(playground, mode_name=None):
    """Modes are chosen at run time, not baked into the system, so `--mode game`
    and `--mode sim` both work on any playground."""
    name = (mode_name or playground.mode or "game").lower()
    if name == "sim":
        return SimMode()
    if name == "game":
        return GameMode(playground.effective_control())
    raise ValueError(f"unknown mode {name!r} (expected 'game' or 'sim')")


def make_spec(playground, mode_name=None, preset=None):
    """The SystemSpec for a playground, WITHOUT building a LAMMPS instance.

    Callable from the registry so `--list-playgrounds` and the app's system picker
    can show every playground without constructing any of them.
    """
    force_field = ff_registry.get(playground.force_field)(**playground.force_field_options)
    params = force_field.new_params(playground.resolved_params(preset))
    scenario = playground.scenario
    mode_name = (mode_name or playground.mode or "game").lower()
    control = playground.effective_control()
    t_min, t_max = playground.temperature

    return SystemSpec(
        key=playground.key or playground.name,
        name=playground.name,
        description=playground.description,
        element_label=(playground.element_label
                       or ("membrane bead (director)" if force_field.has_directors
                           else "particle")),
        lattice_spacing=playground.lattice_spacing,
        timestep=scenario.timestep,
        temperature=SliderSpec("Temperature", t_min, t_max,
                               playground.temperature_default, fmt="{:.3f}",
                               unit=" T*" if playground.reduced_units else " K"),
        damping=SliderSpec(
            "Puller damping" if mode_name == "game" else "Puller damping (unused)",
            control.damping_range[0], control.damping_range[1],
            control.damping_default if mode_name == "game" else 0.0,
            fmt="{:.2f}", advanced=True),
        melt_temp=playground.melt_temp,
        force_feedback=playground.force_feedback,
        # One ceiling for a unit of input deflection, in whichever domain this
        # control drives -- a force, or a torque (see Control.input_scale).
        max_input_force=control.input_scale if mode_name == "game" else 0.0,
        control_drive=control.drive if mode_name == "game" else "force",
        puller_speed_cap=(playground.puller_speed_cap
                          or 0.06 * playground.lattice_spacing / scenario.timestep),
        crystal_color=playground.crystal_color,
        species_colors=playground.species_colors,
        species_labels=playground.species_labels,
        species_radii_A=playground.species_radii,
        atom_radius_A=playground.particle_radius,
        sim_time_per_frame=scenario.sim_time_per_frame,
        bond_overlay=playground.bond_overlay,
        render_3d=playground.render_3d,
        render_style=playground.render_style,
        camera_orbit=playground.camera_orbit,
        reduced_units=playground.reduced_units,
        director_arrows=scenario.director_arrows,
        wrap_fade_fraction=scenario.wrap_fade_fraction,
        playback_controls=(mode_name == "sim"),
        # Every live force-field parameter becomes a slider, from ONE
        # declaration. The old code wrote each of these out twice: as a
        # SliderSpec here and again as a string key in a hand-maintained
        # set_extra_param dispatch dict, in each of three system modules.
        extra_sliders=_disclose(playground,
                                params.slider_specs(playground.param_ranges)
                                + smoothing_slider_specs(playground, scenario)),
        lesson=playground.lesson,
    )


def _disclose(playground, slider_specs):
    """Apply the lesson's progressive disclosure to the generated sliders.

    A dial the lesson does not call everyday is moved into the panel's collapsible
    "Advanced" group -- hidden, not removed, so the presenter who wants k_tilt on
    the opening slide is one click away from it and the force field has not been
    quietly redefined (see Lesson.everyday_params and Playground.is_everyday for
    the rule and the reason).

    Done here, on the way out of `make_spec`, because this is the one place the
    force field's declarations and the playground's teaching intent are both in
    scope. `advanced` is the mechanism that already exists for exactly this, and
    reusing it means the panel, the joystick's focus cycle (which deliberately
    skips the advanced group -- see control_focus.py) and the Advanced toggle all
    follow with no changes of their own.
    """
    return tuple(
        ss if playground.is_everyday(ss.key, ss.advanced)
        else dataclasses.replace(ss, advanced=True)
        for ss in slider_specs
    )


# The view key `set_extra_param` recognises, alongside the force field's own live
# parameters. Not a ParamSet entry, because it never reaches LAMMPS.
SMOOTHING_KEY = "view_smoothing"
# The slider's top end, as a number of frames' worth of memory. 20 frames is a
# third of a second at 60 Hz -- enough to flatten the rattle completely while a
# real rearrangement still reads as prompt.
SMOOTHING_MAX_FRAMES = 20
# Where the slider STARTS, in the scenario's own time units (tau on every
# playground that offers it -- they are all the reduced-unit MesoMem ones).
#
# Not zero, which is where it used to start. Every one of these scenes is watched
# for a slow collective change -- a membrane healing, a rod being engulfed, a box
# assembling -- and at the default near-frozen temperature the per-bead rattle on
# top of that is still the fastest thing on screen, so an unsmoothed picture reads
# as noisier than the physics actually is. 0.2 tau is a few frames' worth on every
# scenario here: it takes the shimmer off without any visible lag on a real
# rearrangement, which is what the slider's top end (a third of a second) starts to
# have. Turn it to zero to see every bead's own motion, which is still the right
# thing when a single bead IS the subject.
SMOOTHING_DEFAULT_TAU = 0.2


def smoothing_default(scenario):
    """The starting smoothing span for a scenario, in its own time units.

    Clamped into the slider's range: the span is expressed in simulated time and
    the range is `SMOOTHING_MAX_FRAMES` of the scenario's own per-frame slice, so
    a scenario advancing very little per frame could have a top end below the
    default. Every bundled playground's top end is comfortably above it (the
    tightest, the rod's, is 0.6 tau), but a new one need not be.
    """
    return min(SMOOTHING_DEFAULT_TAU,
               SMOOTHING_MAX_FRAMES * scenario.sim_time_per_frame)


# The other view-only key, and the opposite effect: the synthetic thermal rattle
# that keeps the picture moving between the frames a slow wire delivers. Offered
# only where the state ARRIVES more slowly than it is drawn, i.e. on a remote
# playground -- locally the simulation and the renderer share a frame and there
# is nothing to fill in. See playground/jitter.py.
JITTER_KEY = "view_jitter"


def smoothing_slider_specs(playground, scenario):
    """The two view-only sliders, for the playgrounds that want them.

    Both are advanced, both are visuals-only, and they are opposite ends of one
    idea -- how much of the thermal rattle ends up on screen. Smoothing takes the
    real rattle out; Liveliness puts a synthetic one back where the wire could not
    afford to send it.

    The smoothing span is expressed in the scenario's own simulated time, scaled
    to its per-frame slice, so "full right" means the same ~20 frames of averaging
    on every playground rather than an arbitrary constant that would be gentle on
    one and glacial on the next.
    """
    specs = []
    if playground.trajectory_smoothing:
        per_frame = scenario.sim_time_per_frame
        specs.append(SliderSpec("Smoothing", 0.0,
                                SMOOTHING_MAX_FRAMES * per_frame,
                                smoothing_default(scenario),
                                fmt="{:.2f}", unit=" tau", key=SMOOTHING_KEY,
                                advanced=True))
    if playground.remote is not None:
        # A pure multiplier on an amplitude measured live off the wire, so it has
        # no unit -- 1.0 is "as much motion as the wire is dropping".
        specs.append(SliderSpec("Liveliness", 0.0, jitter.JITTER_MAX,
                                jitter.DEFAULT_JITTER, fmt="{:.2f}",
                                key=JITTER_KEY, advanced=True))
    return tuple(specs)


class PlaygroundSystem(MDSystem3D):
    """A playground, running."""

    def __init__(self, playground, mode_name=None, preset=None, host_profile=None,
                 analysis=True, settle=True):
        self.playground = playground
        # WHETHER TO ACTUALLY INTEGRATE THE SETTLE. False builds the deck exactly as
        # usual -- every fix installed, every command validated -- but takes zero
        # steps where the scenario asked for a relaxation. It exists for one caller:
        # a build whose whole purpose is to have a parked run scattered into it a
        # moment later (see remote/server.py), where relaxing a configuration that is
        # about to be overwritten is the most expensive pointless thing in the flow --
        # measured on the vesicle+polymer playground, 4 of its 8 seconds.
        #
        # Not a general speed switch: a scenario's settle is what makes its initial
        # state a physical one (the sheet's barostat finds the tension-free lattice
        # spacing), so skipping it and then NOT restoring anything leaves the run
        # starting from the made-up geometry the scenario guessed.
        self._settle = bool(settle)
        self.preset = preset
        self.force_field = ff_registry.get(playground.force_field)(
            **playground.force_field_options)
        # A HOST PROFILE is how the same playground runs on a LAMMPS that is not
        # the pip wheel this app was written against -- specifically the cluster's
        # Kokkos/CUDA build, where the MesoMem pair style is compiled in rather
        # than loaded as a plugin and the atom style has a different name. It
        # adapts the force field in place and contributes command-line arguments
        # (`-k on g 1 -sf kk`). None -- every local playground -- changes nothing.
        # See remote/hosts.py.
        self.host_profile = host_profile
        if host_profile is not None:
            host_profile.adapt(self.force_field)
        self.scenario = playground.scenario
        self.params = self.force_field.new_params(playground.resolved_params(preset))
        self.scenario_params = self.scenario.new_params()
        self.mode_name = (mode_name or playground.mode or "game").lower()
        self.mode = make_mode(playground, self.mode_name)
        self.mode.attach(self)
        self.spec = make_spec(playground, self.mode_name, preset)

        self.has_directors = self.force_field.has_directors
        self._target_temp = self.spec.temperature.default
        self._sim_time = 0.0
        # Kept so `reset` can rebuild an identically-configured Analysis rather
        # than silently reverting to the defaults.
        self._analysis_kwargs = dict(energy_every=playground.analysis_energy_every,
                                     enabled=bool(analysis))
        self.analysis = Analysis(self.force_field, playground.observables,
                                 **self._analysis_kwargs)
        # Frame-state cache: several readouts want the same id-ordered gather over
        # every particle, and rebuilding it per readout is what made the old code
        # walk the whole system several times a frame. All of it is declared by
        # `_invalidate_frame_caches`, which is also what a rebuild calls -- see
        # there for why that matters more than it looks.
        self._frame = 0
        self._invalidate_frame_caches()
        # Visual-only trajectory smoothing (see smoothing.py). Its own frame cache,
        # deliberately separate from the analysis one: the filtered coordinates must
        # reach the renderer and NOTHING else, so the two states cannot share a
        # slot that an observable might pick up by accident.
        # Started at the slider's own default rather than at zero, so a system
        # driven without the app (a test, the remote server) smooths the same way
        # the app's first frame will ask it to.
        self._smoothing_tau = (smoothing_default(self.scenario)
                               if playground.trajectory_smoothing else 0.0)
        self._smoother = TrajectorySmoother()
        # The cluster colouring's labelling (see clustering.py). Paced by frame
        # count rather than cached per frame -- unlike the energies, recomputing
        # it every frame would be both expensive and pointless, since it is a
        # slow-moving fact about the configuration.
        self._clusters = ClusterTracker(contact_cutoff(self.spec.atom_radius_A))
        self._last_step_dt = 0.0
        self.analysis_seconds = 0.0
        # Latched message once the simulation has been driven unstable, else None,
        # plus the last finite thermo reading to fall back on.
        self._unstable = None
        self._last_thermo = None
        # The last event that destroyed the simulation, waiting to be shown once
        # (see take_fault). Not the same thing as `_unstable`: that says "stepping
        # has stopped", this says "somebody should be told why".
        self._fault = None
        # PARAMETER VALUES KNOWN TO BUILD. Two snapshots, because a rebuild that
        # fails needs somewhere to fall back TO -- and a slider is allowed to reach
        # a value that only turns out to be impossible when the next build validates
        # it, at which point the value that was fine is already overwritten.
        #   _built_params    what was in force the last time a build succeeded
        #   _initial_params  the playground's own values, before anyone touched a
        #                    slider -- the last resort
        self._initial_params = self._snapshot_params()
        self._built_params = None
        # Whether this scenario overrides housekeeping at all -- checked once so
        # the per-frame path can skip the gather entirely.
        self._has_housekeeping = (
            type(self.scenario).housekeeping is not Scenario.housekeeping
        )
        self._has_director_housekeeping = (
            type(self.scenario).director_housekeeping
            is not Scenario.director_housekeeping
        )

        # Whether the next `run` has to do a full setup (neighbour rebuild + force
        # evaluation) because something structural changed since the last one.
        # See command() and step().
        self._setup_dirty = True
        self.lmp = None
        self._setup(self._new_seed())

    # ---- parameter snapshots -------------------------------------------------

    def _snapshot_params(self):
        """The raw values of both parameter sets, copied.

        Raw, not effective: a clamp is re-applied on read, so storing the clamped
        value would quietly make a restore lossy the moment a clamp's dependency
        moved.
        """
        return (dict(self.params.values), dict(self.scenario_params.values))

    def _restore_params(self, snapshot):
        """Put both parameter sets back, and report what actually moved."""
        ff_values, scenario_values = snapshot
        changed = {}
        for name, value in ff_values.items():
            if self.params.has(name) and self.params.set(name, value):
                changed[name] = value
        for name, value in scenario_values.items():
            if self.scenario_params.has(name):
                self.scenario_params.set(name, value)
        return changed

    def _new_seed(self):
        if self.playground.seed is not None:
            return int(self.playground.seed)
        return random.randint(1, 900_000_000)

    # ---- construction -------------------------------------------------------

    def _setup(self, seed):
        """Build a fresh LAMMPS instance from the composed pieces.

        The command order reproduces what the three hand-written MesoMem systems
        did, including the two distinct relaxation patterns: the sheet's barostat
        settle runs BEFORE any thermostat or control fix exists (so it relaxes a
        genuinely free sheet), while the patch's short settle runs with everything
        already in place.
        """
        self._seed = seed
        rng = np.random.default_rng(seed)
        ff, scenario, params = self.force_field, self.scenario, self.params
        sparams = self.scenario_params

        build = scenario.build(sparams, rng)
        self.box = build.box
        self.bonds = list(build.bonds)
        self.brightness = build.brightness
        self._tints = self.scenario.render_tints(sparams)

        cmdargs = ["-log", "none", "-screen", "none"]
        if self.host_profile is not None:
            cmdargs += list(self.host_profile.lammps_args)
        self.lmp = lammps(cmdargs=cmdargs)
        # A new instance: nothing cached about the old one's local arrays may be
        # read against it (`_pick_controlled` is the first thing that would, on a
        # scenario whose atoms LAMMPS places itself). See _invalidate_frame_caches.
        self._invalidate_frame_caches()
        try:
            self._issue_setup(seed, rng, build, params, sparams)
        except BaseException:
            # A DECK THAT FAILS MUST NOT TAKE THE PROCESS WITH IT. Whatever went
            # wrong (a command this build does not accept, a pair style that is not
            # there), the instance we just made is still holding a GPU context; left
            # for the garbage collector, its destructor runs during interpreter
            # shutdown, when CUDA has already begun unloading. Kokkos then throws
            # from a destructor and the process dies of SIGABRT with a core dump,
            # burying the actual error under sixty lines of backtrace -- which is
            # exactly how the `mass` failure above arrived from the cluster.
            #
            # Closing it here finalises Kokkos while CUDA is still alive, so the
            # error propagates as itself and the caller can report it.
            self.close()
            raise
        # It built, so these values are the ones to come back to.
        self._built_params = self._snapshot_params()

    def _issue_setup(self, seed, rng, build, params, sparams):
        """The deck itself, on the instance `_setup` has just created."""
        ff, scenario = self.force_field, self.scenario
        lmp = self.lmp
        c = lmp.command
        ff.ensure_available(lmp)

        c(f"units {ff.units}")
        c(f"dimension {ff.dimension}")
        c(f"atom_style {ff.atom_style}")
        c(self.box.boundary_command())
        c("atom_modify map array")
        c(self.box.region_command("box"))
        # The box's type counts, plus whatever a bonded force field needs sized
        # up front -- bond and angle types, and the per-atom allowances that
        # cannot be grown after the fact (see ForceField.box_keywords).
        c(f"create_box {ff.n_types} box{ff.box_keywords()}")

        # Particles either come from the scenario's positions array in ONE call,
        # or are placed by LAMMPS itself (random fill with overlap rejection).
        creation = scenario.atom_creation_commands(sparams, seed)
        if creation:
            for cmd in creation:
                c(cmd)
        else:
            # Only the particles the runtime is responsible for. A scenario
            # carrying bonded particles builds those itself, in create_commands
            # below, because bonds cannot come through create_atoms -- see
            # ScenarioBuild.n_direct.
            n = build.n_uploaded
            lmp.create_atoms(n, list(range(1, n + 1)),
                             [int(t) for t in build.types[:n]],
                             build.positions[:n].ravel().tolist())

        for cmd in ff.setup_commands(params):
            c(cmd)
        for cmd in scenario.create_commands(sparams, build, seed):
            c(cmd)
        for cmd in ff.pair_commands(params):
            c(cmd)
        for cmd in ff.bonded_commands(params):
            c(cmd)
        c(f"neighbor {scenario.neighbor_skin} bin")
        c("neigh_modify every 1 delay 0 check yes")

        self.natoms = lmp.get_natoms()
        self.all_ids = np.arange(1, self.natoms + 1)
        self._pick_controlled(build)

        # Groups: the mode's first (it defines `controlled`), then the scenario's,
        # which may partition the rest (a frozen floor, a mobile crystal).
        for cmd in self.mode.group_commands():
            c(cmd)
        for cmd in scenario.group_commands(sparams, self.controlled_id):
            c(cmd)

        # Integrators: a scenario that partitions its particles installs its own;
        # otherwise the force field's single global one covers everything.
        scenario_integrators = scenario.integrator_commands(sparams)
        if scenario_integrators:
            for cmd in scenario_integrators:
                c(cmd)
        else:
            global_integrator = ff.integrator_command()
            if global_integrator:
                c(global_integrator)

        for cmd in scenario.wall_commands(self.box):
            c(cmd)
        for cmd in scenario.extra_setup_commands(sparams):
            c(cmd)
        c(f"timestep {scenario.timestep}")
        c("thermo 100000")

        settle = scenario.pre_control_settle(sparams, seed)
        if settle:
            for cmd in self._settle_commands(settle):
                c(cmd)
            for cmd in scenario.settle_cleanup_commands():
                c(cmd)

        # The thermostat's own temperature compute (if it has one) must exist
        # before the fix that binds to it.
        self._bath_group = scenario.thermostat_group() or self.mode.thermostat_group()
        self._thermostat = scenario.get_thermostat()
        tc = self._thermostat.temperature_compute(self._bath_group)
        if tc is not None:
            self._temp_compute, tc_cmd = tc
            c(tc_cmd)
        else:
            self._temp_compute = "bath_temp"
            c(f"compute bath_temp {self._bath_group} temp")
        for cmd in self._thermostat.initial_commands(
                self._bath_group, self._target_temp,
                scenario.thermostat_damp(sparams), seed):
            c(cmd)

        for cmd in self.mode.control_commands(params):
            c(cmd)

        c("compute ke_atom all ke/atom")
        c("compute pe_atom all pe/atom")

        post = scenario.post_control_settle(sparams)
        for cmd in self._settle_commands(post or ["run 0"]):
            c(cmd)

        # The cell may have been rescaled by a barostat during settling, so read
        # it back rather than trusting the requested geometry.
        self._refresh_box_from_lammps()
        # AGAIN, and this one is not redundant even though the instance was fresh
        # a moment ago: the deck above has RUN (the settle, and `run 0` at the
        # least), and a run is free to reorder the local arrays -- while
        # `_pick_controlled` may already have built a permutation from before it,
        # on a scenario whose atoms LAMMPS places itself. `on_built` is the first
        # reader, and what it reads is the controlled particle's own coordinate, so
        # the permutation it uses has to be this instance's, taken after
        # everything that could have reordered it.
        self._invalidate_frame_caches()
        if hasattr(self.mode, "on_built"):
            self.mode.on_built()
        self._rdf = self._make_rdf()
        self._setup_dirty = True   # the first chunk after a build sets up in full
        # A rebuild is a new set of particles; a filter still holding the old ones
        # would drag the first frames of the new scene toward the previous one.
        self._smoother.reset()
        self._clusters.reset()
        self._sim_time = 0.0
        # Populate the panels for the paused first frame (sim mode shows its fresh
        # state before Play is pressed, and would otherwise show empty bars).
        self._refresh_analysis(force=True)

    # ---- parking a run, so switching away does not throw it out --------------

    def snapshot_state(self):
        """Everything a run IS, in stable id order: where the particles are, how
        fast, which way they point, how fast they are turning, the cell, and how far
        the run had got. None if there is no instance to read.

        WHAT THIS IS FOR. One GPU serves several remote playgrounds by rebuilding
        (see remote/server.py), and rebuilding used to mean the coarsened box you had
        been watching for four minutes was gone -- "the honest cost of the trade", as
        the old comment put it. It does not have to be: the state is a few megabytes
        of arrays, and putting them back into a freshly built instance costs nothing
        now that building one is a second rather than a minute (see
        RandomFill.build). So a switch parks this and a switch back restores it.

        ID ORDER, not local order, and that is the whole subtlety: LAMMPS sorts its
        local arrays spatially as it runs, and it is emphatically not the same
        ordering on the other side of a rebuild. Ids are, so they are what the
        snapshot is keyed on -- and `restore_state` reads them back the same way.

        The parameters travel with it, because a state is only meaningful under the
        coefficients it was produced with: restoring a half-assembled box under a
        `k_tilt` that would never have formed it is a picture of nothing. Note what
        that is and is not -- over a network link the CLIENT pushes its own slider
        values on attach, and those win, because they are the ones the person is
        looking at. What the parked parameters buy is that the restored state is
        validated and integrated under its own coefficients until then, rather than
        under whatever the fresh build happened to declare.
        """
        if self.lmp is None:
            return None
        order = self._order()
        n = self.natoms
        take = lambda name, cols: np.array(       # noqa: E731 -- five identical reads
            self.lmp.numpy.extract_atom(name)[:n], dtype=float)[order][:, :cols]
        state = {
            "natoms": int(n),
            "x": take("x", 3),
            "v": take("v", 3),
            "box_lo": tuple(self.box.lo),
            "box_hi": tuple(self.box.hi),
            "sim_time": float(self._sim_time),
            "params": self._snapshot_params(),
        }
        if self.has_directors:
            # FOUR columns of `mu`, not three. LAMMPS keeps the dipole's MAGNITUDE
            # in the fourth, and the pair style divides by it to get the unit
            # director -- so writing three new components over a magnitude that
            # belongs to the old ones would silently rescale every director on the
            # restore. It is 1.0 for a membrane bead and 0 for a polymer bead that
            # has no director at all, and both of those have to survive.
            state["mu"] = take("mu", 4)
            state["omega"] = take("omega", 3)
        return state

    def restore_state(self, state):
        """Put a `snapshot_state` back onto the current instance. True if it took.

        REFUSED RATHER THAN FORCED when the particle count does not match, which is
        the only check worth making: a snapshot of a different playground, or of the
        same one at a different `n`, would otherwise scatter into whatever happens to
        be there and produce a scene that is neither. Everything else about the two
        instances is the same by construction -- same scenario, same force field,
        same commands -- because the caller rebuilt from the same playground.

        The cell is restored too, and it has to be for any scenario with a barostat:
        a parked run's cell has shrunk to whatever its wrap or its pressure asked
        for, and dropping the particles from it into the freshly built (larger) one
        would leave the whole system at the wrong density.
        """
        if self.lmp is None or not state:
            return False
        if int(state.get("natoms", -1)) != int(self.natoms):
            return False
        # The box first: `change_box` moves the boundaries, and the coordinates
        # written after it are then written into the cell they came from.
        lo, hi = state["box_lo"], state["box_hi"]
        if (tuple(lo), tuple(hi)) != (tuple(self.box.lo), tuple(self.box.hi)):
            self.command(f"change_box all x final {lo[0]} {hi[0]} "
                         f"y final {lo[1]} {hi[1]} z final {lo[2]} {hi[2]} units box")
            self._refresh_box_from_lammps()
        order = self._order()
        put = lambda name, values: self._scatter(name, order, values)  # noqa: E731
        put("x", state["x"])
        put("v", state["v"])
        if self.has_directors and "mu" in state:
            put("mu", state["mu"])
            put("omega", state["omega"])
        self._restore_params(state.get("params") or self._snapshot_params())
        for cmd in self.force_field.pair_commands(self.params):
            self.command(cmd)
        self._sim_time = float(state.get("sim_time", 0.0))
        # A restored run is a different set of coordinates from the one the filters
        # and the cluster labelling were following, exactly as a rebuild is.
        self._smoother.reset()
        self._clusters.reset()
        self._invalidate_frame_caches()
        # `change_box` and the re-issued coefficients both invalidate `pre no`, and
        # `command` has already said so; this makes the panels describe the restored
        # state rather than the built-and-thrown-away one.
        self._refresh_analysis(force=True)
        return True

    def _scatter(self, name, order, values):
        """Write an id-ordered array back into LAMMPS' local array, through the
        id-order permutation.

        The column count comes from the snapshot rather than from the target, so an
        array LAMMPS keeps wider than the physics needs is written exactly as far as
        it was read -- see `snapshot_state` on `mu`'s fourth column.
        """
        target = self.lmp.numpy.extract_atom(name)
        if target is None:
            return
        cols = values.shape[1]
        target[order[:len(values)], :cols] = values

    def _settle_commands(self, commands):
        """A settle command list, with its `run N` turned into `run 0` when this
        instance was built with `settle=False`.

        The RUNS are what is dropped, and nothing else. A settle list is not only
        integration: it installs the fixes that do the relaxing, and on some
        scenarios one of those OUTLIVES it -- the rod's barostat is issued in
        `post_control_settle` and is never unfixed, because a wrap needs it running
        (see RodOnSheet). Dropping whole commands would drop that too. `run 0` also
        keeps the one thing every build needs from this step: a full setup, which is
        what validates the coefficients and populates the forces.
        """
        if self._settle:
            return list(commands)
        out = []
        for cmd in commands:
            bits = cmd.split()
            out.append("run 0" if bits and bits[0] == "run" else cmd)
        return out

    def _pick_controlled(self, build):
        """Choose the controlled particle from the INITIAL configuration, and
        remember it by atom id -- LAMMPS reorders its local arrays between steps
        (spatial sorting), so an index would not survive."""
        self.controlled_index = None
        self.controlled_id = None
        if not self.mode.needs_control_particle:
            return
        positions = build.positions
        if not len(positions):
            # LAMMPS placed the particles, so read them back to choose.
            positions = self._read_positions_by_id()
        idx = select_controlled(positions, self.box,
                               self.playground.effective_control().atom)
        if idx is None:
            return
        self.controlled_index = int(idx)
        self.controlled_id = int(idx) + 1

    def _measured_temp(self):
        return self.lmp.extract_compute(self._temp_compute, 0, 0)

    def _refresh_box_from_lammps(self):
        lo = tuple(self.lmp.extract_global(f"box{a}lo") for a in "xyz")
        hi = tuple(self.lmp.extract_global(f"box{a}hi") for a in "xyz")
        self.box = Box(lo, hi, self.box.periodic)

    def _spacing(self):
        """The scenario's nearest-neighbour spacing if it declares one, else 1.0
        -- used only to scale RDF ranges."""
        return self.scenario_params["a"] if self.scenario_params.has("a") else 1.0

    def _make_rdf(self):
        """A bulk periodic cell gets the 3D radial g(r); anything else (a patch,
        or a monolayer periodic only in-plane) gets the in-plane one, which is
        what actually reads out lateral order for a sheet. A scenario that knows
        better -- a crystal slab with vacuum above it, where an in-plane
        bounding-box density is meaningless -- overrides the choice.

        r_max spans a handful of lattice shells but never more than half the box
        (the minimum-image limit), so the hexagonal peaks -- and their collapse on
        melting -- are all visible.
        """
        override = self.scenario.make_rdf(self.scenario_params, self.lmp, self.box)
        if override is not None:
            return override
        lengths = self.box.lengths
        if all(self.box.periodic):
            return RadialRDF3D(min(0.5 * min(lengths), 6.0), lengths[0])
        if self.box.periodic[0] and self.box.periodic[1]:
            r_max = min(0.5 * min(lengths[0], lengths[1]), 6.0 * self._spacing())
            return InPlaneRDF(r_max, box=(lengths[0], lengths[1]))
        # A small non-periodic patch resolves only the first couple of neighbour
        # shells rather than a bulk phase, but it populates the RDF panel instead
        # of leaving it stuck on "warming up".
        return InPlaneRDF(3.0 * self._spacing(), nbins=48, box=None, sample_every=1)

    def reset(self, restore_params=True):
        """Put it back how it started: the playground's own parameter values, and a
        fresh initial state. Used by sim mode's Reset, and by R everywhere.

        THE PARAMETERS GO BACK TOO, which they did not use to. Reset used to keep
        whatever the sliders were holding, on the reasoning that you are resetting
        the STATE and not your settings. In front of an audience that is the wrong
        default by a wide margin: half the value of these playgrounds is that you
        can push a dial somewhere absurd, and the thing you then want is one button
        that undoes all of it -- not a hunt back down five sliders for the value
        each of them started at, with no record of what that was. So R means "back
        to the beginning", both halves of it, and the sliders follow the system
        afterwards (App._reset_simulation puts them where this left the values).

        A preset counts as the beginning: `_initial_params` is snapshotted after the
        playground's presets have been applied, so Reset on `--preset buckled`
        returns to buckled rather than to the bare declaration.

        `restore_params=False` keeps whatever the sliders hold, which is what the
        app's automatic recovery after a blow-up wants: nobody pressed anything, so
        the values being explored with must survive the rebuild that saves them (see
        App._handle_faults). That is also the path the fallback ladder below exists
        for -- with the declared values restored, its first rung is already the last
        resort.

        A PARAMETER VALUE CANNOT MAKE THIS RAISE. That is the whole point: Reset is
        the way out of a simulation the sliders destroyed, so if Reset itself dies
        on the same value there is no way back and the app (or, on the cluster, the
        server holding an A100) goes down with it. Which is exactly what happened:
        a `zeta` below 1 is fine to the CPU pair style and rejected by the Kokkos
        one, so it streamed happily -- the per-chunk `run ... pre no` never
        re-validates coefficients -- until Reset rebuilt, `run 0` validated them,
        and the exception took out the server and its allocation.

        So a rebuild falls back: the current values, then the ones that built last
        time, then the playground's own. What it fell back to is recorded as a
        `Fault` for the caller to show, along with the parameters it had to put
        back, so the sliders can be moved to where the simulation actually is.
        """
        if self.lmp is not None:
            self.lmp.close()
            self.lmp = None
        self.analysis = Analysis(self.force_field, self.playground.observables,
                                 **self._analysis_kwargs)
        self._unstable = None      # a rebuild is the way out of an unstable state
        # Back to the declared values BEFORE the rebuild, so the fresh state is
        # built with them rather than with whatever was on the sliders. That also
        # makes the fallback ladder in `_rebuild` a formality on this path (the
        # first rung IS the playground's own values now) rather than something a
        # bad slider can reach -- which is the point of Reset being the way out.
        if restore_params:
            self._restore_params(self._initial_params)
        self._rebuild()

    def _rebuild(self):
        """`_setup`, with the two fallbacks. Raises only if all three fail."""
        ladder = [(None, None)]
        if self._built_params is not None:
            ladder.append((self._built_params, "the values it last built with"))
        ladder.append((self._initial_params, "this playground's own values"))
        failure = None
        for snapshot, note in ladder:
            reverted = self._restore_params(snapshot) if snapshot else {}
            try:
                self._setup(self._new_seed())
            except Exception as exc:                  # noqa: BLE001 -- reported
                failure = failure or exc
                continue
            if failure is not None:
                # It came back, on the second or third try. The fault is not fatal:
                # there IS a running simulation now, it is just not the one the
                # sliders were asking for.
                self._fault = Fault.from_error(failure, reverted=reverted,
                                               fatal=False)
                self._fault.summary += f" Restarted with {note}."
            return
        # Nothing builds, not even the playground's own values -- so this is not a
        # parameter problem and there is nothing left to fall back to.
        self._fault = Fault.from_error(failure)
        raise failure

    # ---- mid-flight commands ------------------------------------------------

    def command(self, cmd):
        """Issue a LAMMPS command between chunks, and remember that the next `run`
        must therefore do a full setup.

        EVERY mid-flight command goes through here rather than straight to
        self.lmp, because `run ... pre no` (see step()) is only valid while the
        system has not changed structurally, and the things this app changes
        between chunks -- redefining the thermostat fix, re-issuing pair
        coefficients or a whole pair style, a scenario's per-frame command --
        are exactly the changes that invalidate it.

        The flag is set for ANY command, not only the ones that genuinely matter
        (a bare `velocity` change does not need a re-setup), because a
        conservative flag costs one full setup on the frames where a control
        actually moved, while getting the distinction wrong costs silently wrong
        forces. The systems where the saving matters -- the big MesoMem ones --
        issue no per-frame commands at all, so they take `pre no` every frame.

        Direct writes into the extracted x/v/mu/omega arrays (housekeeping, the
        controlled particle's constraints) deliberately do NOT set the flag: they
        change no fix, pair style, box or atom count. What they do cost is that
        the first step of the next chunk sees forces computed before the write --
        one step of staleness on a clamp that moves an atom by a fraction of a
        lattice spacing, which is far below the thermostat noise it swims in.
        """
        self._setup_dirty = True
        self.lmp.command(cmd)

    # ---- id <-> local-index bookkeeping -------------------------------------

    def _invalidate_frame_caches(self):
        """Drop everything cached "for this frame". Called on every build.

        THE FRAME COUNTER IS NOT ENOUGH OF A KEY, and that is the whole reason
        this is a method rather than five assignments at the top of `_setup`.
        Each of these caches is keyed on `self._frame`, which is right for the
        steady state -- one gather per frame, however many readouts ask -- but
        `_frame` is a clock, not an identity: it does not move when the LAMMPS
        instance underneath it is replaced, nor when a settle inside a build
        reorders the local arrays. Either of those makes a cache that is still
        "for this frame" describe particles that no longer exist.
        `_order_cache` is the one that bites, because it is a permutation of LOCAL
        INDICES: served stale, `controlled_local` names some other particle
        entirely, `GameMode.on_built` reads ITS coordinate as the control plane,
        and the next `constrain()` teleports the real controlled bead most of a
        sigma onto that plane -- on top of a neighbour, which is a blow-up on the
        first frame after Reset. That was a real bug, and it is what
        tests/test_faults.py's rebuild test now pins.
        """
        self._state = None
        self._state_frame = -1
        self._order_cache = None
        self._order_frame = -1
        # Visual-only trajectory smoothing's own frame cache, deliberately
        # separate from the analysis one: the filtered coordinates must reach the
        # renderer and NOTHING else, so the two states cannot share a slot that an
        # observable might pick up by accident. Same again for the smoothed bead
        # energies, which are read only while that colouring is switched on.
        self._render_cache = None
        self._render_frame = -1
        self._energy_cache = None
        self._energy_render_frame = -1

    def _order(self):
        """Local indices in stable atom-id order.

        Every readout needs this, because LAMMPS is free to reorder its local
        arrays between steps (periodic spatial sorting) and an array index cannot
        survive that -- the renderer's per-particle brightness and the motion
        trails both key off identity.

        Ids run 1..N, so sorting by id IS the id-order gather, and argsort does it
        in numpy instead of building an N-entry Python dict per call (which the
        old code did, several times per frame). Cached per frame.
        """
        if self._order_frame == self._frame and self._order_cache is not None:
            return self._order_cache
        n = self.lmp.get_natoms()
        ids = self.lmp.numpy.extract_atom("id")[:n]
        self.natoms = n
        self._order_cache = np.argsort(ids, kind="stable")
        self._order_frame = self._frame
        return self._order_cache

    def controlled_local(self):
        """Local array index of the controlled particle, via its stable id."""
        if self.controlled_id is None:
            return None
        order = self._order()
        k = self.controlled_id - 1
        return int(order[k]) if k < len(order) else None

    def controlled_position(self):
        ic = self.controlled_local()
        if ic is None:
            return None
        return np.array(self.lmp.numpy.extract_atom("x")[ic][:3], dtype=float)

    def _read_positions_by_id(self):
        order = self._order()
        x = np.array(self.lmp.numpy.extract_atom("x")[:self.natoms], dtype=float)
        return x[order]

    def frame_state(self):
        """This frame's particle state as plain arrays, in stable id order."""
        order = self._order()
        n = self.natoms
        x = np.array(self.lmp.numpy.extract_atom("x")[:n], dtype=float)
        dirs = None
        if self.has_directors:
            mu = np.array(self.lmp.numpy.extract_atom("mu")[:n], dtype=float)[:, :3]
            dirs = normalize_rows(mu[order])
        # Types are only gathered for a multi-species force field, where they
        # decide how each particle is drawn; with one type they are a constant
        # array nobody reads.
        types = None
        if self.force_field.n_types > 1:
            types = np.array(self.lmp.numpy.extract_atom("type")[:n], dtype=int)[order]
        return FrameState(positions=x[order], directors=dirs, types=types,
                          ids=self.all_ids[:len(order)], box=self.box)

    # ---- controls -----------------------------------------------------------

    def set_input_force(self, fx, fy):
        self.mode.set_input_force(fx, fy)

    def set_puller_damping(self, gamma):
        self.mode.set_damping(gamma)

    def steer_orientation(self, rate, dt):
        self.mode.steer_orientation(rate, dt)

    def set_target_temp(self, T):
        t_min, t_max = self.playground.temperature
        T = max(t_min, min(t_max, T))
        if T == self._target_temp:
            return
        self._target_temp = T
        # The thermostat may need to act before the new setpoint -- seeding a
        # Maxwell-Boltzmann distribution when heating a lattice that is at rest,
        # or zeroing the net momentum on a sharp quench. See thermostat.py.
        seed = random.randint(1, 900_000_000)
        for cmd in self._thermostat.pre_change_commands(
                self._bath_group, self._measured_temp(), T, seed):
            self.command(cmd)
        for cmd in self._thermostat.set_commands(
                self._bath_group, T,
                self.scenario.thermostat_damp(self.scenario_params), seed):
            self.command(cmd)

    def set_extra_param(self, key, value):
        """Live force-field dials. Generic: the Param's declared tier decides
        whether the coefficients or the whole pair style get re-issued, and its
        declared clamp is applied when the value is read. No per-system dispatch
        table, and no `if key == "rc"` special case."""
        # The one slider that is not a force-field parameter: it changes how the
        # frame is DRAWN and issues no command, so it is handled before the
        # ParamSet lookup (which would reject it) and never reaches self.command
        # -- a view setting must not cost the next chunk its `pre no`.
        if key == SMOOTHING_KEY:
            self._smoothing_tau = max(0.0, float(value))
            return
        if not self.params.has(key):
            return
        if not self.params.set(key, value):
            return
        for cmd in self.force_field.live_commands(self.params, key):
            self.command(cmd)

    def step(self, n):
        """Advance the simulation, surviving a user-induced blow-up.

        Exploring a parameter space means being able to reach settings that
        destroy the simulation -- switch off the cohesion, or push a length scale
        past what the fixed lattice can accommodate, and LAMMPS raises "simulation
        unstable" on the next run. Letting that propagate kills the whole app and
        loses the session, which is a far worse outcome than a visibly broken
        simulation the user can Reset or dial back out of. So it is caught, latched,
        and reported in the HUD; stepping stops until a rebuild.

        A chunk is a fresh `run`, and by default LAMMPS treats each one as a new
        simulation: it rebuilds the neighbour lists and evaluates the forces once
        before taking the first step. Nothing in a 20-step chunk warrants that --
        it repeats work the previous chunk's last step already did -- so it is
        skipped with `pre no` whenever nothing structural has changed since,
        which command() tracks. `post no` drops the end-of-run timing summary
        that -screen none throws away anyway.

        Measured on the MesoMem systems (900-1500 beads, 20 steps/chunk), this
        takes 1-2 ms off an 18 ms chunk, i.e. 5-14%; the end-of-run thermo
        evaluation still happens, so get_thermo and the per-atom computes the app
        reads every frame stay exact (verified against a forced `run 0`).
        """
        from time import perf_counter
        if self._unstable:
            return
        dt = n * self.scenario.timestep
        setup = "" if self._setup_dirty else " pre no post no"
        try:
            self.lmp.command(f"run {n}{setup}")
        except Exception as exc:                      # noqa: BLE001 -- latched
            self._unstable = str(exc).strip().splitlines()[0]
            # Reported once, as well as latched: the HUD line says the state, the
            # fault says the event, and only the event can be shown as it happens.
            self._fault = Fault.from_error(exc)
            return
        # Only now: a run that threw leaves the instance in an unknown state, and
        # the next one (after a rebuild) should set up in full.
        self._setup_dirty = False
        # Bump the frame counter immediately: `run` may have reordered LAMMPS'
        # local arrays, so every id-order and frame-state cache is now stale.
        # Everything below this line sees consistent, freshly-gathered data.
        self._frame += 1
        # A scenario running a barostat has a cell that is a different size than
        # it was a chunk ago. Read it back BEFORE anything derived from it is
        # gathered: the frame state carries the box into the analysis (which
        # minimum-images the pair list with it) and into the renderer (the drawn
        # outline, the periodic seam). Six extract_globals, and only for the
        # scenarios that say their cell moves.
        if self.scenario.cell_is_live:
            self._refresh_box_from_lammps()
        # The sim time this frame covered, which is the interval the visual
        # smoothing filter averages over (see _render_state). Taken from the actual
        # chunk rather than the scenario's nominal per-frame slice, so a capped or
        # short chunk filters by what it really advanced.
        self._last_step_dt = dt
        self._apply_housekeeping(dt)
        # Constrain AFTER the housekeeping kick, so the readouts (and the
        # renderer) see the post-constraint positions -- the order the original
        # systems used.
        self.mode.after_step(dt)
        # Per-frame commands that depend on measured state and so cannot be a fix
        # (e.g. a drag proportional to this frame's centre-of-mass velocity).
        for cmd in self.scenario.frame_commands(self.scenario_params, self.lmp):
            self.command(cmd)
        self._sim_time += dt
        t0 = perf_counter()
        # The pulled bead is named so its own energy panel keeps a full pair list
        # (see Analysis.update); on a system with no puller the analysis is free
        # to sample.
        self.analysis.update(self._analysis_state(), self.params,
                             keep_index=self._panel_index())
        # Reported to the app's --debug breakdown as its own section, so the
        # Python-side analysis cost is visible rather than hidden inside "sim".
        self.analysis_seconds = perf_counter() - t0

    def _refresh_analysis(self, force=False):
        """Rebuild the frame state and tick the analysis. Used once at build time
        so the panels are populated on the paused first frame."""
        self._frame += 1
        self.analysis.update(self._analysis_state(), self.params, force=force,
                             keep_index=self._panel_index())

    def _panel_index(self):
        """The particle whose own neighbourhood a panel reads, or None."""
        return None if self.controlled_id is None else self.controlled_id - 1

    def _apply_housekeeping(self, dt):
        """Apply the scenario's soft corrections as momentum kicks (delta v = F*dt
        with m = 1). Pure numpy in the scenario; only the array write happens here.

        Skipped entirely -- including the id-order gather -- for scenarios that
        declare no housekeeping, which is most of them.
        """
        if not (self._has_housekeeping or self._has_director_housekeeping):
            return
        order = self._order()
        if not len(order):
            return
        x = np.array(self.lmp.numpy.extract_atom("x")[:self.natoms], dtype=float)
        # housekeeping() works in id order, so the controlled particle's index has
        # to be expressed there too: id k lives at id-order position k - 1.
        controlled = None
        if self.controlled_id is not None and self.controlled_id - 1 < len(order):
            controlled = self.controlled_id - 1
        if self._has_housekeeping:
            f = self.scenario.housekeeping(x[order], self.scenario_params,
                                           controlled, self.box)
            if f is not None:
                v = self.lmp.numpy.extract_atom("v")
                v[order] += f * dt
        # The second channel: corrections that turn the particles' ORIENTATIONS
        # rather than move them. A periodic cell cannot be rigidly rotated -- a
        # membrane spanning it would have to tear -- so a scenario that wants to
        # steer which way its structures face does it through the directors, and
        # lets the force field's own tilt coupling carry the geometry round.
        if self._has_director_housekeeping and self.has_directors:
            mu = np.array(self.lmp.numpy.extract_atom("mu")[:self.natoms],
                          dtype=float)[:, :3]
            w = self.scenario.director_housekeeping(
                x[order], normalize_rows(mu[order]), self.scenario_params,
                controlled, self.box)
            if w is not None:
                omega = self.lmp.numpy.extract_atom("omega")
                omega[order] += w * dt

    # ---- readouts -----------------------------------------------------------

    def get_puller_state(self):
        return self.mode.puller_state()

    def get_puller_energy(self):
        ic = self.controlled_local()
        if ic is None:
            return None, None
        n = self.natoms
        ke = self.lmp.numpy.extract_compute("ke_atom", 1, 1)[:n]
        pe = self.lmp.numpy.extract_compute("pe_atom", 1, 1)[:n]
        return float(ke[ic]), float(pe[ic])

    def get_interaction_force(self):
        return self.mode.interaction_force()

    def toggle_puller_attached(self):
        """Grab / release the controlled particle. Returns the new state."""
        return self.mode.toggle_attached()

    def puller_attached(self):
        return self.mode.attached

    def get_torque_signals(self):
        return self.mode.torque_signals()

    def get_torque_vectors(self):
        return self.mode.torque_vectors()

    def get_thermo_state(self):
        # Hold the last finite reading once unstable, so the plots and the
        # heat-fraction haptics do not ingest NaN.
        if self._unstable and self._last_thermo is not None:
            return self._last_thermo
        state = (self._measured_temp(), self.lmp.get_thermo("press"),
                 self.lmp.get_thermo("ke"), self.lmp.get_thermo("pe"),
                 self.lmp.get_thermo("etotal"))
        if all(np.isfinite(v) for v in state):
            self._last_thermo = state
        elif self._last_thermo is not None:
            return self._last_thermo
        return state

    def get_sim_time(self):
        return self._sim_time

    def take_fault(self):
        """The last simulation-destroying event, once. None if there was none.

        Popped rather than read, so the caller does not have to remember which ones
        it has already shown -- and so two callers cannot both decide to reset.
        """
        fault, self._fault = self._fault, None
        return fault

    def live_param_values(self):
        """The effective value of every live parameter, for the sliders to follow.

        Needed because a rebuild may have put a parameter back: the app pushes its
        slider values into the system every frame, so a slider left pointing at the
        value that destroyed the simulation would push it straight back in.
        """
        return {p.name: float(self.params[p.name])
                for p in self.params.live_params()}

    @property
    def unstable(self):
        """The message latched when these parameters destroyed the simulation, or
        None. Read by the HUD locally and sent down the wire by the remote server,
        so a blow-up on the cluster reads the same as a blow-up here."""
        return self._unstable

    def get_rdf(self):
        state = self._analysis_state()
        if all(self.box.periodic):
            self._rdf.add(state.positions)
        else:
            self._rdf.add(state.positions[:, :2])
        return self._rdf.get()

    def current_state(self):
        """This frame's FrameState, built at most once per frame.

        The public form of `_analysis_state`, for a caller that wants the frame
        rather than a readout derived from it -- the remote server, which sends it
        down the wire (remote/server.py). Sharing the cache is the point: a second
        `frame_state()` call would re-gather every position and director.
        """
        return self._analysis_state()

    def _analysis_state(self):
        """The frame state, rebuilt at most once per frame.

        Once the simulation has gone unstable the particle coordinates are NaN, so
        the last good frame is held instead -- the scene freezes on something
        renderable rather than feeding NaN into the camera and the pair list.
        """
        if self._unstable and self._state is not None:
            return self._state
        if self._state_frame != self._frame:
            self._state = self.frame_state()
            self._state_frame = self._frame
        return self._state

    def _render_state(self):
        """The frame state AS DRAWN: the physics state, optionally temporally
        smoothed (see smoothing.py). Rebuilt at most once per frame, which is also
        what advances the filter exactly one step per frame however many readouts
        ask for it.

        Everything that goes on screen comes through here, so the beads, their
        directors, the bond sticks between them and their wrapped ghosts all agree
        on where a particle is. Everything that MEASURES -- the observables, the
        RDF, the energy panels, the puller's own position and the haptics -- goes
        through _analysis_state() instead and never sees the filtered coordinates.
        """
        state = self._analysis_state()
        if self._smoothing_tau <= 0.0:
            if self._smoother.active:
                self._smoother.reset()
            return state
        if self._render_frame != self._frame:
            # The controlled particle is written straight through: the user is
            # moving it deliberately, so its motion is signal, not the jitter this
            # is here to hide -- and it has to keep agreeing with the puller marker
            # and force arrows, which are drawn from the unsmoothed physics.
            self._render_cache = self._smoother.apply(
                state, self._smoothing_tau, self._last_step_dt,
                keep_exact=(None if self.controlled_id is None
                            else self.controlled_id - 1))
            self._render_frame = self._frame
        return self._render_cache

    def get_potential_terms(self):
        """The controlled particle's share of the additive energy.

        The analysis works in id order, where id k sits at position k - 1, so no
        index translation is needed.
        """
        if self.controlled_id is None or self.playground.pair_annotation:
            return None
        scale = (self.playground.pulled_energy_scale
                 or self.force_field.energy_scale_per_particle * 2.0)
        return self.analysis.energy_panel(
            "Pulled bead energy -- additive (reduced units)", scale,
            index=self.controlled_id - 1)

    def get_pair_annotation(self):
        """The first two particles' interaction, taken apart term by term, or None.

        AND IT REPLACES BOTH ENERGY PANELS, which is why the two methods around it
        go quiet when it is on. A playground only declares this where the scene is
        one pair (see Playground.pair_annotation), and there the pulled bead's
        panel and the whole-system panel are the same three numbers as each other
        AND as this -- the bead has exactly one neighbour, so its share IS the
        system. Three copies of one reading, two of them in the corner of the
        screen and one between the beads it is about.

        Only where the playground asked for it (`Playground.pair_annotation`),
        because on any scene bigger than a pair it annotates an arbitrary two beads
        out of a crowd. See pair_probe.py for what is in it and how the per-term
        forces are obtained.

        Off the RENDER state, not the physics one, so the numbers can never
        disagree with the positions of the two beads the connector is drawn
        between -- the annotation is a label on a picture, and a label reading 1.4
        on a bond drawn at 1.5 is worse than no label. (They are the same arrays on
        the two-bead scene, which switches smoothing off; this is what keeps that
        from being load-bearing.)
        """
        if not self.playground.pair_annotation:
            return None
        # The plane the driven particle is confined to, as its normal -- what turns
        # the force field's landmark SPHERES into the circles the particle can
        # actually cross (see PairAnnotation.plane_normal). A mode with no control
        # plane has no pin axis and hands back None, and the renderer falls back.
        pin = getattr(self.mode, "pin_axis", None)
        normal = None if pin is None else tuple(np.eye(3)[pin])
        return probe_pair(self.force_field, self._render_state(), self.params,
                          plane_normal=normal,
                          torque_scale=self.playground.effective_control()
                          .reaction_torque_max)

    def get_total_potential_terms(self):
        """The whole system's additive energy. Comes from the SAME evaluation as
        get_potential_terms -- the energy expression runs once per analysis frame,
        not once per panel."""
        if self.playground.pair_annotation:
            return None
        n = max(1, len(self.all_ids))
        scale = self.force_field.energy_scale_per_particle * n
        return self.analysis.energy_panel(
            "Whole-system energy -- additive (reduced units)", scale)

    def get_hud_lines(self):
        if self._unstable:
            return [
                "SIMULATION UNSTABLE -- these parameters destroyed it.",
                self._unstable,
                "Dial the sliders back, then press R (or restart) to rebuild.",
            ]
        lines = list(self.analysis.hud_lines() or [])
        # Whether the input device is holding the particle, always shown for a
        # game-mode system: it is a state the user can be in without having meant
        # to be, and the line is also where they find out the key exists.
        if self.mode.needs_control_particle and self.controlled_id is not None:
            lines.append("puller: STEERED (B / trigger releases)" if self.mode.attached
                         else "puller: RELEASED -- free in the simulation (B / trigger grabs)")
        return lines or None

    def get_all_positions(self):
        """2D shadow required by the interface; the 3D renderer uses
        get_positions_3d.

        `species` is the LAMMPS atom type, zero-based, for a force field with
        more than one -- it indexes the playground's species_colors/labels/radii,
        so an ionic lattice draws its cations and anions at their own sizes and
        colours. A single-type force field reports None and every particle is
        drawn the same."""
        state = self._render_state()
        species = None
        if self.force_field.n_types > 1 and state.types is not None:
            species = (np.asarray(state.types) - 1).astype(int)
        return (state.ids, state.positions[:, :2], self._is_controlled_mask(),
                species)

    def _is_controlled_mask(self):
        mask = np.zeros(len(self.all_ids), dtype=bool)
        if self.controlled_id is not None:
            mask[self.controlled_id - 1] = True
        return mask

    def get_bead_brightness(self):
        return self.brightness

    def get_bead_tints(self):
        # The scenario's, computed once at build: it is a fact about the
        # composition, and the composition does not change while a build stands.
        return self._tints

    def get_bead_energies(self):
        """Per-particle potential energy, in id order -- what the energy colouring
        paints.

        THE FACTOR OF 2 IS THE POINT. LAMMPS' `pe/atom` gives each particle HALF
        of every pair energy it takes part in, so that summing over particles
        returns the total potential energy without double counting. That is the
        right convention for a total, and the wrong number to colour by: it is
        half of what it costs to pull the particle out, and half of what the
        additive-energy panel reports for the controlled one. Colouring by it put
        the same bead at -2.8 on the scale while the panel above said -5.8, which
        is exactly as confusing as it sounds.

        So this returns the WHOLE energy of the bonds touching each particle,
        which for a pairwise-additive style is twice the per-atom share. A force
        field that offers no pairwise decomposition (EAM's embedding term is
        many-body) has no such identity, and reports the share as LAMMPS gives it.
        """
        order = self._order()
        pe = np.array(self.lmp.numpy.extract_compute("pe_atom", 1, 1)[:self.natoms],
                      dtype=float)[order]
        energies = 2.0 * pe if self.force_field.energy_terms_labels else pe
        return self._smooth_energies(energies)

    def get_bead_clusters(self):
        """Per-bead cluster colour slot, in id order -- what the cluster colouring
        paints. -1 for a bead in nothing worth naming; see clustering.py.

        Run on the DRAWN coordinates, not the raw ones. Two beads are in the same
        cluster if they are within a contact cutoff of each other, and at any
        finite temperature the pair sitting right at that distance crosses it
        several times a second -- so a labelling built on the raw coordinates
        inherits the whole thermal rattle, exactly as the energy colouring did.
        Reading the smoothed state instead puts the clusters on the same footing
        as the picture they are painted on, for free, and the tracker's own
        hysteresis then only has the genuine ambiguity left to absorb.
        """
        state = self._render_state()
        return self._clusters.slots(state.positions, self.box, self._frame)

    def _smooth_energies(self, energies):
        """Put the colouring through the same low-pass as the drawn positions.

        It is a DRAWN quantity, so it belongs on the drawn side of the line this
        class draws everywhere else (see _render_state): the energy panels, the
        observables and everything that measures keep reading the raw per-atom
        compute. Without it, turning the smoothing up holds the beads still and
        leaves them twinkling -- each one's energy is a sum over neighbours that are
        still rattling, so the colour swings frame to frame far more than the
        configuration it is painting.

        Filtered once per frame however many readouts ask for it, like
        _render_state, so the filter advances with the simulation rather than with
        the number of callers.
        """
        if self._smoothing_tau <= 0.0:
            return energies
        if self._energy_render_frame != self._frame:
            self._energy_cache = self._smoother.smooth_scalar(
                "bead_energy", energies, self._smoothing_tau, self._last_step_dt)
            self._energy_render_frame = self._frame
        return self._energy_cache

    # ---- 3D rendering data --------------------------------------------------

    def get_positions_3d(self):
        state = self._render_state()
        return state.ids, state.positions, self._is_controlled_mask()

    def get_dipoles_3d(self):
        state = self._render_state()
        if state.directors is None:
            return np.zeros((len(state.positions), 3))
        return state.directors

    def get_bonds_3d(self):
        return list(self.bonds)

    def get_glyph_spheres(self):
        """Extra spheres the force field wants drawn -- the rod's body.

        Taken off the RENDER state, so the rod's body follows exactly the same
        (optionally smoothed) coordinates as the bead at its centre; a physics
        state here would let the two slide apart under smoothing.
        """
        return self.force_field.glyph_spheres(self._render_state(), self.params)

    def get_camera_params(self):
        return self.scenario.camera(self.box)

    def get_control_grid(self):
        return self.mode.control_grid()

    def get_box_bounds_3d(self):
        return self.box.bounds_3d()

    def get_box_periodic(self):
        """(x, y, z) periodicity of the cell -- which axes the renderer may
        legitimately repeat when drawing periodic images."""
        return tuple(bool(p) for p in self.box.periodic)

    def get_scene_fit_points(self):
        """World points the camera zooms to fit, or None to keep its fixed FOV.

        A scenario that names its own points has the final say. The control net
        is only added for one that does not: the net spans the full leash, and on
        a small patch that is several times the size of what you are actually
        looking at, so forcing it into frame is what pushed the camera back until
        the beads were a fifth of the width. A scenario that has stated what
        matters should not have that overridden by the interaction's outer limit.
        """
        pts = self.scenario.fit_points(self.scenario_params, self.box)
        if pts is not None:
            return pts
        grid = self.mode.control_grid()
        if grid is None:
            return None
        origin = np.asarray(grid["origin"], dtype=float)
        u = np.asarray(grid["u_axis"], dtype=float)
        v = np.asarray(grid["v_axis"], dtype=float)
        (u0, u1), (v0, v1) = grid["u_range"], grid["v_range"]
        return np.array([origin + uu * u + vv * v
                         for uu in (u0, u1) for vv in (v0, v1)])

    def get_box_size(self):
        return self.box.lengths[0], self.box.lengths[1]

    def close(self):
        if self.lmp is not None:
            self.lmp.close()
            self.lmp = None
