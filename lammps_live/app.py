"""Main control loop: owns the active system, the input source, the
renderer, and the force-feedback shaping that connects them. Switching
systems at runtime (number keys / Tab / shift-Tab / joystick buttons 3-4) tears down the old
LAMMPS instance and rebuilds the UI state (renderer scale, sliders, history,
smoothers) for the new one, in place.

The joystick reaches all of that without the keyboard or the pointer: one focus
at a time -- the viewport or one slider -- moved with the hat switch (left/right
between the scene and the panel, up/down between the panel's rows), which is what
decides whether the stick is flying the camera, holding a bead, or setting a
value. The trigger starts and stops the simulation and button 2 resets it, on
every playground alike. See control_focus.py for the focus model, and
_route_stick / _poll_device_buttons below for the mapping.
"""
import atexit
import math
import os
import signal
import sys
from time import perf_counter

import pygame

from . import config, units
from .control_focus import Choice, ControlFocus
from .view_slice import ViewSlice
from .forcefeedback import (
    ExponentialSmoother2D, shape_damper_coefficient, shape_interaction_force,
    shape_stiffness, shape_velocity_damping,
)
from .playground import registry
from .stepper import SimStepper
from .input import (
    CP_OFFSET_MAX, DAMPER_COEFFICIENT_MAX, JoystickInput, KeyboardInput,
    MouseInput, SPRING_STIFFNESS_MAX,
)
from .ui import BEAD_COLOR_MODES, AtomTrails, Renderer, RollingHistory, Slider
from .ui.camera import Camera3D, OrbitController
from .ui.alert import Alert
from .ui.remote_panel import RemotePanel

STEPS_PER_FRAME_CAP = 200  # sanity cap if a system's timestep is set absurdly small


class App:
    def __init__(self, input_mode, initial_system_key, fullscreen=False, debug=False,
                 mode=None, preset=None, remote_address=None, remote_token="",
                 ui_scale=None):
        self.input_mode = input_mode
        self.debug = debug
        # Whether `_shutdown` has run. It is reachable from the loop's `finally`,
        # from an atexit hook and from a signal that unwinds through both, and
        # releasing the same session twice is not free -- the second scancel goes
        # out over an ssh that is already closing.
        self._shut_down = False
        # `--remote HOST:PORT`: connect a remote playground straight to a server
        # that is already running, instead of allocating one. The loopback path --
        # no SSH, no Slurm, no connect panel.
        self.remote_address = remote_address
        self.remote_token = remote_token
        # Interaction mode ("game"/"sim") and named parameter preset, applied to
        # playgrounds. Both are None for legacy systems, which have neither.
        self.mode_override = mode
        self.preset = preset
        # Exponential moving averages (ms) of the per-frame timing breakdown, and
        # the header line built from them -- shown only under --debug. The line is
        # built from the PREVIOUS frame (a frame's own render time isn't known
        # until after it's drawn), which is fine for a running average.
        # "analysis" is broken out of "sim" because it is the Python-side cost the
        # playground layer adds between LAMMPS steps (energy decomposition and
        # observables); keeping it visible is what makes the frame budget
        # something you can check rather than assume.
        # "gather" is the block that reads a frame's worth of state out of the
        # system (step 3 of _tick): whole-system position, director and energy
        # arrays, the RDF sample, the panels. It grows with the bead count and it
        # runs on THIS thread, in front of the drawing, so on the big scenes it is
        # the part of "other" worth being able to see.
        self._prof_ms = {"sim": 0.0, "analysis": 0.0, "read": 0.0, "ff": 0.0,
                         "gather": 0.0, "render": 0.0, "other": 0.0}
        # Wall time the last frame's analysis took on the stepper thread, whether
        # or not any of it landed on the frame. Reported alongside the breakdown
        # rather than inside it -- see _update_debug.
        self._analysis_wall_ms = 0.0
        self._debug_line = None
        # [(key, SystemSpec), ...] in a stable order for the picker and the
        # number keys. Specs only -- no LAMMPS instance is built to list them.
        self.systems = registry.list_playgrounds()

        # Not pygame.init() -- that also brings up SDL's joystick subsystem,
        # which grabs the Sidewinder as a native SDL game controller. When
        # JoystickInput then claims the same device exclusively via libusb,
        # the device vanishes out from under SDL mid-session and corrupts
        # pygame's event-translation state (observed as `KeyError: 0` inside
        # pygame.event.get()). We drive the joystick ourselves via raw HID
        # reports, so SDL's joystick subsystem is never needed.

        # macOS: keep the green (zoom) button OUT of a native fullscreen "Space".
        # A Space is animated and, with our OpenGL context, not cleanly
        # reversible -- returning from it either leaves the GL drawable stale (a
        # black window) or, if we re-set_mode mid-transition, traps the window in
        # the Space. Disabling Spaces (before the first video init, when SDL reads
        # the hint) makes the green button a plain, reversible window zoom; F11
        # still gives a real fullscreen we control. Must precede display.init().
        if sys.platform == "darwin":
            os.environ.setdefault("SDL_VIDEO_MAC_FULLSCREEN_SPACES", "0")
        pygame.display.init()
        pygame.font.init()

        # ui_scale=None lets the renderer pick from the screen (see ui/scale.py).
        self.renderer = Renderer(config.WINDOW_SIZE, fullscreen=fullscreen,
                                 ui_scale=ui_scale)
        self.clock = pygame.time.Clock()

        # Opening the joystick is a run of blocking HID handshakes (one per
        # force-feedback effect); building the first system is LAMMPS plus the
        # shader compile. Put a frame up and pump the queue between them.
        self._startup_frame("LAMMPS live", f"opening the {self.input_mode} input")
        self.source = self._make_source()
        # The simulation runs on a worker thread while the frame is drawn -- see
        # stepper.py for the rule that imposes and what it buys.
        self.stepper = SimStepper(enabled=config.OVERLAP_SIM_AND_RENDER)

        self.ff_smoother = ExponentialSmoother2D(config.FF_SMOOTHING_TAU)
        self.interaction_smoother = ExponentialSmoother2D(config.FF_SMOOTHING_TAU)

        self.temp_slider = None
        self.damping_slider = None
        self.extra_sliders = []
        self.extra_slider_keys = []
        # Whether the collapsible "Advanced" slider group is expanded. Toggled by
        # clicking its header (see _handle_events); pushed to the renderer each
        # frame so draw_panel knows whether to draw the advanced sliders.
        self.show_advanced = False
        self.history = None
        self.atom_trails = None
        self._trail_frame_counter = 0
        self.energy_baseline = None
        self.sim_wall_time = 0.0
        self.steps_per_frame = 1
        self.total_steps = 0
        # Whether the simulation is running. EVERY playground has this, and every
        # playground shows the Play/Pause/Reset buttons for it -- being able to
        # stop a scene and look at it is not a property of one kind of scene. What
        # differs is only where it STARTS: a playback playground comes up paused on
        # its fresh state (that first configuration is the thing to look at before
        # it moves), an interactive one comes up running (there is a bead to push,
        # and pushing it against a frozen box is nothing). Set per system in
        # _build_system.
        self.sim_playing = False

        # Turntable camera for the 3D systems that ask for one (spec.camera_orbit),
        # and whether the left button is currently dragging it. Rebuilt per
        # system, kept across window resizes -- see _setup_viewport.
        self.orbit_cam = None
        self._orbit_dragging = False
        # What the joystick is currently driving -- the viewport (camera / puller)
        # or one slider -- rebuilt per system in _build_system. See
        # control_focus.py; the hat switch is what moves it.
        self.focus = ControlFocus()
        # The thrust lever's cut through the scene. Not part of the focus cycle --
        # it has its own axis on the device and drives nothing else, so it works
        # whatever the stick is currently pointed at. See view_slice.py.
        self.view_slice = ViewSlice()
        # The bead colouring, as a focus stop: the same state the mouse toggle
        # flips (renderer.bead_color_mode), reachable from the hat cycle. Its
        # options are pictures rather than points on a scale, so it steps once per
        # push instead of walking -- see control_focus.Choice.
        self.color_choice = Choice(
            "bead colour", BEAD_COLOR_MODES,
            on_change=lambda i: setattr(self.renderer, "bead_color_mode",
                                        BEAD_COLOR_MODES[i]))
        # Whether the puller was released BY moving the focus off the viewport, so
        # coming back re-grabs it -- and a bead the user let go of with the trigger
        # is left alone (see _cycle_focus).
        self._focus_released_puller = False
        # What the stick drove on the last frame -- "puller", "camera" or "slider"
        # (see _route_stick). The force-feedback shaping reads it: only a stick that
        # is actually holding a bead gets the contact force rendered onto it.
        self._stick_target = "puller"
        # Previous frame's joystick buttons and hat direction, for edge detection
        # (see _poll_device_buttons).
        self._prev_buttons = frozenset()
        self._prev_hat = (0, 0)

        # The connect panel for a playground whose simulation runs elsewhere. It
        # owns the SSH/Slurm session, so it is created once and lives as long as the
        # app -- what changes is which system it is pointed at (see _build_system).
        self.remote_panel = RemotePanel()
        # The red card that says the simulation died and what was done about it.
        self.alert = Alert()
        # When the last automatic rebuild happened, so a parameter that destroys
        # every fresh state cannot put the app in a reset loop (see _handle_faults).
        self._last_auto_reset = 0.0

        self.system_key = None
        self.system = None
        self._startup_frame("LAMMPS live", f"building {initial_system_key}")
        self._build_system(initial_system_key)
        # Nothing before this point serviced the event queue, so anything in it
        # now was aimed at a window that could not answer -- see
        # _drain_startup_input.
        self._drain_startup_input()

    # ---- startup ------------------------------------------------------------

    def _startup_frame(self, message, detail=None):
        """Draw the splash and service the event queue, mid-startup.

        Between the window appearing and the main loop's first
        `pygame.event.get()` there is a second or two of blocking work -- the
        joystick handshake, LAMMPS, the shaders -- and for all of it the window
        is on screen with nobody reading its events. macOS calls that an
        unresponsive app, and clicking a window in that state is how the press
        and the release stop arriving as a pair. Pumping here keeps the app
        answering; `_drain_startup_input` deals with whatever was aimed at it
        meanwhile.
        """
        pygame.event.pump()
        self.renderer.draw_splash(message, detail)

    def _drain_startup_input(self):
        """Throw away everything queued before the first real frame.

        Those events describe a UI that did not exist when they happened: the
        sliders had not been laid out (their rects are still the placeholder at
        the origin until the first draw_panel), and no system was loaded, so
        replaying a click into the main loop can only produce a drag nobody
        started. Closing the window during startup still counts, though, so a
        QUIT is put back.
        """
        pygame.event.pump()
        quit_requested = bool(pygame.event.get(pygame.QUIT))
        pygame.event.clear()
        if quit_requested:
            pygame.event.post(pygame.event.Event(pygame.QUIT))

    def _make_source(self):
        """Build the input source for the current mode and window geometry. Only
        MouseInput depends on the window geometry, so it's the only one rebuilt on
        a fullscreen toggle; the keyboard is geometry-free and the joystick is
        screen-independent (its hardware handle is kept)."""
        if self.input_mode == "mouse":
            h = self.renderer.window_size[1]
            max_radius_px = min(self.renderer.sim_width, h) * 0.35
            sim_rect = (0, 0, self.renderer.sim_width, h)
            return MouseInput(self.renderer.sim_center_px(), max_radius_px, sim_rect)
        if self.input_mode == "keyboard":
            return KeyboardInput()
        return JoystickInput()

    def _setup_viewport(self):
        """(Re)establish everything tied to the sim viewport size: the box<->
        screen mapping and, for 3D systems, the perspective camera. Called when
        the system changes and after a fullscreen toggle."""
        spec = self.system.spec
        self.renderer.set_box_size(self.system.get_box_size())
        if spec.render_3d:
            cam = self.system.get_camera_params()
            self.camera3d = Camera3D(
                cam["eye"], cam["target"], cam["up"], cam["fov_deg"],
                self.renderer.sim_width, self.renderer.window_size[1],
            )
            # Zoom to fit the scene to the (possibly fullscreen, any-aspect) sim
            # viewport instead of the fixed vertical FOV, so the beads fill the
            # available width/height rather than leaving big side margins. This
            # is also what fixes the focal length an orbit then keeps: the
            # turntable dollies by moving the eye, not by re-zooming.
            fit_pts = self.system.get_scene_fit_points()
            if fit_pts is not None:
                self.camera3d.fit_to_points(fit_pts)
            # A turntable camera survives a resize (self.orbit_cam is cleared
            # only when the SYSTEM changes), so the view does not snap back to
            # its starting angle just because the window was dragged.
            if spec.camera_orbit is not None and self.orbit_cam is None:
                self.orbit_cam = OrbitController(cam["eye"], cam["target"],
                                                 spec.camera_orbit)
            if self.orbit_cam is not None:
                self.camera3d.move_to(self.orbit_cam.eye(), self.orbit_cam.target)
        else:
            self.camera3d = None
            self.orbit_cam = None

    def _toggle_fullscreen(self):
        self.renderer.toggle_fullscreen()
        self._after_resize()

    def _exit_fullscreen(self):
        self.renderer.set_windowed()
        self._after_resize()

    def _after_resize(self):
        """Re-establish everything tied to the window size after the display
        surface changed (fullscreen toggle or an OS window resize)."""
        # Resizing happens between frames, i.e. potentially mid-step, and
        # re-establishing the viewport reads back through the system. Rare enough
        # that simply waiting is the right trade.
        self._sim_idle()
        self._setup_viewport()
        if self.input_mode == "mouse":
            self.source = self._make_source()
        # The simulation runs on a worker thread while the frame is drawn -- see
        # stepper.py for the rule that imposes and what it buys.
        self.stepper = SimStepper(enabled=config.OVERLAP_SIM_AND_RENDER)

    def _sim_idle(self):
        """Let any in-flight step finish. Anything that rebuilds, resets or
        closes the simulation has to call this first -- the worker is inside
        LAMMPS until it returns (see stepper.py)."""
        self.stepper.wait()

    def _build_system(self, key):
        """(Re)build the active system and everything downstream of its
        SystemSpec -- renderer scale, sliders, history, smoothers. Safe to
        call again later to switch systems live."""
        if self.system is not None:
            self._sim_idle()
            self.system.close()

        self.system = registry.build(key, mode=self.mode_override,
                                     preset=self.preset)
        self.system_key = key
        spec = self.system.spec

        # A remote playground comes up disconnected, with the connect panel open --
        # unless a session from earlier is still up, in which case it reconnects to
        # the simulation the server has been holding (see RemotePanel.attach_system).
        # Switching AWAY no longer gives the GPU back: going to another playground
        # and returning is a normal thing to do mid-demo, and it should cost a
        # socket rather than another queue wait. What ends the allocation is closing
        # the window, Disconnect, or the server's own idle timeout.
        from .remote.client import LinkClosed, RemoteSystem
        if isinstance(self.system, RemoteSystem) and self.remote_address:
            self.remote_panel.release()
            host, port = self.remote_address
            try:
                self.system.connect(host, port, self.remote_token)
                print(f"[lammps-live] connected to {host}:{port} -- "
                      f"{self.system.status}")
            except LinkClosed as exc:
                # Not fatal: the scene comes up empty with the reason on the HUD,
                # which is more use than a traceback over a server that has not
                # been started yet.
                print(f"[lammps-live] {exc}")
        elif isinstance(self.system, RemoteSystem):
            self.remote_panel.attach_system(self.system, key)
        else:
            self.remote_panel.detach_system()

        # Box<->screen mapping and (for 3D systems) the perspective camera. The
        # turntable is dropped first: a new system means a new scene, so it must
        # be framed from that scenario's own angle, not the last one's.
        self.orbit_cam = None
        self._orbit_dragging = False
        self._setup_viewport()

        if self.temp_slider is None:
            self.temp_slider = Slider.from_spec((0, 0, 100, 4), spec.temperature)
            self.damping_slider = Slider.from_spec((0, 0, 100, 4), spec.damping)
        else:
            self.temp_slider.reset(spec.temperature)
            self.damping_slider.reset(spec.damping)

        # Extra live-tunable parameters (per-system, variable count -- e.g. the
        # MesoMem k_tilt / k_splay / eta dials), rebuilt from scratch since the
        # count and identities differ between systems. Each remembers the
        # SliderSpec key so set_extra_param knows which parameter it drives.
        self.extra_sliders = [Slider.from_spec((0, 0, 100, 4), ss)
                              for ss in spec.extra_sliders]
        self.extra_slider_keys = [ss.key for ss in spec.extra_sliders]

        # What the joystick can drive here, and where its cycle starts: the
        # viewport, then the bead colouring, then every EVERYDAY slider in panel
        # order. The advanced group is deliberately left out -- see
        # control_focus.py. On the MesoMem playgrounds this is viewport, bead
        # colour, Temperature, k_tilt, k_splay, zeta. The colouring is only a
        # stop on the scenes that have one (the 2D crystals colour by species,
        # which is not a choice), and it follows whatever the toggle is set to
        # rather than resetting it -- the colouring is the viewer's preference, not
        # the playground's.
        choices = ()
        if spec.render_3d:
            self.color_choice.index = BEAD_COLOR_MODES.index(
                self.renderer.bead_color_mode)
            choices = (self.color_choice,)
        self.focus.set_stops([s for s in self._sliders() if not s.advanced], choices)
        self._focus_released_puller = False
        # A different box, and a lever nobody has touched since: start whole again.
        self.view_slice.reset()

        if self.history is None:
            self.history = RollingHistory(config.HISTORY_WINDOW_SECONDS, ["temp", "press", "ke", "pe", "etotal"])
        else:
            self.history.reset()
        if self.atom_trails is None:
            self.atom_trails = AtomTrails(config.TRAIL_WINDOW_SECONDS)
        else:
            self.atom_trails.reset()
        self._trail_frame_counter = 0
        self.energy_baseline = None
        self.sim_wall_time = 0.0
        self.total_steps = 0
        # Playback systems start paused, showing their fresh initial state until
        # Play is pressed; interactive ones start running -- see `sim_playing`.
        self.sim_playing = not spec.playback_controls

        self.ff_smoother.reset()
        self.interaction_smoother.reset()

        sim_time_per_frame = spec.sim_time_per_frame or config.SIM_TIME_PER_FRAME
        self.steps_per_frame = max(1, min(STEPS_PER_FRAME_CAP, round(sim_time_per_frame / spec.timestep)))

    def _cycle_system(self, step=1):
        keys = [key for key, _ in self.systems]
        idx = keys.index(self.system_key)
        self._build_system(keys[(idx + step) % len(keys)])

    def _reset_simulation(self):
        """Restart the system from a fresh initial state (e.g. re-randomize the
        self-assembly box), keeping the current slider values, and clear the
        derived per-run state (plots, trails, energy baseline, step count).

        It leaves the run in whatever state that playground STARTS in -- paused for
        a playback scene, so the fresh configuration is visible before it moves;
        running for an interactive one, because a reset in the middle of a demo is
        "put it back how it was", and having to find Play afterwards is not that.

        The wait is not optional. Reset arrives from the event handler, which runs
        BETWEEN frames -- and between frames is exactly when a step is in flight
        (see stepper.py: it is launched at the end of one frame and collected at
        the start of the next). Rebuilding under it means tearing down a LAMMPS
        instance the worker is inside; on the remote system, whose `reset` replaces
        the analysis and clears the smoother, it means doing that to objects the
        worker thread is using mid-frame, which is how Reset could leave the run
        wedged. Every other rebuild path in this file already waits first.
        """
        self._sim_idle()
        self.system.reset()
        self.history.reset()
        self.atom_trails.reset()
        self._trail_frame_counter = 0
        self.energy_baseline = None
        self.total_steps = 0
        self.sim_playing = not self.system.spec.playback_controls

    # How long to wait before rebuilding automatically a second time. A value that
    # destroys every fresh state (a temperature far above the melt, say) would
    # otherwise blow up, rebuild, blow up again and leave the app flashing a card
    # forever. One free recovery, then it stops and says so.
    AUTO_RESET_COOLDOWN = 5.0

    def _handle_faults(self):
        """Show what killed the simulation, put it back on its feet, once.

        The two failures look the same from here and are handled the same way: a
        chunk that made LAMMPS raise (`step` latches it), and a rebuild that this
        build would not accept (`reset` falls back and reports what it had to put
        back). Both arrive as a `Fault`; both end with a running simulation and a
        card on screen for three seconds.
        """
        fault = self.system.take_fault()
        if fault is None:
            return
        now = perf_counter()
        if fault.fatal:
            # Nothing is running: only a rebuild brings it back.
            if now - self._last_auto_reset > self.AUTO_RESET_COOLDOWN:
                self._last_auto_reset = now
                # Keep playing if it was playing. `_reset_simulation` pauses on
                # purpose -- somebody pressed Reset and should see the fresh state
                # before it moves -- but nobody pressed anything here, and a demo
                # that silently stops until you find the Play button has still
                # failed in front of an audience.
                was_playing = self.sim_playing
                self._reset_simulation()
                self.sim_playing = was_playing
                # A rebuild that had to fall back reports its own, better-informed
                # fault -- it knows which parameter it put back.
                fault = self.system.take_fault() or fault
            else:
                fault.summary += (" These settings destroy every fresh state -- "
                                  "dial them back, then press R.")
        # Whatever the rebuild settled on is now the truth; the sliders follow it
        # rather than the other way round.
        self._sync_sliders_to_system()
        self.alert.show_fault(fault)

    def _sync_sliders_to_system(self):
        """Move the live-parameter sliders to the values the system actually holds.

        The app pushes sliders into the system every frame, so this is the only way
        a value the system chose for itself (a clamp, or a rebuild's fallback) can
        survive more than one frame.
        """
        values = self.system.live_param_values()
        for key, slider in zip(self.extra_slider_keys, self.extra_sliders):
            if key in values:
                slider.value = max(slider.vmin, min(slider.vmax, values[key]))

    def _draw_overlays(self, renderer):
        """Everything that goes over the sim view, in back-to-front order.

        The alert is last so it is readable even while the connect panel's modal
        wash is up -- a session that failed and a simulation that died are exactly
        the pair of things that can happen at the same moment.
        """
        self.remote_panel.draw(renderer)
        self.alert.draw(renderer)

    def _playback_action(self, name):
        """Apply a Play/Pause/Reset button (or its keyboard shortcut)."""
        if name == "play":
            self.sim_playing = True
        elif name == "pause":
            self.sim_playing = False
        elif name == "reset":
            self._reset_simulation()

    def run(self):
        dt = 1.0 / 60  # seconds; seed value, replaced by the real measured frame time below
        running = True
        self._install_exit_signals()
        # The last line of defence, and the only one that covers a `sys.exit` from
        # somewhere that never entered this loop. `_shutdown` is idempotent, so the
        # ordinary path running it first costs nothing.
        atexit.register(self._shutdown)
        try:
            while running:
                running = self._handle_events(dt)
                dt = self._tick(dt)
        finally:
            self._shutdown()

    # Signals whose default action is to end the process, and which therefore skip
    # every `finally` in this file. `kill` sends the first, a closed terminal the
    # second, a macOS logout both. SIGKILL is not here because it cannot be: that
    # one is what the server's own idle timeout exists for.
    EXIT_SIGNALS = ("SIGTERM", "SIGHUP", "SIGINT")

    def _install_exit_signals(self):
        """Turn a kill, a closed terminal or a logout into a normal exit.

        AN ALLOCATION IS NOT A FILE HANDLE. Everything else this app holds is
        released by the OS when the process goes; a GPU on the other side of the
        country is not, and stays held until Slurm's own hour is up. So the signals
        that would otherwise end the process silently are caught and turned into a
        `SystemExit`, which unwinds through `run`'s `finally` and gives the job
        back on the way out.

        Raised rather than handled in place because the handler runs on the main
        thread at an arbitrary point: unwinding puts the teardown back on the
        thread that owns the window, in the one place that already knows the order
        to do it in.
        """
        def _bail(signum, _frame):
            print(f"[lammps-live] signal {signum} -- closing down")
            raise SystemExit(128 + signum)

        for name in self.EXIT_SIGNALS:
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, _bail)
            except (ValueError, OSError):
                # Not the main thread, or a platform without it. The `finally` and
                # the atexit hook still cover everything that unwinds.
                pass

    def _shutdown(self):
        """Give everything back, once, however the app is ending.

        EACH STEP GUARDED SEPARATELY, and the allocation released first. This used
        to be five statements in a `finally`, which meant that anything raising --
        a joystick whose device had already gone, a stepper re-raising the error
        that ended the run -- skipped every step after it. The one that must not be
        skipped is `remote_panel.release`: it is the call that runs `scancel`, and
        the only one whose failure costs an A100 for the rest of the hour rather
        than a warning on the way out.

        The simulation thread does not have to be idle for it. A remote step reads
        a socket the release has closed, which is answered rather than raised
        (`FrameLink.send` and `take_frame` both treat a closed link as "nothing"),
        so the release goes first and the wait for the worker follows it.
        """
        if self._shut_down:
            return
        self._shut_down = True
        # Ordered by what it costs to skip. `release` is a no-op for a local
        # playground; for a remote one it cancels the job and closes the tunnel.
        for step in (self.remote_panel.release, self._sim_idle, self.source.close,
                     self.system.close, pygame.quit):
            try:
                step()
            except BaseException as exc:               # noqa: BLE001 -- reported
                # Including KeyboardInterrupt: a second Ctrl-C while the teardown
                # is running must not take the rest of the teardown with it.
                print(f"[lammps-live] while closing down: "
                      f"{type(exc).__name__}: {exc}")

    def _sliders(self):
        """Every slider that can be dragged, whatever system is loaded."""
        return [self.temp_slider, self.damping_slider, *self.extra_sliders]

    def _drop_lost_drags(self, event):
        """End any drag the left button is demonstrably no longer holding.

        Every drag here is opened by a MOUSEBUTTONDOWN and closed by the
        matching MOUSEBUTTONUP, so a press whose release never arrives -- the
        window not being frontmost when it happened, an event dropped while the
        app was still starting up and not reading its queue -- leaves a widget
        dragging for good. For the turntable that is not a cosmetic stuck
        highlight: `_handle_orbit_mouse` eats MOUSEMOTION whenever it believes
        it is orbiting, so one phantom camera drag silently swallows the motion
        of every slider drag after it AND the release that should have ended
        them -- the pointer stops working, permanently.

        Two events prove no drag can still be open, and both carry the proof
        themselves rather than asking SDL for global mouse state (which a
        headless driver does not track): a motion with the left button up, and
        a fresh left press, since one button cannot open a second drag.
        """
        released = ((event.type == pygame.MOUSEBUTTONDOWN and event.button == 1)
                    or (event.type == pygame.MOUSEMOTION and not event.buttons[0]))
        if not released:
            return
        self._orbit_dragging = False
        for s in self._sliders():
            s.dragging = False

    def _handle_events(self, dt):
        for event in pygame.event.get():
            self._drop_lost_drags(event)
            if event.type == pygame.QUIT:
                return False
            # The connect panel is modal while it is waiting for a login answer:
            # that answer can be all digits, which are otherwise the playground
            # shortcuts, so it takes the keystrokes before anything else sees them.
            if self.remote_panel.handle_event(event):
                continue
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # Escape leaves fullscreen (ours or a macOS-native space)
                    # first; only quits when already windowed.
                    if self.renderer.is_fullscreen():
                        self._exit_fullscreen()
                    else:
                        return False
                elif event.key == pygame.K_F11:
                    self._toggle_fullscreen()
                elif event.key == pygame.K_TAB:
                    # Shift-Tab walks the picker backwards, Tab forwards.
                    back = event.mod & pygame.KMOD_SHIFT
                    self._cycle_system(-1 if back else 1)
                elif event.key == pygame.K_SPACE:
                    self.sim_playing = not self.sim_playing
                elif event.key == pygame.K_r:
                    self._reset_simulation()
                elif event.key == pygame.K_c and self.orbit_cam is not None:
                    self.orbit_cam.toggle_auto()
                elif event.key == pygame.K_n and self.remote_panel.active:
                    self.remote_panel.toggle()
                elif event.key == pygame.K_b:
                    self._toggle_puller_attached()
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    idx = event.key - pygame.K_1
                    if idx < len(self.systems):
                        self._build_system(self.systems[idx][0])
            elif event.type == pygame.VIDEORESIZE:
                # Window dragged to a new size, or the macOS green button sending
                # it into / out of a native fullscreen space -- relayout to fit.
                self.renderer.handle_resize(event.size)
                self._after_resize()
            elif event.type == pygame.MOUSEWHEEL:
                # Over the sim view of a turntable system the wheel dollies the
                # camera; everywhere else (and on every other system) it stays
                # the temperature dial it has always been. MOUSEWHEEL carries no
                # position of its own, so ask where the pointer is.
                if self.orbit_cam is not None and self._in_sim_view(pygame.mouse.get_pos()):
                    self.orbit_cam.zoom(event.y)
                else:
                    step = self.temp_slider.vmax - self.temp_slider.vmin
                    self.temp_slider.nudge(event.y * config.TEMP_WHEEL_STEP_FRACTION * step)
            elif self._handle_orbit_mouse(event):
                pass          # consumed by the turntable camera
            else:
                # A click on a Play/Pause/Reset button (playback systems) is
                # routed to the playback action and consumes the event, so it
                # never falls through to slider/puller handling below.
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    name = self.renderer.playback_hit(event.pos)
                    if name is not None:
                        self._playback_action(name)
                        continue
                    if self.renderer.bead_color_hit(event.pos):
                        # Through the Choice, so clicking and pushing the stick are
                        # two ways of moving one state -- otherwise the next stick
                        # push would step from whatever the click left behind.
                        self.color_choice.step(1)
                        continue
                # A click on the "Advanced" header flips the group open/closed.
                # When collapsing, cancel any in-progress drag on a now-hidden
                # slider so it can't stay stuck "dragging" (which would keep the
                # puller input suppressed).
                toggle = self.renderer.advanced_toggle_rect
                if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                        and toggle is not None and toggle.collidepoint(event.pos)):
                    self.show_advanced = not self.show_advanced
                    if not self.show_advanced:
                        for s in self.extra_sliders:
                            if s.advanced:
                                s.dragging = False
                        if self.damping_slider.advanced:
                            self.damping_slider.dragging = False
                    continue
                self.temp_slider.handle_event(event)
                # Hidden advanced sliders don't receive events (their rects are
                # parked off-screen while collapsed anyway).
                if not (self.damping_slider.advanced and not self.show_advanced):
                    self.damping_slider.handle_event(event)
                for s in self.extra_sliders:
                    if s.advanced and not self.show_advanced:
                        continue
                    s.handle_event(event)

        # The joystick's buttons and hat are polled here, with the keyboard
        # shortcuts they mirror, rather than in _tick: switching playground
        # rebuilds the system, and _tick reads the spec it is drawing at the top
        # of the frame.
        self._poll_device_buttons()
        self._sync_orbit_camera(dt)
        keys = pygame.key.get_pressed()
        temp_range = self.temp_slider.vmax - self.temp_slider.vmin
        rate = config.TEMP_KEY_RATE_FRACTION * temp_range
        if keys[pygame.K_UP]:
            self.temp_slider.nudge(rate * dt)
        if keys[pygame.K_DOWN]:
            self.temp_slider.nudge(-rate * dt)
        return True

    def _toggle_puller_attached(self):
        """Grab / release the puller (B, or moving the focus off the viewport --
        the joystick trigger is the run switch, on every playground). Released, the
        stick stops driving it and stops feeling it -- so the smoothers, which are
        still carrying the last frames of contact force, are reset rather than
        left to decay a force onto a hand that is no longer holding anything."""
        self._sim_idle()
        self.system.toggle_puller_attached()
        self.ff_smoother.reset()
        self.interaction_smoother.reset()

    def _move_focus(self, move):
        """Run one focus move (`move` is a ControlFocus method) and settle the
        puller around it.

        Leaving the viewport RELEASES the puller, and coming back re-grabs it.
        That is not a convenience: the stick cannot hold a bead against a membrane
        and set a number at the same time, and a bead left attached while the
        stick drives a slider would be dragged across the box by every value
        change. It is exactly the state B toggles, so what the hand feels when the
        focus leaves the scene is what it feels when the bead is let go -- the
        force feedback goes limp because the released puller reports no
        interaction force at all (see modes.py).

        Only a puller THIS released is re-grabbed, so a bead the user let go of
        deliberately stays let go.
        """
        was_viewport = self.focus.on_viewport
        move()
        if self.focus.on_viewport == was_viewport:
            return
        if not self.focus.on_viewport:
            if self.system.puller_attached():
                self._toggle_puller_attached()
                self._focus_released_puller = True
        elif self._focus_released_puller:
            self._toggle_puller_attached()
            self._focus_released_puller = False

    def _move_focus_hat(self, hat):
        """One flick of the hat -> one focus move, laid out like the screen.

        Left / right cross between the two AREAS (the scene, the control panel);
        up / down walk the panel's rows once it holds the focus. See
        control_focus.py for why it is not one flat left/right cycle.

        Diagonals are read on the horizontal axis alone: the hat is an eight-way
        switch and a firm push at a corner is a push at the pane you were reaching
        for, not an instruction to do both.
        """
        # The stick's own up/down axis walks these same rows, and at this instant
        # it may be held right over -- on the viewport that axis was flying the
        # camera. Suspend it, or entering the panel with the stick forward would
        # step a row immediately (see RowStepper.reset).
        self.focus.row_stepper.reset()
        dx, dy = hat
        if dx:
            self._move_focus(self.focus.enter_stops if dx > 0
                             else self.focus.to_viewport)
        elif dy and not self.focus.on_viewport:
            # dy = +1 is forward, away from the hand, and the stops are drawn top
            # to bottom -- so forward is the row ABOVE, one place back in the list.
            self._move_focus(lambda: self.focus.step_stop(-dy))

    def _poll_device_buttons(self):
        """Edge-detect the joystick's buttons and hat, and act on them.

        Held is not pressed: without the edge detection the trigger would flip
        play/pause every frame it is down, and one flick of the hat would sweep
        the whole focus cycle.

        Every action here has a keyboard twin (Space, R, Tab, the number keys),
        which is what keeps the two input modes honest -- the joystick reaches the
        same set of things, and this method is where the mapping is written down:

            hat               move the focus (see _move_focus_hat)
            1 (trigger)       start / stop the simulation
            2                 reset the run to a fresh state
            3 / 4             previous / next playground

        THE TRIGGER IS THE RUN SWITCH ON EVERY PLAYGROUND, and 2 resets every
        playground. It used to depend on which kind of scene was loaded -- the run
        switch on a playback one, grab-the-bead on an interactive one -- which made
        the most prominent control on the device the one you could not predict.
        Running and not running is the thing every scene has in common, so that is
        what the trigger means everywhere; the puller is grabbed and released with
        B, and by moving the focus off the viewport.

        The remote connect panel is modal, so while it is up the only buttons that
        still fire are 3/4: switching away is how you leave the card, and it costs
        the session nothing (the job, the tunnel and the server survive, see
        RemotePanel.detach_system). Everything else -- the focus, the trigger,
        reset -- belongs to a scene that is not running yet. The device state is
        still recorded, so a button held through the panel does not fire the moment
        the panel closes.
        """
        buttons = self.source.poll_buttons()
        hat = self.source.poll_hat()
        fired = buttons - self._prev_buttons
        hat_moved = hat != self._prev_hat
        self._prev_buttons = buttons
        self._prev_hat = hat
        if self.remote_panel.visible:
            self._cycle_system_buttons(fired)
            return

        if hat_moved and hat != (0, 0):
            self._move_focus_hat(hat)
        if config.JOYSTICK_PLAY_PAUSE_BUTTON in fired:
            self.sim_playing = not self.sim_playing
        if config.JOYSTICK_RESET_BUTTON in fired:
            self._reset_simulation()
        # Last, and it returns: switching playground rebuilds the system out from
        # under everything above (and under the caller's `spec`).
        self._cycle_system_buttons(fired)

    def _cycle_system_buttons(self, fired):
        """Buttons 3/4 -> previous / next playground. Split out because these two
        are the one pair that still works behind the connect panel."""
        if config.JOYSTICK_PREV_PLAYGROUND_BUTTON in fired:
            self._cycle_system(-1)
        elif config.JOYSTICK_NEXT_PLAYGROUND_BUTTON in fired:
            self._cycle_system(1)

    def _route_stick(self, jx, jy, yaw, dt):
        """Send this frame's stick deflection where the focus points it, and hand
        back what is left for the puller.

        There are three things the stick can drive and the focus picks exactly
        one, so the other two must read a real zero rather than last frame's
        value:

          * a focused slider -- left/right walks its value, with the deadzone and
            the two speed bands from control_focus.py, while up/down walks the
            focus from row to row (the hat's up/down by another route -- see
            ControlFocus.row_step). Both axes are the panel's while it holds the
            focus, which is what lets a whole demo be driven without the hand
            leaving the stick: pick the row, set the value, pick the next;
          * the turntable camera, on a playground with nothing to pull: the stick
            flies around the box and the twist axis dollies in and out;
          * the puller, which is what a game-mode playground has always done with
            the stick, unchanged.

        Joystick only. The mouse's "deflection" is a pointer position and the
        keyboard's is WASD; neither has a hat to move the focus with, and the
        camera stays theirs to drag.
        """
        if self.input_mode != "joystick":
            self._stick_target = "puller"
            return jx, jy, yaw
        if not self.focus.on_viewport:
            self._stick_target = "slider"
            # Which row first, then its value: a frame that does both would move
            # the value of a row the focus is already leaving.
            step = self.focus.row_step(jy, dt, cross=jx)
            if step:
                self._move_focus(lambda: self.focus.step_stop(step))
            self.focus.drive(jx, dt)
            return 0.0, 0.0, 0.0
        # A turntable on a playback playground: nothing to pull, so the scene is
        # what the stick moves. A game-mode playground that also has a turntable
        # keeps the puller on the stick and leaves the camera to the mouse -- the
        # bead is the point there, and it is the only control with force feedback.
        if self.orbit_cam is not None and self.system.spec.playback_controls:
            self._stick_target = "camera"
            self.orbit_cam.steer(jx, jy, dt)
            self.orbit_cam.steer_zoom(yaw, dt)
            return 0.0, 0.0, 0.0
        self._stick_target = "puller"
        return jx, jy, yaw

    # ---- turntable camera ---------------------------------------------------

    def _in_sim_view(self, pos):
        """Is a window position inside the simulation viewport (not the panel)?"""
        return 0 <= pos[0] < self.renderer.sim_width

    def _handle_orbit_mouse(self, event):
        """Left-drag inside the sim view orbits the turntable camera. Returns
        True if the event was consumed, so it never also reaches the sliders or
        the puller.

        The drag has to START in the sim view: a drag that began on a slider and
        wandered left over the scene is still a slider drag, and grabbing the
        camera out from under it would be a surprise. The Play/Pause/Reset
        buttons are drawn INSIDE the sim view, so they are excluded too -- a
        click on Play is a click on Play, not a camera grab.

        Holding SHIFT pans instead of orbiting -- it slides the scene across the
        view, so an off-centre membrane can be brought to the middle and then
        orbited about. The modifier is read per motion event rather than latched at
        the press, so shift can be taken and released mid-drag."""
        if self.orbit_cam is None:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._in_sim_view(event.pos) and self.renderer.playback_hit(event.pos) is None:
                self._orbit_dragging = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self._orbit_dragging:
                self._orbit_dragging = False
                return True
        elif event.type == pygame.MOUSEMOTION and self._orbit_dragging:
            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                self.orbit_cam.pan(*event.rel)
            else:
                self.orbit_cam.drag(*event.rel)
            return True
        return False

    def _sync_orbit_camera(self, dt):
        """Advance the automatic orbit and push the result onto the camera."""
        if self.orbit_cam is None or self.camera3d is None:
            return
        self.orbit_cam.update(dt)
        # Both, because a pan moves what the camera LOOKS AT, not just where it is.
        self.camera3d.move_to(self.orbit_cam.eye(), self.orbit_cam.target)

    def _tick(self, dt):
        t_frame_start = perf_counter()
        spec = self.system.spec
        ff_profile = spec.force_feedback
        # Pick up whatever the remote session did since the last frame: a completed
        # connection hands its link to the system here, and a link that died reopens
        # the panel with the reason.
        self.remote_panel.update()

        # ---- 1. collect the step launched at the end of the LAST frame -------
        # It has been running while this frame's predecessor was drawn, so this
        # blocks only for whatever of it the drawing did not cover. Nothing may
        # have touched LAMMPS since it was launched -- see stepper.py.
        self.total_steps += self.stepper.wait()
        sim_seconds = self.stepper.wait_seconds

        # ---- 1b. did the simulation die? -------------------------------------
        # BEFORE the sliders are pushed, which is the whole reason it is here: a
        # rebuild that had to put `zeta` back would be handed the value that killed
        # it again, one line further down, by a slider still sitting where the user
        # left it.
        self._handle_faults()

        # ---- 2. push this frame's control inputs into the simulation ---------
        self.system.set_target_temp(self.temp_slider.value)
        self.system.set_puller_damping(self.damping_slider.value)
        for key, s in zip(self.extra_slider_keys, self.extra_sliders):
            self.system.set_extra_param(key, s.value)

        # While actively dragging a slider (which lives in the right-hand
        # panel, off to the side of the sim box), don't also feed that mouse
        # position to the puller as a deflection -- zero the input force
        # instead of letting a slider drag yank the atom. Same for a drag that
        # is orbiting the camera: in mouse mode the pointer position IS the
        # puller's deflection, so swinging the camera by hand would otherwise
        # fling the controlled particle across the box.
        #
        # And the same for the pointer merely RESTING on Play/Pause/Reset, which
        # are drawn inside the sim view: since those buttons went onto the
        # interactive playgrounds too, reaching for Play in mouse mode would
        # otherwise drag the bead to the bottom of the frame on the way. Hovering
        # is enough here, not just a press -- a position control is read every
        # frame, so it is the travel that does the damage, not the click.
        ui_capturing_mouse = (self._orbit_dragging
                              or any(s.dragging for s in self._sliders())
                              or self.renderer.playback_hit(
                                  pygame.mouse.get_pos()) is not None)
        # Device I/O is split into two debug fields, since on the joystick both are
        # blocking HID traffic that belongs in neither sim nor "other": "read" is
        # the stick poll here, "ff" is the force-feedback writes further down.
        read_seconds = 0.0
        ff_seconds = 0.0
        t_in = perf_counter()
        if self.input_mode == "mouse" and ui_capturing_mouse:
            jx, jy = 0.0, 0.0
            yaw = 0.0
        else:
            jx, jy = self.source.poll()
            yaw = self.source.poll_yaw()
        # The thrust lever, read every frame whatever the stick is doing: it is a
        # separate axis driving a separate thing (see view_slice.py), and it is a
        # memory read like the rest of the cached device state.
        lever = self.source.poll_throttle()
        read_seconds += perf_counter() - t_in
        # Whatever holds the joystick's focus takes the stick first; jx/jy/yaw
        # come back zeroed if the camera or a slider took it.
        jx, jy, yaw = self._route_stick(jx, jy, yaw, dt)
        # A released puller is driven by nothing, so the input force IS zero --
        # here, not just inside the mode. Everything downstream reads this: the
        # green input arrow, the header readout, and the joystick's cancellation
        # term all go quiet together, instead of drawing a force on a particle
        # that is not receiving it.
        drive = spec.max_input_force if self.system.puller_attached() else 0.0
        input_fx, input_fy = jx * drive, jy * drive
        # Joystick is a force-feedback loop: the puller is driven mainly by the
        # stick's input force, with the MD interaction force reaching it partly
        # *indirectly* -- it's rendered on the stick (force feedback, below), the
        # user's hand yields, and the resulting deflection changes this input
        # force. Cancel out the fraction (1 - felt) of the measured MD force from
        # what's applied to the atom, leaving `felt` of it as direct contact
        # coupling (fully cancelling it felt too detached). Spread across the
        # puller's atoms via puller_bead_count, since set_input_force applies its
        # force to each. Mouse mode keeps the full direct force-on-atom feel.
        #
        # NOT on a torque drive, where the two axes turn the puller's director
        # instead of pushing it (spec.control_drive; see playground/spec.py). There
        # the argument does not carry over: the user's command is an
        # angular-momentum kick and the force field's restoring torque is integrated
        # by LAMMPS, so the two are never summed and there is no double count to
        # take out. Subtracting one from the other would simply cancel the physics
        # the twist exists to feel -- and, since a restoring torque here can be
        # several times a full-deflection command, invert it.
        if self.input_mode == "joystick" and spec.control_drive != "torque":
            n_beads = max(1, self.system.puller_bead_count())
            md_fx, md_fy = self.system.get_interaction_force()
            cancel = (1.0 - config.JOYSTICK_MD_FORCE_FELT_FRACTION) / n_beads
            self.system.set_input_force(input_fx - cancel * md_fx, input_fy - cancel * md_fy)
        else:
            self.system.set_input_force(input_fx, input_fy)
        # Yaw (joystick twist axis, or Q/E in mouse mode) steers the puller's
        # orientation -- a no-op for a system whose puller is a lone atom, and
        # what twists a membrane bead's director against the tilt term.
        self.system.steer_orientation(yaw, dt)

        # ---- 3. read everything this frame needs, while LAMMPS is idle -------
        # Every readout below hands back a copy, so the worker started in step 4
        # cannot move the data out from under the drawing in step 5.
        t_gather_start = perf_counter()
        pos, vel = self.system.get_puller_state()
        interaction_force = self.system.get_interaction_force()
        temp, press, ke, pe, etotal = self.system.get_thermo_state()
        puller_ke, puller_pe = self.system.get_puller_energy()
        rdf = self.system.get_rdf()
        sim_time_ps = self.system.get_sim_time()
        ids, positions, is_puller, species = self.system.get_all_positions()
        hud_lines = self.system.get_hud_lines()
        # Explicit bonds and the hydrogen-bond overlay are drawn only by the 2D
        # path, so a 3D system was previously paying for two whole-system gathers
        # per frame whose results it then discarded.
        bond_pairs = hbond_pairs = None
        if not spec.render_3d:
            bond_pairs = self.system.get_bond_pairs()
            hbond_pairs = self.system.get_hbond_pairs()
        scene_3d = None
        if spec.render_3d:
            box_bounds_3d = self.system.get_box_bounds_3d()
            # Where the lever has put the cut this frame. Advanced here rather
            # than up with the other device reads because it needs the box and
            # the view direction, and both are only known once the 3D scene is
            # being gathered.
            slice_plane = self.view_slice.update(
                lever, dt, forward=self.camera3d.forward,
                box_bounds=box_bounds_3d)
            ids3d, pos3d, is_puller3d = self.system.get_positions_3d()
            scene_3d = {
                "positions3d": pos3d,
                "dipoles3d": self.system.get_dipoles_3d(),
                "is_puller": is_puller3d,
                "bonds": self.system.get_bonds_3d(),
                # Non-particle spheres: the rod's body. None on every system whose
                # particles are the shape they are drawn as.
                "glyph_spheres": self.system.get_glyph_spheres(),
                "camera": self.camera3d,
                "control_grid": self.system.get_control_grid(),
                "potential_terms": self.system.get_potential_terms(),
                "total_potential_terms": self.system.get_total_potential_terms(),
                # The two-bead scene's term-by-term connector, or None everywhere
                # else (see MDSystem.get_pair_annotation).
                "pair_annotation": self.system.get_pair_annotation(),
                "torque_signals": self.system.get_torque_signals(),
                # The two torques as world vectors, for the two RINGS at the puller
                # (a torque is a rotation in a plane, and that is what gets drawn --
                # see Renderer._draw_torque_ring). None on every playground whose
                # input is a force, which has real force vectors to draw there
                # instead.
                "torque_vectors": self.system.get_torque_vectors(),
                "brightness": self.system.get_bead_brightness(),
                # Only gathered when the colouring is on: it is a whole-system
                # readout, and paying for it to be thrown away every frame is
                # exactly what the 3D path was cleaned up to stop doing.
                "bead_energies": (self.system.get_bead_energies()
                                  if self.renderer.bead_color_energy else None),
                # Same rule, for the same reason: the labelling is a pass over
                # every bead (see clustering.py) and nothing else in the frame
                # wants it, so it is gathered only while it is being painted.
                "bead_clusters": (self.system.get_bead_clusters()
                                  if self.renderer.bead_color_clusters else None),
                "box_bounds": box_bounds_3d,
                "box_periodic": self.system.get_box_periodic(),
                # Static per-bead colours the playground declared (the polymer's
                # own palette against the membrane's banding), or None.
                "bead_tints": self.system.get_bead_tints(),
                # This frame's cut through the scene, or None for the whole box.
                "view_slice": slice_plane,
            }
        gather_seconds = perf_counter() - t_gather_start

        # ---- 4. hand the next step to the worker -----------------------------
        # From here to the next frame's wait(), the simulation is off limits.
        # Every playground steps only while playing -- see `sim_playing` for what
        # differs between them, which is only where the flag starts.
        should_step = self.sim_playing
        # Whether the run is going is pushed into the system, not just used here: a
        # remote system has to tell its server, which would otherwise integrate into
        # a socket nobody is reading. A local one does nothing with it.
        self.system.set_playing(should_step)
        if should_step:
            self.stepper.start(self.system, self.steps_per_frame)

        # ---- 5. force-feedback shaping and drawing, over the running step ----
        shaped_fx, shaped_fy = shape_interaction_force(*interaction_force, ff_profile)
        vel_damp_fx, vel_damp_fy = (
            shape_velocity_damping(*vel, ff_profile, spec.puller_speed_cap, CP_OFFSET_MAX)
            if vel is not None else (0.0, 0.0)
        )
        combined_fx, combined_fy = shaped_fx + vel_damp_fx, shaped_fy + vel_damp_fy
        smooth_fx, smooth_fy = self.ff_smoother.update(combined_fx, combined_fy, dt)
        # Stiffness uses its own smoothed copy of the RAW physical
        # interaction force -- separate from smooth_fx/fy above, which is
        # the device-unit-shaped position signal and would saturate
        # stiffness_threshold/knee almost instantly if reused.
        smooth_ifx, smooth_ify = self.interaction_smoother.update(
            interaction_force[0], interaction_force[1], dt
        )
        stiffness = shape_stiffness(smooth_ifx, smooth_ify, ff_profile, SPRING_STIFFNESS_MAX)
        t_in = perf_counter()
        if self._stick_target == "puller":
            self.source.send_force(smooth_fx, smooth_fy, stiffness)
            self.source.set_damper_coefficient(
                shape_damper_coefficient(smooth_ifx, smooth_ify, ff_profile,
                                         DAMPER_COEFFICIENT_MAX)
            )
        else:
            # Flying the camera or setting a value: a strong, plain centring spring
            # instead of a contact force. Both of those are RATE controls read off
            # the stick's own position, so the deadzone only means "stop" if the
            # stick returns to true centre by itself -- and there is nothing being
            # held, so there is no interaction force to render anyway. The
            # smoothers are dropped rather than left to decay the last frames of
            # contact onto a hand that is no longer holding anything.
            self.source.send_force(0.0, 0.0, SPRING_STIFFNESS_MAX)
            self.source.set_damper_coefficient(
                config.JOYSTICK_CENTERING_DAMPER_FRACTION * DAMPER_COEFFICIENT_MAX)
            self.ff_smoother.reset()
            self.interaction_smoother.reset()
        ff_seconds += perf_counter() - t_in

        if self.energy_baseline is None:
            self.energy_baseline = (ke, pe, etotal)
        ke0, pe0, etotal0 = self.energy_baseline
        self.history.add(self.sim_wall_time, temp=temp, press=press,
                          ke=ke - ke0, pe=pe - pe0, etotal=etotal - etotal0)
        t_min = spec.temperature.vmin
        t_max = spec.temperature.vmax
        heat_fraction = max(0.0, min(1.0, (temp - t_min) / (t_max - t_min)))
        t_in = perf_counter()
        self.source.update_jitter(heat_fraction)
        ff_seconds += perf_counter() - t_in

        puller_speed_m_s = units.speed_to_m_per_s(math.hypot(*vel)) if vel is not None else None
        # Motion trails are a 2D-path overlay, and pure Python over the positions
        # already gathered, so they cost the running step nothing.
        if not spec.render_3d:
            self._trail_frame_counter += 1
            if self._trail_frame_counter % config.TRAIL_SAMPLE_EVERY_N_FRAMES == 0:
                self.atom_trails.add(self.sim_wall_time, ids, positions, is_puller)

        t_render_start = perf_counter()
        self.renderer.show_advanced = self.show_advanced
        self.renderer.draw(
            positions, is_puller, pos,
            (input_fx, input_fy), interaction_force, self.clock.get_fps(),
            spec, self.systems, self.system_key,
            (self.temp_slider, self.damping_slider, *self.extra_sliders),
            (temp, press, ke, pe, etotal), (puller_ke, puller_pe),
            self.history, rdf, heat_fraction=heat_fraction,
            sim_time_ps=sim_time_ps, puller_speed_m_s=puller_speed_m_s,
            atom_trails=self.atom_trails, species=species, bond_pairs=bond_pairs,
            hbond_pairs=hbond_pairs, hud_lines=hud_lines, scene_3d=scene_3d,
            total_steps=self.total_steps, steps_per_frame=self.steps_per_frame,
            debug_line=self._debug_line,
            playback_playing=self.sim_playing,
            puller_attached=self.system.puller_attached(),
            # The cyan frame and the panel's "joystick drives:" line. None on the
            # mouse and keyboard, which have no focus to show.
            control_focus=self.focus if self.input_mode == "joystick" else None,
            # "the GPU is still yours, on that other playground" -- None unless a
            # remote session is being held in the background.
            remote_note=self.remote_panel.standby_note(),
            # Drawn last, inside the renderer, so it lands on top of the 3D scene
            # rather than under the composited frame.
            overlay=self._draw_overlays,
        )
        if self.debug:
            render_seconds = perf_counter() - t_render_start
            # Playgrounds report the time their throttled analysis spent inside
            # step(), so it can be shown separately from the LAMMPS run rather
            # than hiding inside it.
            analysis_seconds = getattr(self.system, "analysis_seconds", 0.0)
            self._update_debug(perf_counter() - t_frame_start, sim_seconds,
                               render_seconds, read_seconds, ff_seconds,
                               analysis_seconds, gather_seconds)

        new_dt = self.clock.tick(60) / 1000.0
        self.sim_wall_time += new_dt
        return new_dt

    def _update_debug(self, work_seconds, sim_seconds, render_seconds,
                      read_seconds, ff_seconds, analysis_seconds=0.0,
                      gather_seconds=0.0):
        """Fold this frame's timings into the smoothed breakdown and rebuild the
        header line for the next frame. 'work' is everything the app does per
        frame except the fps-cap sleep. The device I/O is split into 'read' (the
        stick poll) and 'ff' (the force-feedback writes), both broken out because
        on the joystick they are blocking HID traffic; 'gather' is step 3, reading
        a frame's worth of state out of the system, which grows with the bead
        count; 'other' is the remainder -- force shaping, the history, the panel.

        With the sim/render overlap on (config.OVERLAP_SIM_AND_RENDER), 'sim' is
        no longer the cost of the step: it is how long the frame had to WAIT for
        a step that has been running under the previous frame's drawing. It goes
        to zero whenever the simulation fits entirely under the render, which is
        the point -- what it measures is the part that did not fit.

        'analysis' FOLLOWS THAT SAME RULE, and it did not used to. It is measured
        on the stepper thread, where it runs under the drawing exactly as the step
        does, so charging the frame its whole wall time claimed 150 ms of cost on
        a frame that took 37 -- the breakdown added up to more than the frame and
        pointed at the wrong thing. What lands on the frame is only the part the
        drawing did not cover, which is the part inside the wait; the full wall
        time is worth knowing too, so it is printed after the sum rather than
        inside it."""
        # Both are measured INSIDE step(), on the stepper thread, so what reaches
        # this frame is bounded by what the frame waited for.
        analysis_charged = min(analysis_seconds, sim_seconds)
        sim_only = max(0.0, sim_seconds - analysis_charged)
        other_seconds = max(0.0, work_seconds - sim_seconds - render_seconds
                            - read_seconds - ff_seconds - gather_seconds)
        alpha = 0.1   # EMA weight -- steady enough to read, quick enough to track
        for name, secs in (("sim", sim_only), ("analysis", analysis_charged),
                           ("read", read_seconds), ("ff", ff_seconds),
                           ("gather", gather_seconds),
                           ("render", render_seconds), ("other", other_seconds)):
            self._prof_ms[name] += alpha * (secs * 1000.0 - self._prof_ms[name])
        self._analysis_wall_ms += alpha * (analysis_seconds * 1000.0
                                          - self._analysis_wall_ms)
        total = sum(self._prof_ms.values()) or 1e-9
        def part(name):
            ms = self._prof_ms[name]
            return f"{name} {100.0 * ms / total:2.0f}% ({ms:4.1f}ms)"
        self._debug_line = (
            f"DEBUG  {part('sim')}  {part('analysis')}  {part('read')}  "
            f"{part('ff')}  {part('gather')}  {part('render')}  {part('other')}  "
            f"frame {total:4.1f}ms  "
            # Not part of the sum: what the analysis cost on its own thread,
            # whether or not the drawing covered it. Bigger than its charged
            # share means the overlap is doing its job.
            f"[analysis {self._analysis_wall_ms:5.1f}ms wall]"
        )
