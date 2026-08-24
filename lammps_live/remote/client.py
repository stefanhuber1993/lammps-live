"""RemoteSystem: an MDSystem3D whose simulation is somewhere else.

The app talks to this exactly as it talks to `PlaygroundSystem` -- same interface,
same spec, same readouts -- and cannot tell the difference. `step(n)` waits for the
next frame off the socket instead of integrating; the sliders send their values
instead of issuing LAMMPS commands. Everything else is genuinely the same code:
the analysis, the observables, the energy panels, the RDF and the trajectory
smoothing all run here, on the received frames, out of `lammps_live.playground`.

WHERE THE WORK LANDS, AND WHY IT FITS. Three clocks, not one, and which one a
piece of work is on is the difference between 60 fps and a stutter:

  * THE WINDOW, 60 Hz. The drawn state -- the rattle, the smoothing, the readouts
    the renderer asks for. See `_render_state`.
  * THE WIRE, 20 Hz, on the stepper's worker thread (see stepper.py), so the
    network wait overlaps the drawing of the previous frame. `step()` and the
    decode that follows it, and nothing else: `App._tick` joins this before it
    draws, so whatever sits here is a frame the window did not get.
  * THE MEASURING, whenever it finishes. The observables, the energy panel, the
    g(r) and the cluster labelling, on a thread of their own -- because at the
    50,000 beads this playground exists to show, one labelling is 185 ms and no
    amount of overlapping hides eleven frames. See FrameAnalysis, which is also
    where the numbers are.

FRAMES ARE DROPPED, NEVER QUEUED. The reader thread keeps only the newest frame.
If this machine falls behind -- a hitch, a slow analysis frame, a window resize --
the next `step()` picks up where the simulation actually IS, not where it was three
frames ago. A queue would trade latency for smoothness and, over a link that
cannot be flow-controlled by the renderer, the latency would only ever grow. The
dropped-frame count is on the HUD.

DISCONNECTED IS A NORMAL STATE. Selecting this playground builds a RemoteSystem
with no connection: it holds the empty scene, reports what to do in the HUD, and
every readout answers safely. That is what lets the connect panel exist inside the
running app, and what lets the whole client be tested without a cluster.
"""
import socket
import threading
import time
from collections import deque

import numpy as np

from ..mdsystem import MDSystem3D
from ..playground import forcefield as ff_registry
from ..playground.modes import SimMode
from ..playground.clustering import ClusterTracker, contact_cutoff
from ..playground.faults import Fault
from ..playground.observables import Analysis
from ..playground.rdf import InPlaneRDF, RadialRDF3D
from ..playground.smoothing import TrajectorySmoother
from ..playground.state import Box, FrameState
from ..playground import jitter
from ..playground.jitter import RattleFill
from ..playground.system import JITTER_KEY, SMOOTHING_KEY, make_spec
from . import protocol


class LinkClosed(Exception):
    """The server went away, cleanly or otherwise."""


class FrameLink:
    """One connection to a frame server: a reader thread and a send lock.

    Holds the LATEST frame, not a queue of them (see the module docstring), plus
    the counters the HUD needs to say how the link is doing.
    """

    def __init__(self, sock, welcome):
        self.sock = sock
        self.welcome = welcome
        self.error = None
        self.closed = threading.Event()
        self._new = threading.Event()
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._latest = None            # (header, payload)
        self.received = 0
        self.dropped = 0
        self.bytes_in = 0
        self.rtt_ms = None
        self._pings = {}
        # Simulation-destroying events reported by the server, oldest first. Small
        # and bounded: if more than a handful arrive before anyone looks, the older
        # ones have been superseded by the newer anyway.
        self.faults = deque(maxlen=8)
        # Byte and frame counts over the last second, for the HUD's rate readout.
        self._recent = deque(maxlen=240)
        self._thread = threading.Thread(target=self._read_loop, name="frame-reader",
                                        daemon=True)
        self._thread.start()

    # ---- connecting ---------------------------------------------------------

    # How long to wait for a welcome once the server has said it is BUILDING.
    # Building is LAMMPS' own setup on 10k+ beads -- tens of seconds, and minutes
    # if the size is raised -- and it is bounded by nothing this end can see, so
    # this is a "something has gone wrong" limit rather than an estimate.
    BUILD_TIMEOUT = 600.0

    @classmethod
    def connect(cls, host, port, token, timeout=10.0, on_notice=None,
                playground=None):
        """Open a link, or raise LinkClosed with something a user can act on.

        `timeout` covers reaching the server and its answer -- EXCEPT that a server
        which says it is building first is then given BUILD_TIMEOUT instead. That
        distinction is the whole point: a silent server is broken and should be
        given up on quickly, while one that has told us it is building LAMMPS is
        working, and giving up on it drops a socket the server is about to answer
        (and, worse, makes it throw away the build and start over for the retry).
        `on_notice` receives such messages, so the panel can show the wait.

        `playground` ASKS THE SERVER FOR A PARTICULAR DEMO, and is what lets one
        allocation carry more than one of them: a server already holding a
        different playground throws that simulation away and builds this one
        instead, on the GPU it already has (see server.FrameServer.serve_client,
        and session.RemoteSession.switch_playground, which is the caller). None --
        the default, and what the CLI path uses -- means "whatever you were
        started with", so a server pointed at a playground by its own --playground
        flag is never second-guessed by a client that happens to name it
        differently.
        """
        try:
            sock = socket.create_connection((host, int(port)), timeout=timeout)
        except OSError as exc:
            raise LinkClosed(f"could not reach {host}:{port} -- {exc.strerror or exc}. "
                             f"Is the tunnel up and the server running?") from exc
        protocol.set_socket_options(sock)
        hello = {"t": "hello", "version": protocol.VERSION, "token": token}
        if playground:
            hello["playground"] = str(playground)
        try:
            sock.sendall(protocol.pack(hello))
            while True:
                header, _payload = protocol.recv_message(sock)
                if header is None or header.get("t") != "building":
                    break
                if on_notice is not None:
                    on_notice(str(header.get("msg", "the server is building")))
                sock.settimeout(cls.BUILD_TIMEOUT)
        except (OSError, protocol.ProtocolError) as exc:
            sock.close()
            raise LinkClosed(f"handshake failed: {exc}") from exc
        if header is None:
            sock.close()
            raise LinkClosed("the server closed the connection during the handshake")
        if header.get("t") == "error":
            sock.close()
            raise LinkClosed(str(header.get("msg", "the server refused us")))
        if header.get("t") != "welcome":
            sock.close()
            raise LinkClosed(f"unexpected first message {header.get('t')!r}")
        # Back to blocking with no timeout: create_connection leaves the socket on
        # a 10 s timeout, which would turn every quiet moment on the control
        # channel into a spurious read error.
        sock.settimeout(None)
        return cls(sock, header)

    # ---- reading ------------------------------------------------------------

    def _read_loop(self):
        try:
            while True:
                header, payload = protocol.recv_message(self.sock)
                if header is None:
                    self.error = "the server closed the connection"
                    break
                kind = header.get("t")
                if kind == "frame":
                    with self._lock:
                        if self._latest is not None:
                            self.dropped += 1
                        self._latest = (header, payload)
                        self.received += 1
                        self.bytes_in += len(payload)
                        self._recent.append((time.monotonic(), len(payload)))
                    self._new.set()
                elif kind == "fault":
                    # Queued, not folded into the latest-frame slot: this is the one
                    # kind of message that must not be dropped, and the frame slot
                    # exists precisely to drop things (see take_frame).
                    self.faults.append(header)
                elif kind == "pong":
                    sent = self._pings.pop(header.get("id"), None)
                    if sent is not None:
                        self.rtt_ms = 1000.0 * (time.monotonic() - sent)
                elif kind == "error":
                    self.error = str(header.get("msg"))
                    break
        except (OSError, protocol.ProtocolError) as exc:
            self.error = self.error or f"{type(exc).__name__}: {exc}"
        finally:
            self.closed.set()
            self._new.set()

    def take_frame(self, timeout=0.25):
        """The newest frame, or None if none arrived inside `timeout`."""
        self._new.wait(timeout)
        with self._lock:
            frame, self._latest = self._latest, None
            if frame is None:
                self._new.clear()
        return frame

    def rates(self):
        """(frames/s, MB/s) over the last second."""
        cutoff = time.monotonic() - 1.0
        with self._lock:
            recent = [(t, n) for t, n in self._recent if t >= cutoff]
        if not recent:
            return 0.0, 0.0
        return float(len(recent)), sum(n for _t, n in recent) / 1e6

    # ---- writing ------------------------------------------------------------

    def send(self, message):
        """Send one control message. Swallows a dead socket: the reader thread is
        the one authority on whether the link is up, and a slider must not raise."""
        if self.closed.is_set():
            return False
        try:
            with self._send_lock:
                self.sock.sendall(protocol.pack(message))
            return True
        except OSError as exc:
            self.error = self.error or f"send failed: {exc}"
            self.closed.set()
            return False

    def ping(self):
        ping_id = self.received + 1
        self._pings[ping_id] = time.monotonic()
        # Never let unanswered pings accumulate on a link that stopped answering.
        if len(self._pings) > 8:
            self._pings.clear()
        self.send({"t": "ping", "id": ping_id})

    def close(self, say_goodbye=True):
        if say_goodbye:
            self.send({"t": "bye"})
        self.closed.set()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


class FrameAnalysis:
    """Everything this end measures off a received frame, on a thread of its own.

    THE FRAME PATH MUST NOT WAIT FOR ANY OF THIS. Measured at the demo's 50,000
    beads, on a coarsened configuration rather than the opening gas:

        decode + rattle              4 ms   every frame   <- stays on the frame path
        RDF sample                   5 ms   every frame
        Analysis.update             16 ms   one frame in 4
        cluster labelling          185 ms   one frame in 33

    The bottom three used to run inside `_ingest` as well, i.e. inside `step()`,
    i.e. inside the stepper's join at the top of `App._tick` -- so the labelling's 185 ms was
    eleven frames the window did not draw, arriving every 33rd frame of a 20 fps
    wire. THAT WAS THE HITCH EVERY SECOND AND A HALF. Putting the labelling on the
    stepper thread (which is where it was before this class) did not fix it and
    could not: the drawing only hides work up to a frame's worth, and the frame
    waits for the rest.

    Threading it does not make it cheaper; it makes it nobody's frame. And nothing
    here is frame-synchronous to begin with -- a g(r) averaged over 40 samples,
    observables on a 4-frame cadence, an energy panel on an 8-frame one, a
    labelling held by half a second of hysteresis. What they lose by landing a
    frame or two later is not something there is to see.

    HOW IT STAYS SAFE, which is the whole of the design:

      * The worker OWNS the Analysis, the RDF and the ClusterTracker. Nothing else
        touches them once this object exists, so there is no shared mutable state
        to lock around -- the failure mode that would otherwise be waiting here is
        `energy_panel` reading `_energy` from one frame and `_energy_pairs` from
        the next, which is an IndexError with a plausible-looking traceback.
      * It takes ONE frame at a time, the newest, and drops whatever it could not
        keep up with -- the same rule the link itself follows (see FrameLink), and
        for the same reason: falling behind must cost freshness, never latency.
      * Results are PUBLISHED, never mutated: each pass replaces the whole of what
        the drawing thread reads, so a reader gets the previous answer entire or
        the new one entire and never a half of each.
      * A reconnect or a Reset bumps a generation. Results computed for an older
        one are dropped, and the worker rebuilds its own objects rather than
        having them replaced under it mid-pass.

    The one thing it reads from outside is the parameter set, which the app writes
    a slider into every frame. That needs no lock: the values are floats in a dict
    and the analysis only ever reads them, so the worst a race can do is measure
    one frame with the value from the frame before -- which is what a 20 fps wire
    does to the sliders anyway.
    """

    # The energy panel's heading, which is a property of what is being shown
    # rather than of any one frame.
    PANEL_TITLE = "Whole-system energy -- additive (reduced units)"

    # How long to sit in the wait before looking again. Only `stop()` and a
    # `restart()` with no frame behind it need this to be finite at all, so it is
    # sized for a prompt shutdown rather than for latency: a frame wakes the thread
    # the moment it is submitted.
    POLL = 0.05

    def __init__(self, system):
        self._system = system
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._idle = threading.Event()
        self._stop = threading.Event()
        self._pending = None          # (state, frame, want_clusters, generation)
        self._generation = 0
        self._rebuild = False
        # What the drawing thread reads. Replaced wholesale, never edited.
        self.hud_lines = ()
        self.energy_panel = None
        self.rdf_curve = None
        self.cluster_slots = None
        # What the last pass cost, for the --debug breakdown's wall figure.
        self.seconds = 0.0
        self._build()
        self._idle.set()
        self._thread = threading.Thread(target=self._loop, name="frame-analysis",
                                        daemon=True)
        self._thread.start()

    # ---- what the rest of this end says to it -------------------------------

    def submit(self, state, frame, want_clusters):
        """Hand over a received frame. Returns immediately, always: this is called
        from `_ingest`, which is the frame path."""
        with self._lock:
            self._pending = (state, frame, want_clusters, self._generation)
            self._idle.clear()
        self._new.set()

    def restart(self):
        """Forget the run that has just gone away -- a reconnect, a Reset, a
        playground switch on the far side.

        Everything measured off the old run's frames goes, and so does anything
        the worker is part-way through: its generation is stale from here on. The
        objects themselves are rebuilt on the worker's own thread (the box may have
        changed, and with it the RDF's binning), which is why this only sets a flag
        rather than replacing them here."""
        with self._lock:
            self._generation += 1
            self._pending = None
            self._rebuild = True
            self.hud_lines = ()
            self.energy_panel = None
            self.rdf_curve = None
            self.cluster_slots = None

    def wait_idle(self, timeout=5.0):
        """Block until everything submitted so far has been measured.

        NOT USED BY THE APP, and it must not be -- the point of this class is that
        no frame ever waits. It is for a caller that needs the measurement of a
        particular frame to have happened before it looks: the loopback tests, and
        anything headless that steps a fixed number of frames and then reads the
        observables."""
        return self._idle.wait(timeout)

    def stop(self):
        self._stop.set()
        self._new.set()

    # ---- the thread ---------------------------------------------------------

    def _build(self):
        """The three things the worker owns, for the run it is now looking at."""
        system = self._system
        self.analysis = Analysis(
            system.force_field, system.playground.observables,
            energy_every=system.playground.analysis_energy_every)
        self._rdf = system._make_rdf()
        self._clusters = ClusterTracker(contact_cutoff(system.spec.atom_radius_A))

    def _loop(self):
        while not self._stop.is_set():
            self._new.wait(self.POLL)
            with self._lock:
                # Cleared while the job is taken, so a frame submitted in between
                # costs one spare turn of this loop rather than being missed.
                self._new.clear()
                job, self._pending = self._pending, None
                rebuild, self._rebuild = self._rebuild, False
            t0 = time.perf_counter()
            try:
                if rebuild:
                    self._build()
                if job is not None:
                    state, frame, want_clusters, generation = job
                    results = self._measure(state, frame, want_clusters)
                    self.seconds = time.perf_counter() - t0
                    self._publish(generation, results)
            except Exception as exc:      # noqa: BLE001 -- reported, not raised
                # NOTHING HERE MAY KILL THE THREAD. The picture does not depend on
                # any of it, so the cost of a bad frame is a stale panel; the cost
                # of a dead worker is every panel frozen for the rest of the
                # session, with nothing on screen to say why.
                self.seconds = time.perf_counter() - t0
                print(f"[lammps-live] frame analysis failed: "
                      f"{type(exc).__name__}: {exc}")
            self._mark_idle()

    def _measure(self, state, frame, want_clusters):
        """One pass over one frame. Everything expensive on this end is here."""
        if all(state.box.periodic):
            self._rdf.add(state.positions)
        else:
            self._rdf.add(state.positions[:, :2])
        self.analysis.update(state, self._system.params)
        # The labelling, only while the colouring is actually painting it -- the
        # same asked-recently rule the per-bead energies use, decided by the caller
        # (see RemoteSystem._ingest) because it is the frame counter that knows.
        # On the received positions rather than the drawn ones, which differ by a
        # fraction of a bead radius of synthetic rattle and by nothing at all in
        # which beads are in contact.
        slots = (self._clusters.slots(state.positions, state.box, frame)
                 if want_clusters else None)
        n = max(1, len(state.positions))
        scale = self._system.force_field.energy_scale_per_particle * n
        return (tuple(self.analysis.hud_lines() or ()),
                self.analysis.energy_panel(self.PANEL_TITLE, scale),
                self._rdf.get(), slots)

    def _publish(self, generation, results):
        hud, panel, curve, slots = results
        with self._lock:
            if generation != self._generation:
                return                     # measured for a run that has gone
            self.hud_lines = hud
            self.energy_panel = panel
            self.rdf_curve = curve
            # Held rather than cleared when the colouring is off, so switching it
            # back on paints the last labelling while the next one is computed.
            if slots is not None:
                self.cluster_slots = slots

    def _mark_idle(self):
        """Say so if there is nothing left to measure -- see `wait_idle`."""
        with self._lock:
            if self._pending is None:
                self._idle.set()


class RemoteSystem(MDSystem3D):
    """A playground running elsewhere, drawn here."""

    def __init__(self, playground, preset=None, target=None):
        self.playground = playground
        self.preset = preset
        self.target = (target or playground.remote).resolved()
        self.mode_name = "sim"
        # SimMode supplies every puller-shaped readout its neutral answer, which is
        # the same object the local sim-mode playgrounds use -- so "no puller" means
        # exactly the same thing here as there rather than a second set of stubs.
        self.mode = SimMode()
        self.mode.attach(self)
        self.force_field = ff_registry.get(playground.force_field)(
            **playground.force_field_options)
        self.params = self.force_field.new_params(playground.resolved_params(preset))
        self.scenario = playground.scenario
        self.scenario_params = self.scenario.new_params()
        self.spec = make_spec(playground, "sim", preset)
        self.has_directors = self.force_field.has_directors

        # The cell is known from the scenario without building anything -- which is
        # what lets the camera frame the scene, and the box outline be drawn, before
        # a single frame has arrived. The server's own box replaces it on connect
        # (authoritative: a scenario that lets a barostat settle would differ).
        # `outline`, not `build`: this end wants the cell, the composition and the
        # count, and on the 50,000-bead scenarios placing the particles to find them
        # out is half a second of a frozen window every time the app switches to
        # this playground -- for arrays nothing here ever reads. See
        # Scenario.outline.
        build, natoms = self.scenario.outline(self.scenario_params)
        self.box = build.box
        # THE COMPOSITION IS KNOWN HERE, and it has to be, because none of it
        # travels: the wire carries positions, directors and (on request) energies,
        # and nothing that is the same on every frame. Which species each bead is
        # is exactly that -- fixed when the far end built the very same scenario
        # this line just built -- so it is recovered rather than sent, and the
        # analysis at this end gets the type-aware answers it would otherwise
        # silently skip (a two-species force field with no types reports the whole
        # system as one species; see MesoMemPolymer.energy_terms).
        # A scenario that skipped the placement has no per-particle types to hand
        # over either -- but it only skips it when its composition is uniform, which
        # is the same condition under which `_types` is None anyway. A multi-species
        # scenario's outline carries the real array.
        self._types = (np.asarray(build.types, dtype=int)
                       if self.force_field.n_types > 1 and len(build.types) else None)
        self._tints = self.scenario.render_tints(self.scenario_params)
        self.natoms = int(natoms)
        self.all_ids = np.arange(1, self.natoms + 1)
        self.bonds = []
        self.brightness = None
        self.controlled_index = None
        self.controlled_id = None

        # The measuring half of this end -- the observables, the energy panel, the
        # g(r) and the cluster labelling -- on a thread of its own, because at this
        # size none of it fits in a frame. See FrameAnalysis.
        self._frame_analysis = FrameAnalysis(self)
        self._smoother = TrajectorySmoother()
        self._smoothing_tau = 0.0
        # The other half of the drawn-state filtering, and the one that only a
        # remote system needs: the wire runs slower than the screen, so something
        # has to move between the frames it delivers. See playground/jitter.py.
        self._rattle = RattleFill()
        self._jitter_strength = jitter.DEFAULT_JITTER
        self._render_cache = None
        self._render_frame = -1
        # `_render_state` is called two or three times per DRAWN frame (positions,
        # directors, the 2D readout) and must advance the rattle exactly once, so
        # the cache is keyed on wall time as well as on the received frame. Any
        # calls closer together than this share a result; the next drawn frame is
        # a whole 60 Hz period later and gets a fresh one.
        self._render_wall = 0.0
        # Wall seconds between the last two received frames. The rattle needs no
        # such thing -- it runs on wall time directly -- but the smoother's weight
        # is quoted in SIMULATED time, so converting a drawn frame's share of it
        # needs to know how long a wire frame lasts here.
        self._wire_period = 0.0
        self._frame_wall = 0.0
        self._energy_cache = None
        self._energy_render_frame = -1
        # The cluster colouring is computed at THIS end from the positions off the
        # wire. Nothing about it has to travel: it is a fact about the geometry
        # already in hand, and the far end has enough to do. It is also the most
        # expensive thing this end does -- 185 ms at 50,000 beads -- which is why
        # it lives on the analysis thread with the rest of the measuring, and why
        # it runs only while the colouring is on (see clustering.py, FrameAnalysis).
        self._clusters_asked_frame = -999
        self._frame = 0
        self._state = None
        self._energies = None
        self._thermo = (0.0, 0.0, 0.0, 0.0, 0.0)
        self._sim_time = 0.0
        self._last_step_dt = 0.0
        self._unstable = None
        self._seq = 0
        self.wait_seconds = 0.0

        self.link = None
        self.status = "not connected"
        self._playing = False
        self._target_temp = self.spec.temperature.default
        self._sent_params = {}
        # THE VALUES RESET GOES BACK TO. Taken here, before anything has touched a
        # slider, and after the preset has been folded in (`resolved_params` above)
        # -- so Reset returns to the preset when one was asked for, exactly as it
        # does locally. Kept as raw values rather than clamped ones for the same
        # reason PlaygroundSystem._snapshot_params does: a clamp is re-applied on
        # read, so storing the clamped number would make the restore lossy the
        # moment its dependency moved.
        self._initial_params = dict(self.params.values)
        self._initial_temp = self._target_temp
        self._fault = None             # the far side's last simulation-killing event
        # Which received frame last had `get_bead_energies` called against it. The
        # request is derived from that with a couple of frames of hysteresis rather
        # than from an edge, because how many times `step()` is called per drawn
        # frame is not something this end gets to assume (a stepper that polls, a
        # paused run that only draws) and an edge would flap between the two states.
        self._energy_asked_frame = -1000
        self._energy_requested = False
        self._ping_countdown = 0
        # Waiting for the far side to finish rebuilding after a Reset -- see reset()
        # and _ingest, which clears it when a frame from the new run arrives.
        self._resetting = False
        self._reset_started = 0.0

    # ---- the connection -----------------------------------------------------

    def connect(self, host=None, port=None, token="", timeout=10.0):
        """Open a link to a server (the CLI path). The connect panel uses `attach`
        instead, because it has already opened the socket through its tunnel."""
        link = FrameLink.connect(host or "127.0.0.1",
                                 port or self.target.local_port, token, timeout,
                                 on_notice=lambda msg: print(f"[lammps-live] {msg}"))
        self.attach(link)
        return link

    def attach(self, link):
        """Adopt an open link and align the far end with the local state."""
        self.detach()
        self.link = link
        welcome = link.welcome or {}
        box = welcome.get("box")
        if box:
            self.box = Box(tuple(box["lo"]), tuple(box["hi"]),
                           tuple(bool(p) for p in box["periodic"]))
        n = int(welcome.get("natoms") or self.natoms)
        if n != self.natoms:
            self.natoms = n
            self.all_ids = np.arange(1, n + 1)
        self._sim_time = float(welcome.get("sim_time") or 0.0)
        self._unstable = None
        self._fault = None
        self._energies = None
        self._energy_cache = None
        self._energy_render_frame = -1
        # Nothing measured off the last connection's frames belongs to this one:
        # the box may be a different size, and the labelling's colours are a fact
        # about a configuration that has gone. The worker rebuilds around the new
        # box on its own thread -- see FrameAnalysis.restart.
        self._frame_analysis.restart()
        self._smoother.reset()
        # A fresh connection is a fresh scene: nothing measured off the previous
        # one's frames (the rattle's amplitude, the wire's period) carries over.
        self._rattle.reset()
        self._wire_period = 0.0
        self._frame_wall = 0.0
        self._render_wall = 0.0
        self.status = (f"{welcome.get('host', '?')} "
                       f"[{welcome.get('profile', '?')}], "
                       f"{self.natoms:,} beads, job {welcome.get('slurm_job') or '-'}")
        # The far end starts from its own defaults, so push everything this end is
        # currently showing: the sliders, the temperature, the play state, and the
        # frame configuration. Anything not sent here would silently disagree with
        # the panel the user is looking at.
        link.send({"t": "config", "fps": self.target.fps,
                   "codec": self.target.codec, "free_run": self.target.free_run,
                   "energies": self._energy_requested})
        link.send({"t": "temp", "value": self._target_temp})
        for param in self.params.live_params():
            value = float(self.params[param.name])
            self._sent_params[param.name] = value
            link.send({"t": "set", "key": param.name, "value": value})
        link.send({"t": "play" if self._playing else "pause"})
        return link

    def detach(self, say_goodbye=True):
        if self.link is not None:
            self.link.close(say_goodbye=say_goodbye)
            self.link = None
        self.status = "not connected"
        # Nothing is going to answer a reset that was in flight when the link went.
        self._resetting = False

    @property
    def connected(self):
        return self.link is not None and not self.link.closed.is_set()

    def link_error(self):
        """Why the link went down, once it has -- for the connect panel."""
        if self.link is not None and self.link.closed.is_set():
            return self.link.error or "the connection closed"
        return None

    # ---- what the app drives ------------------------------------------------

    def set_playing(self, playing):
        """Play/Pause, forwarded. Without this the server would keep integrating
        into a socket nobody is reading -- an idle A100 is cheaper than a busy one
        computing frames that are thrown away."""
        playing = bool(playing)
        if playing == self._playing:
            return
        self._playing = playing
        if self.connected:
            self.link.send({"t": "play" if playing else "pause"})

    def set_target_temp(self, T):
        t_min, t_max = self.playground.temperature
        T = max(t_min, min(t_max, float(T)))
        if T == self._target_temp:
            return
        self._target_temp = T
        if self.connected:
            self.link.send({"t": "temp", "value": T})

    def set_extra_param(self, key, value):
        """A live parameter moved. Held locally too, because the energy panels are
        computed HERE and must use the coefficients the far end is running."""
        if key == SMOOTHING_KEY:
            self._smoothing_tau = max(0.0, float(value))
            return
        if key == JITTER_KEY:
            self._jitter_strength = max(0.0, float(value))
            return
        if not self.params.has(key):
            return
        if not self.params.set(key, value):
            return
        # The clamped value, not the raw slider value: `wc` is capped at `rc`, and
        # sending the uncapped number would make the far end run coefficients the
        # local energy decomposition is not using.
        clamped = float(self.params[key])
        if self._sent_params.get(key) == clamped:
            return
        self._sent_params[key] = clamped
        if self.connected:
            self.link.send({"t": "set", "key": key, "value": clamped})

    def _take_wire_fault(self, message):
        """Adopt a fault the server reported.

        THE REVERTED VALUES MATTER AS MUCH AS THE MESSAGE. When the far side's
        rebuild had to put a parameter back, this end is still holding the value
        that killed it -- in `params`, which the energy panels are computed from,
        and in `_sent_params`, which decides what is worth sending. Left alone, the
        panels would describe coefficients nobody is running, and dragging the
        slider back to the bad value would send nothing at all (because this end
        thinks it already sent it) so the picture and the sliders would disagree
        with no way to resync.
        """
        fault = Fault.from_message(message)
        if fault is None:
            return None
        for key, value in fault.reverted.items():
            if self.params.has(key):
                self.params.set(key, value)
                self._sent_params[key] = float(self.params[key])
        return fault

    def take_fault(self):
        """The next fault the far side reported, once. See PlaygroundSystem.

        Drained here rather than when the message arrives, because adopting one
        writes to `params` and `_sent_params` -- and the message arrives on the
        link's reader thread while the caller of this is the app's own.
        """
        if self._fault is None and self.link is not None and self.link.faults:
            self._fault = self._take_wire_fault(self.link.faults.popleft())
        fault, self._fault = self._fault, None
        return fault

    def live_param_values(self):
        """The effective value of every live parameter, for the sliders to follow."""
        return {p.name: float(self.params[p.name])
                for p in self.params.live_params()}

    def _restore_declared_params(self):
        """Back to the values this playground declares, here and over the wire.

        `set_extra_param` is the one door: it clamps, it keeps `_sent_params`
        honest, and it skips the send when the far side already has the value --
        all of which a hand-rolled loop here would have to repeat. The temperature
        goes back through its own setter for the same reason.
        """
        for name, value in self._initial_params.items():
            if self.params.has(name):
                self.set_extra_param(name, value)
        self.set_target_temp(self._initial_temp)

    def reset(self, restore_params=True):
        """Put the far side back how it started: the declared parameters, and a
        fresh state.

        The rebuild happens THERE, and it used to be the slowest thing in this app
        by two orders of magnitude: 44 seconds measured for this playground's 50,000
        beads, against 43 for the initial build, which is the same work -- and
        essentially all of it was `create_atoms random ... overlap`, LAMMPS placing
        particles on the host one at a time and rejecting the ones that land too
        close. That is gone; the placement is a bulk numpy pass now (see
        state.random_points_min_separation) and the same rebuild measures HALF A
        SECOND. A scenario with a relaxation in it still pays for the relaxation.

        This still latches `_resetting`, and the HUD still says the far side is
        rebuilding with the seconds on it, until a frame from the new run arrives
        (see `_is_new_run`) -- because half a second is not zero over a tunnel from
        Amsterdam, and because the alternative is the picture sitting on the last
        frame of the OLD run, which is indistinguishable from a Reset that did
        nothing and, with only a static notice, from one that hung.

        Called only with the simulation thread idle (App._reset_simulation waits
        first): it clears the drawn-state filters, which the stepper thread reads
        mid-frame. The analysis is no longer among them -- it owns its own state and
        is told to start over rather than having it swapped underneath (see
        FrameAnalysis.restart).

        THE PARAMETERS GO BACK TOO, matching PlaygroundSystem.reset -- see there for
        why. On this end that means two things rather than one: the local `params`
        (which the energy panels are computed from) and the values the far side is
        running, which have to be re-sent because the server keeps whatever it was
        last told. They are sent BEFORE the reset message, so the rebuild over there
        happens with the values that are going back on the sliders rather than with
        the ones being abandoned -- and so that a value the far side's Kokkos build
        rejects is gone before the rebuild that would trip over it.

        `restore_params=False` keeps whatever the sliders hold, which is what the
        app's automatic recovery after a blow-up wants -- see PlaygroundSystem.reset
        for why. On that path nothing is re-sent either: the server's own rebuild
        ladder decides, and reports what it had to put back.
        """
        if restore_params:
            self._restore_declared_params()
        self._unstable = None
        self._energies = None          # the old run's colours are not the new box's
        self._energy_cache = None
        self._energy_render_frame = -1
        self._smoother.reset()
        # Both filters, and for the same reason: the new run's coordinates have
        # nothing to do with the old run's, so a carried-over wobble or average
        # would be drawn on top of a scene it was never measured from.
        self._rattle.reset()
        self._wire_period = 0.0
        self._frame_wall = 0.0
        self._render_wall = 0.0
        # The measuring side goes with them: a rolling g(r), an observable average
        # and a set of cluster colours are all statements about the run that is
        # being thrown away.
        self._frame_analysis.restart()
        self._playing = False
        # LATCHED ONLY IF THE ASK ACTUALLY WENT OUT. `send` answers False on a dead
        # socket rather than raising (a slider must not end the app), so latching
        # first would leave the HUD saying the far side is rebuilding when nothing
        # over there ever heard the request.
        if self.connected and self.link.send({"t": "reset"}):
            self._resetting = True
            self._reset_started = time.perf_counter()

    def _is_new_run(self, seq):
        """Is this frame the first one the far side sent after a rebuild?

        The server restarts its sequence at 0 on a reset and numbers the first
        frame of the new run 1, so a number at or below the one we were on is the
        giveaway -- and that test ALONE HAS A HOLE IN IT. A Reset pressed before
        this system had ingested anything leaves `_seq` at 0, and no sequence
        number is ever <= 0, so nothing would ever clear the notice: a perfectly
        healthy run streamed on behind "rebuilding from a fresh state..." with
        frames arriving the whole time and no way out short of reconnecting. Which
        is the worst possible place to put that, because Reset is the button
        somebody reaches for when the picture already looks wrong.

        So a literal 1 counts as well. The cost of the extra test is a notice that
        clears one frame early if a frame from the OLD run happened to be numbered
        1 and was still in flight -- a fresh connection's very first frame, nothing
        else -- and a notice that goes away a moment early is not a failure mode.
        """
        return seq <= self._seq or seq == 1

    def set_input_force(self, fx, fy):
        pass

    def set_puller_damping(self, gamma):
        pass

    def steer_orientation(self, rate, dt):
        pass

    # ---- the frame ----------------------------------------------------------

    # The longest this end will block waiting for a frame. THIS IS WHAT DECOUPLES
    # THE WINDOW FROM THE WIRE, and it is the whole reason a 20 fps wire is usable
    # at all: `App._tick` begins by waiting for the step launched under the last
    # frame's drawing, so as long as `step()` blocks for a wire period, the app
    # loop runs at the wire's rate and no amount of filling in between frames can
    # help. Returning promptly instead lets the window keep its own 60 Hz and pick
    # frames up as they land.
    #
    # Not zero, because a stepper that returns instantly spins a core; a short
    # sleep is what a "nothing yet" answer costs. Sized under half a 60 Hz frame
    # so a frame that arrives during one is picked up on the very next tick.
    FRAME_POLL = 0.006

    def step(self, n):
        """Take in the next frame if one has arrived, and do not wait long if not.

        Called on the stepper's worker thread, which is what puts both the network
        wait and the analysis alongside the drawing rather than in front of it.
        `n` is ignored: how far the simulation advances per frame is the server's
        decision, and it reports it (the `dt` on each frame).

        Most calls return with nothing, and that is the normal case rather than a
        failure: at a 20 fps wire and a 60 fps window, two drawn frames in three
        have no new state behind them. What they draw instead is `_render_state`'s
        business.
        """
        if not self.connected:
            # Nothing to wait for. Sleep out a frame rather than spinning, so a
            # disconnected system does not burn a core in the stepper thread.
            time.sleep(1.0 / 60)
            return
        self._sync_energy_request()
        self._ping_countdown -= 1
        if self._ping_countdown <= 0:
            self._ping_countdown = 60
            self.link.ping()
        t0 = time.perf_counter()
        frame = self.link.take_frame(timeout=self.FRAME_POLL)
        self.wait_seconds = time.perf_counter() - t0
        if frame is not None:
            self._ingest(frame)

    ENERGY_REQUEST_HOLD = 3        # frames of asking-nothing before it is dropped

    def _sync_energy_request(self):
        """Ask for per-bead energies only while the bead colouring is painting them.

        They are a tenth of the frame's bytes and the server has to gather them from
        a per-atom compute, so the request follows whether `get_bead_energies` is
        being called -- i.e. whether the renderer is actually using them. Costs a
        frame or two of lag on the toggle, and nothing at all while it is off.
        """
        wanted = (self._frame - self._energy_asked_frame) < self.ENERGY_REQUEST_HOLD
        if wanted != self._energy_requested:
            self._energy_requested = wanted
            self.link.send({"t": "config", "energies": wanted})

    def _ingest(self, frame):
        """Take in one frame: decode it, and hand it to the analysis thread.

        WHAT IS AND IS NOT ALLOWED IN HERE is the thing to keep. This runs on the
        frame path -- inside `step()`, which `App._tick` joins before it draws --
        so it holds only the work the picture itself needs: the decode, and the
        rattle's amplitude. Everything that MEASURES goes to FrameAnalysis, which
        has its own thread and its own clock, because at 50,000 beads a single
        labelling is eleven frames long and no scheduling hides that.
        """
        header, payload = frame
        codec = header.get("codec", protocol.DEFAULT_CODEC)
        arrays = protocol.decode_frame(
            header.get("arrays") or [], payload, self.box,
            energy_range=self.spec.render_style.energy_range, codec=codec)
        positions = arrays.get("positions")
        if positions is None:
            return
        n = len(positions)
        if n != len(self.all_ids):
            self.all_ids = np.arange(1, n + 1)
            self.natoms = n
        self._state = FrameState(positions=positions,
                                 directors=arrays.get("directors"),
                                 types=self._frame_types(n), ids=self.all_ids,
                                 box=self.box)
        self._energies = arrays.get("energies")
        seq = int(header.get("seq") or 0)
        if self._resetting and self._is_new_run(seq):
            self._resetting = False
        self._seq = seq
        self._sim_time = float(header.get("sim_time") or 0.0)
        self._last_step_dt = float(header.get("dt") or 0.0)
        thermo = header.get("thermo")
        if thermo and all(np.isfinite(v) for v in thermo):
            self._thermo = tuple(float(v) for v in thermo)
        self._unstable = header.get("unstable")
        now = time.monotonic()
        if self._frame_wall > 0.0:
            measured = now - self._frame_wall
            # Averaged, and only over plausible values: a frame that arrived after
            # a stall, a pause or a Reset is not evidence about the wire's rate.
            if 0.001 <= measured <= 1.0:
                self._wire_period = (measured if self._wire_period <= 0.0
                                     else self._wire_period
                                     + 0.2 * (measured - self._wire_period))
        self._frame_wall = now
        self._frame += 1
        self._render_cache = None
        # The rattle's amplitude is measured off the frames that DO arrive, so
        # this is where it learns how much motion the wire is dropping. Fed the
        # raw received state, before any of the drawn-state filtering.
        self._rattle.observe(self._state)
        # And off to the measuring thread: the RDF sample, the observables, the
        # energy panel and -- only while the colouring is painting it, on the same
        # asked-recently rule the per-bead energies use -- the cluster labelling.
        # This call does not wait for any of it.
        self._frame_analysis.submit(
            self._state, self._frame,
            (self._frame - self._clusters_asked_frame) < self.ENERGY_REQUEST_HOLD)

    def _ensure_current(self):
        """While the run is paused, keep the picture up to date from the readouts.

        Paused, the app does not call `step()` at all (see App._tick), so nothing
        would take in the frame that shows the state the server is holding -- and
        the scene would stay empty after connecting until Play was pressed. While
        playing, `step()` has already taken the newest frame and this does nothing.
        """
        if self._playing or not self.connected:
            return
        frame = self.link.take_frame(timeout=0.0)
        if frame is not None:
            self._ingest(frame)

    # Two calls to `_render_state` closer together than this are taken to be the
    # same drawn frame (they are: positions, directors and the 2D readout all
    # land within microseconds of each other). A 60 Hz frame is 16.7 ms, so this
    # separates "the same frame asking again" from "the next frame" with an order
    # of magnitude of headroom either side.
    RENDER_COALESCE = 0.002

    def _render_state(self):
        """The frame as drawn: the received state, rattled and optionally smoothed.

        Mirrors PlaygroundSystem._render_state, including that NOTHING which
        measures comes through here -- but with one difference that only a remote
        system has, and it is the reason this is no longer a straight cache on the
        received frame: THIS RUNS AT THE RATE THE SCREEN REFRESHES, not at the rate
        frames arrive. The wire runs at 20 fps and the window at 60, so between two
        received frames this is asked three times and must hand back three
        different pictures or the scene is a slideshow.

        Order matters and is the one thing to preserve here: rattle first, smooth
        second, so that the Smoothing slider removes the synthetic motion exactly
        as it removes the real kind.
        """
        self._ensure_current()
        state = self._state
        if state is None:
            return None
        now = time.monotonic()
        fresh = (self._render_frame != self._frame
                 or self._render_cache is None
                 or (now - self._render_wall) >= self.RENDER_COALESCE)
        if not fresh:
            return self._render_cache

        # The wall-clock slice this drawn frame covers. Clamped because the first
        # frame after a connect, a window resize or a Reset can be an arbitrarily
        # long gap, and neither filter should be handed one: the rattle would jump
        # to a fresh independent draw (a visible twitch) and the smoother would
        # take a step so large it discards its own history.
        wall_dt = 0.0 if self._render_wall <= 0.0 else min(now - self._render_wall,
                                                           4.0 / 60.0)
        self._render_wall = now
        self._render_frame = self._frame

        share = self._render_share(wall_dt)
        drawn = self._rattle.apply(state, self._jitter_strength,
                                   wall_dt, share)

        if self._smoothing_tau <= 0.0:
            if self._smoother.active:
                self._smoother.reset()
            self._render_cache = drawn
            return drawn
        # The smoother's weight comes from how much SIMULATED time a frame
        # advanced, and it is now being applied once per DRAWN frame rather than
        # once per received one -- so it must be given this frame's share of the
        # wire frame's sim time, not the whole of it. Without the split, a 20 Hz
        # wire drawn at 60 would apply three full steps of the filter per frame of
        # physics and smooth three times as hard as the slider says.
        self._render_cache = self._smoother.apply(
            drawn, self._smoothing_tau, self._sim_dt_share(wall_dt))
        return self._render_cache

    def _render_share(self, wall_dt):
        """What fraction of one wire frame this drawn frame covers.

        1.0 whenever the wire is not yet measured, or is slower than the window is
        drawing -- the fallback both filters below want, since it says "this frame
        stands for the whole of a wire frame", which is exactly true when they are
        arriving no faster than they are drawn.
        """
        period = self._wire_period
        if period <= 0.0 or wall_dt <= 0.0:
            return 1.0
        return min(wall_dt / period, 1.0)

    def _sim_dt_share(self, wall_dt):
        """How much of the last wire frame's simulated time this drawn frame is.

        Falls back to the whole of it when there is no measured wire period yet
        (the first frames after a connect), which is the pre-existing behaviour
        and errs toward smoothing slightly too hard for a fraction of a second
        rather than not at all.
        """
        return self._last_step_dt * self._render_share(wall_dt)

    # ---- readouts -----------------------------------------------------------

    def _empty_positions(self):
        return np.zeros((0, 3))

    def get_positions_3d(self):
        state = self._render_state()
        if state is None:
            return np.zeros(0, dtype=int), self._empty_positions(), np.zeros(0, bool)
        return state.ids, state.positions, np.zeros(len(state.positions), bool)

    def get_dipoles_3d(self):
        state = self._render_state()
        if state is None or state.directors is None:
            n = 0 if state is None else len(state.positions)
            return np.zeros((n, 3))
        return state.directors

    def get_all_positions(self):
        state = self._render_state()
        if state is None:
            return (np.zeros(0, dtype=int), np.zeros((0, 2)), np.zeros(0, bool), None)
        return (state.ids, state.positions[:, :2],
                np.zeros(len(state.positions), bool), None)

    def get_bead_energies(self):
        """The per-bead energies from the newest frame that carried them, smoothed
        the same way the drawn positions are (see PlaygroundSystem._smooth_energies).

        None until some have actually arrived -- NOT an array of zeros, which is
        what this returned at first and which is a lie the colour ramp believes:
        zero sits at the TOP of the energy range, so a scene that had simply not
        been sent its energies yet came up painted uniformly white, as if every bead
        were free. None means "no data", the renderer keeps the director banding,
        and the colours appear when the numbers do -- one or two frames later, since
        asking is itself how the request gets made (see _sync_energy_request).
        """
        self._energy_asked_frame = self._frame
        if self._energies is None:
            return None
        if self._smoothing_tau <= 0.0:
            return self._energies
        if self._energy_render_frame != self._frame:
            self._energy_cache = self._smoother.smooth_scalar(
                "bead_energy", self._energies, self._smoothing_tau,
                self._last_step_dt)
            self._energy_render_frame = self._frame
        return self._energy_cache

    def get_bead_clusters(self):
        """Per-bead cluster colour slot, computed here from the drawn positions.

        Same call and same meaning as PlaygroundSystem.get_bead_clusters -- and
        unlike the energies it needs nothing from the far end, so there is no
        request to make and no lag on the toggle beyond the frame it takes for
        the ask to reach the labelling.

        ASKING IS ALL THIS DOES -- and the ask is the whole of how the labelling
        gets scheduled, since it is the most expensive thing this end computes and
        nothing else in the frame wants it. What comes back is the last labelling
        the analysis thread finished (see FrameAnalysis), so the drawing thread
        pays an attribute read.

        Guarded on the length, not trusted: the slots were computed off a frame
        that may be a run older than the one being drawn, and a run switch changes
        the bead count. A mismatch means "no colours yet", which the renderer
        already handles by keeping the director banding, rather than a labelling
        applied to the wrong beads.
        """
        self._clusters_asked_frame = self._frame
        slots = self._frame_analysis.cluster_slots
        if slots is None or self._state is None:
            return None
        return slots if len(slots) == len(self._state.positions) else None

    def _frame_types(self, n):
        """The species of each bead in a frame of `n` of them, or None.

        Guarded on the length rather than trusted, because the two ends agreeing
        is exactly the thing that could stop being true -- a server built from a
        different revision of this package, or a scenario whose count depends on
        something the client resolved differently. A mismatch degrades to "no
        types", which every consumer already handles, instead of mislabelling
        half the system.
        """
        if self._types is None or len(self._types) != n:
            return None
        return self._types

    def get_bead_tints(self):
        # Static, and derived from this end's own build of the scenario -- see
        # __init__. Guarded on the count for the same reason _frame_types is.
        if self._tints is None or len(self._tints) != self.natoms:
            return None
        return self._tints

    def get_bead_brightness(self):
        return None

    def get_bonds_3d(self):
        return []

    def get_glyph_spheres(self):
        """The rod's body, derived at THIS end.

        Nothing has to come down the wire for it: the shape is a function of the
        force field's own parameters and the frame's positions and directors, both
        of which this end already has. Same call as the local system makes, off the
        same render state, so a remote rod would draw identically.
        """
        return self.force_field.glyph_spheres(self._render_state(), self.params)

    def get_thermo_state(self):
        self._ensure_current()
        return self._thermo

    def get_sim_time(self):
        return self._sim_time

    def get_rdf(self):
        """The rolling g(r). A pure read: the sampling happens on the analysis
        thread, and so does the averaging (see FrameAnalysis).

        It used to sample here, which put an O(max_atoms^2) pair pass -- 5 ms at
        this scale -- on the drawing thread once every `sample_every` DRAWN frames.
        On a 60 Hz window in front of a vsync'd flip that is 5 ms the frame does
        not have, for a plot that is a rolling average over dozens of frames and
        cannot tell which thread fed it.
        """
        return self._frame_analysis.rdf_curve

    def get_potential_terms(self):
        return None                     # no controlled particle in sim mode

    def get_total_potential_terms(self):
        # Built on the analysis thread, from the pair list its own energy terms
        # were computed with -- which is what keeps the panel's masking honest
        # across the two cadences (see Analysis.energy_panel).
        return self._frame_analysis.energy_panel

    def get_hud_lines(self):
        """What the scene overlay says. The link's own state belongs here: when
        frames stop arriving, the picture simply freezes, and without a line saying
        so that is indistinguishable from a simulation that has stopped moving."""
        if not self.connected:
            error = self.link_error()
            # Short lines: the HUD sits bottom-left, and the Play/Pause row starts
            # about 240 px in, so anything longer runs under the buttons.
            return ["REMOTE: not connected",
                    (error[:40] if error else "press N to connect")]
        lines = []
        if self._resetting:
            # WITH THE CLOCK ON IT. A rebuild there is a whole LAMMPS setup -- the
            # plugin, the placement, the neighbour lists, and any relaxation the
            # scenario asks for -- during which nothing comes back and the picture
            # does not move. It is well under a second on the assembly playgrounds
            # now (it was 44 s; see `reset`), and still seconds on one with a settle
            # in it. Without a number the only two explanations available to whoever
            # is watching are "slow" and "hung", and they look identical; with one, a
            # count that is still going up is an answer.
            waited = time.perf_counter() - self._reset_started
            lines.append(f"REMOTE: rebuilding from a fresh state... {waited:.0f}s")
        if self._unstable:
            lines += ["SIMULATION UNSTABLE on the remote node -- these parameters "
                      "destroyed it.", str(self._unstable),
                      "Dial the sliders back, then press R to rebuild it there."]
        lines += list(self._frame_analysis.hud_lines)
        fps, mbs = self.link.rates()
        rtt = f"{self.link.rtt_ms:.0f} ms" if self.link.rtt_ms else "-"
        # One short line: the HUD stack is already three observables tall here, and
        # it grows upward from the bottom-left corner where the debug breakdown also
        # lives.
        lines.append(f"link {fps:.0f} f/s, {mbs:.2f} MB/s, {rtt}, "
                     f"{self.link.dropped} drop")
        return lines

    def get_puller_state(self):
        return self.mode.puller_state()

    def get_puller_energy(self):
        return None, None

    def get_interaction_force(self):
        return self.mode.interaction_force()

    def get_torque_signals(self):
        return self.mode.torque_signals()

    def get_torque_vectors(self):
        return self.mode.torque_vectors()

    def get_control_grid(self):
        return self.mode.control_grid()

    def puller_attached(self):
        return False

    def toggle_puller_attached(self):
        return False

    def get_camera_params(self):
        return self.scenario.camera(self.box)

    def get_box_bounds_3d(self):
        return self.box.bounds_3d()

    def get_box_periodic(self):
        return tuple(bool(p) for p in self.box.periodic)

    def get_scene_fit_points(self):
        return self.scenario.fit_points(self.scenario_params, self.box)

    def get_box_size(self):
        return self.box.lengths[0], self.box.lengths[1]

    def get_bond_pairs(self):
        return None

    def get_hbond_pairs(self):
        return None

    def close(self):
        """Done with this system for good -- the app has switched playgrounds.

        The analysis thread goes too: it is one per RemoteSystem, and the app
        builds a fresh system every time a playground is selected, so leaving them
        running would accumulate a thread per switch, each waking up to look for
        frames that will never come."""
        self.detach()
        self._frame_analysis.stop()

    # ---- what the measuring thread has produced -----------------------------

    @property
    def analysis(self):
        """The Analysis object itself, for a caller that wants more than the
        panels: the observable values and the pair list behind them.

        READ-ONLY, AND IT BELONGS TO ANOTHER THREAD. Reading a finished value out
        of it is safe (a float, an array reference); driving it -- calling
        `update`, replacing it -- is not, which is why the frame path no longer
        does either. Anything that needs the numbers for a PARTICULAR frame should
        say so first: see `wait_for_analysis`."""
        return self._frame_analysis.analysis

    @property
    def analysis_seconds(self):
        """What the last analysis pass cost, for the --debug breakdown.

        It is not charged to the frame any more and should not be: it runs on the
        analysis thread, on its own clock, and the app already knows to charge a
        frame only what the frame actually waited for (see App._update_debug, which
        prints this as the wall figure after the sum)."""
        return self._frame_analysis.seconds

    def wait_for_analysis(self, timeout=5.0):
        """Block until the frames taken in so far have been measured.

        For a caller that steps a fixed number of frames and then reads the
        observables -- the loopback tests, a headless run. The app never calls it:
        the whole point of the analysis thread is that no frame waits for it."""
        return self._frame_analysis.wait_idle(timeout)

    # ---- construction helpers -----------------------------------------------

    def _make_rdf(self):
        """The same choice PlaygroundSystem makes, minus the scenario override --
        which takes a live LAMMPS instance to build, and there isn't one here.

        Called on the analysis thread, which owns the object it returns (see
        FrameAnalysis._build) -- so it is called again on a reconnect, when the far
        side's box may be a different size, rather than rebinned in place.

        `sample_every=1` throughout, unlike the local systems': the throttle
        exists to keep an O(max_atoms^2) pass off every frame, and here the caller
        is the wire, which is already three times slower than the window. Left at the default the average would cover three times as much
        wall time as it does locally, which is a different plot rather than a
        cheaper one.
        """
        lengths = self.box.lengths
        if all(self.box.periodic):
            return RadialRDF3D(min(0.5 * min(lengths), 6.0), lengths[0],
                               sample_every=1)
        if self.box.periodic[0] and self.box.periodic[1]:
            return InPlaneRDF(min(0.5 * min(lengths[0], lengths[1]), 6.0),
                              box=(lengths[0], lengths[1]), sample_every=1)
        return InPlaneRDF(3.0, nbins=48, box=None, sample_every=1)
