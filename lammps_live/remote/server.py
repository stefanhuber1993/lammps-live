"""The headless half of the remote demo: integrate here, draw somewhere else.

    python -m lammps_live.remote.server --playground mesomem_remote \
           --profile cluster-gpu --port 5723 --token-stdin

WHAT IT IS. A `PlaygroundSystem` -- the very same class, built from the very same
playground file the app would build locally -- with a socket in place of a
renderer. Nothing about the physics is re-expressed for the cluster: the deck, the
housekeeping, the thermostat, the live-parameter handling and `Reset` are the app's
own code, and the only differences are collected in a `HostProfile` (hosts.py).
That is the point of doing it this way rather than shipping a hand-written .in
file: there is one definition of what this demo IS, and both ends read it.

WHAT IT DOES NOT DO: any analysis. `Analysis` is switched off here (see
observables.py) and the client measures the frames it receives instead. Two
reasons. The measurement is Python at 1.5 us/bead/chunk (docs/a100-plan.md
section 3), which at 10k beads is most of a 60 fps frame and would throttle the
integrator it is supposed to be observing; and the machine that draws the panels is
the machine that has spare cycles, because all it does otherwise is draw. The wire
carries state, not conclusions.

GAME MODE IS NOT SERVED. A haptic loop over a network is a different project (a
2 ms force-feedback loop cannot survive a 10 ms RTT), so the server refuses
anything but sim mode rather than appearing to support it.

PACING. One stride per sent frame, capped at `--fps`, and the STRIDE IS DERIVED
FROM THE RATE (`_stride_for`) so that the simulation advances at the same rate per
wall-clock second as the local demo whatever the send rate is. That indirection is
the whole reason the wire can drop to 20 fps to save bandwidth without also
running the demo at a third speed: 20 fps takes three times the stride, the pace is
unchanged, and only the sampling of the trajectory gets coarser (which is what the
client fills in -- see playground/jitter.py).

`--free-run` lets the integrator run flat out between sends instead, turning the
demo into a fast-forward of the same physics. It is a legitimate thing to want
(assembly by t ~ 2000 tau arrives in seconds) and it is not the default, for two
reasons: the sliders then act on something that has already moved on, and at 20
fps on an A100 it puts consecutive frames ~14 tau apart, far enough that they are
nearly uncorrelated -- which leaves the client's smoothing and rattle fill nothing
to work with. The GPU idling ~96% at the honest pace is not a reason to reach for
it; the A100 is here to make 50k beads possible, not to maximise steps per second.

RECONNECTING IS FREE. The system is built on the first connection and kept
afterwards, so a dropped tunnel, a closed laptop lid or a restarted client picks up
the same run where it left off -- paused, because a client that is not there cannot
be watching. Only `--exit-when-idle` ends the process, which is what makes an
abandoned GPU allocation release itself.
"""
import argparse
import hmac
import os
import queue
import signal
import socket
import sys
import threading
import time

from ..playground.faults import Fault
from . import hosts, protocol

# How long a new connection has to complete its handshake before it is dropped.
# It exists because the server deliberately serves ONE client at a time, so a
# connection that opens and then says nothing would sit in `recv` forever and the
# server would never accept anybody again -- a denial of service by accident, which
# is all it takes when the port is reachable from a shared cluster network. Twenty
# seconds is far more than a hello needs and far less than a demo can tolerate
# being wedged for.
HANDSHAKE_TIMEOUT = 20.0


class ControlChannel:
    """Reads client messages on their own thread and hands them over as a queue.

    Separate from the send path because the two have nothing to say to each other:
    a frame send may block for as long as the client takes to drain its buffer
    (which is the backpressure that keeps this honest), and a slider change must
    not wait behind it.
    """

    def __init__(self, sock):
        self.sock = sock
        self.messages = queue.Queue()
        self.closed = threading.Event()
        self._thread = threading.Thread(target=self._run, name="control-reader",
                                        daemon=True)
        self._thread.start()

    def _run(self):
        try:
            while True:
                header, _payload = protocol.recv_message(self.sock)
                if header is None:
                    break
                self.messages.put(header)
        except (OSError, protocol.ProtocolError):
            pass
        finally:
            self.closed.set()

    def drain(self):
        out = []
        while True:
            try:
                out.append(self.messages.get_nowait())
            except queue.Empty:
                return out


class FrameServer:
    def __init__(self, playground="mesomem_remote", profile="cluster-gpu",
                 port=protocol.DEFAULT_PORT, bind="0.0.0.0", token="",
                 fps=60.0, steps_per_frame=None, free_run=False,
                 codec=protocol.DEFAULT_CODEC,
                 coeff_values=None, exit_when_idle=0.0, verbose=True):
        self.playground_ref = playground
        self.profile_name = profile
        self.port = int(port)
        self.bind = bind
        self.token = token
        self.fps = float(fps)
        self.steps_override = steps_per_frame
        self.codec = codec
        self.free_run = bool(free_run)
        self.coeff_values = coeff_values
        self.exit_when_idle = float(exit_when_idle)
        self.verbose = verbose
        self.system = None
        self.playing = False
        self.want_energies = False
        self.seq = 0
        self._sent_bytes = 0
        self._sent_frames = 0
        # Set to bring the serve loop to a stop between chunks. It exists because
        # `close()` MUST NOT be called while the loop is inside `lmp.command("run
        # ...")`: closing a LAMMPS instance under a running step segfaults the
        # process (found exactly that way, from a test fixture that tore the server
        # down while it was integrating). So stopping is a two-step move -- stop the
        # loop, then close what it was using.
        self._stop = threading.Event()
        # The last thing that killed or nearly killed the simulation, waiting for
        # the next frame to carry it to the client.
        self._fault = None
        # PARKED RUNS, by playground: the state of each simulation this server has
        # been asked to put aside for another one (see switch_playground). A parked
        # 50k run is under 5 MB of arrays, so the cap is a guard against a client
        # cycling through playgrounds forever rather than a real budget -- and it
        # evicts the oldest, which is the one least likely to be gone back to.
        self._parked = {}

    # ---- lifecycle ----------------------------------------------------------

    def log(self, *bits):
        """One line, flushed. This process' stdout is an SSH pipe the GUI reads,
        so an unflushed line is a line the user never sees."""
        if self.verbose:
            print("[server]", *bits, flush=True)

    MAX_PARKED = 4

    def build(self):
        """Construct the simulation, and put a parked run back into it if this
        playground has one.

        Deferred to the first connection so a misconfigured deck fails where someone
        is watching, and so the allocation is not spent integrating before anyone has
        connected.

        THE RESUME IS WHAT MAKES SWITCHING FREE. Building is now a second or so even
        at 50,000 beads (it used to be three quarters of a minute, almost all of it
        the random fill -- see RandomFill.build), so the cost of a switch was never
        really the CPU: it was that the run you walked away from was gone, and four
        minutes of coarsening with it. A parked state goes straight back into the
        fresh instance and the demo carries on from where it was, with its own
        parameters (see PlaygroundSystem.restore_state).
        """
        from ..playground import registry

        playground = registry.load(self.playground_ref)
        if (playground.mode or "sim") != "sim":
            raise SystemExit(
                f"{self.playground_ref} is a {playground.mode!r} playground. Only "
                f"sim mode is served: an interactive force-feedback loop cannot "
                f"run over a network link (see the module docstring).")
        profile = hosts.get(self.profile_name)
        if self.coeff_values is not None:
            profile = profile.with_coeff_values(self.coeff_values)
        self.log(f"building {self.playground_ref} on profile {profile.name}"
                 + (f", pair_coeff truncated to {self.coeff_values} values"
                    if self.coeff_values else ""))
        # A run waiting to be resumed makes the scenario's own relaxation pointless
        # work -- it would relax a configuration that is about to be overwritten --
        # so the deck is built with every fix in place and no steps taken. Measured
        # on the vesicle+polymer playground: 7.2 s to build, 0.4 s to build and
        # resume. See PlaygroundSystem._settle_commands.
        parked = self._parked.pop(self.playground_ref, None)
        self._construct(playground, profile, settle=parked is None)
        if parked is not None and not self.system.restore_state(parked):
            self.log(f"the parked {self.playground_ref} run no longer fits this "
                     f"build ({parked['natoms']} particles against "
                     f"{self.system.natoms}) -- starting fresh")
            # And it must be a REAL fresh start: the build above skipped the settle
            # on the strength of a resume that did not happen, so the state it is
            # holding is the scenario's made-up starting geometry rather than a
            # relaxed one. Cheaper to build again than to serve that.
            self.system.close()
            self._construct(playground, profile, settle=True)
        elif parked is not None:
            self.log(f"resumed the parked {self.playground_ref} run at "
                     f"t = {parked['sim_time']:.1f} tau")
        return self.system

    def _construct(self, playground, profile, settle):
        """One PlaygroundSystem, with the timing line the log is read for."""
        from ..playground.system import PlaygroundSystem
        t0 = time.perf_counter()
        self.system = PlaygroundSystem(playground, mode_name="sim",
                                       host_profile=profile, analysis=False,
                                       settle=settle)
        self.steps_per_frame = self._stride_for(self.fps)
        self.log(f"built {self.system.natoms} particles in "
                 f"{time.perf_counter() - t0:.1f}s"
                 + ("" if settle else " (settle skipped, resuming)") + ", "
                 f"{self.steps_per_frame} steps/frame, "
                 f"box {self.system.box.lengths[0]:.2f} sigma")

    def _park_current(self):
        """Set the loaded run aside so switching back can resume it."""
        if self.system is None:
            return
        try:
            snapshot = self.system.snapshot_state()
        except Exception as exc:                      # noqa: BLE001 -- best effort
            # Parking is a convenience and the switch is not: a snapshot that cannot
            # be taken must cost the state being parked, not the switch itself.
            self.log(f"could not park {self.playground_ref}: {exc}")
            return
        if snapshot is None:
            return
        while len(self._parked) >= self.MAX_PARKED:
            self._parked.pop(next(iter(self._parked)))
        self._parked[self.playground_ref] = snapshot
        self.log(f"parked {self.playground_ref} at t = "
                 f"{snapshot['sim_time']:.1f} tau ({snapshot['natoms']} particles)")

    # The frame rate a scenario's `sim_time_per_frame` is quoted against. It is
    # the app's own refresh rate, because that is what the local playgrounds run
    # at and what every one of these scenarios was tuned watching.
    REFERENCE_FPS = 60.0

    def _stride_for(self, fps):
        """MD steps per sent frame, such that the SIMULATION'S PACE does not
        depend on how often frames are sent.

        A scenario declares `sim_time_per_frame`: how much simulated time one
        DRAWN frame should advance, chosen by watching it. Sending at 20 fps
        instead of 60 must therefore take three times the stride, or the same
        demo runs three times slower in wall-clock -- and this is easy to get
        wrong in the other direction too. `free_run` (integrate flat out until
        the next send is due) looks like the obvious way to keep the GPU busy,
        and on an A100 at 50k beads it advances 1,420 steps per frame instead of
        60: the pace goes from 12 tau/s to 284, which is not the demo, and
        consecutive frames land ~14 tau apart, far enough that they are nearly
        uncorrelated. Nothing on the client can fill in between two frames that
        share no motion, so it takes the trajectory smoothing and the rattle fill
        down with it.

        The GPU sitting ~96% idle at the honest pace is not a problem worth
        solving. The A100 is here to make 50k beads possible at all, not to
        maximise steps per second; the allocation costs the same either way.
        """
        if self.steps_override:
            return max(1, int(self.steps_override))
        scenario = self.system.scenario
        per_frame = max(1, round(scenario.sim_time_per_frame / scenario.timestep))
        if fps <= 0:
            # No pacing at all (the loopback tests, and `--fps 0`): there is no
            # send rate to hold the pace against, so the scenario's own stride is
            # the only meaningful answer.
            return per_frame
        return max(1, round(per_frame * self.REFERENCE_FPS / float(fps)))

    def stop(self):
        """Ask the serve loop to return. Safe from any thread; `close()` is not."""
        self._stop.set()

    def close(self):
        """Release the simulation. Only valid once the serve loop has returned."""
        if self.system is not None:
            self.system.close()
            self.system = None

    def serve_forever(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.bind, self.port))
        listener.listen(1)
        # The line the connect flow waits for. Keep the shape stable: session.py
        # matches on it to know the tunnel has something to reach.
        self.log(f"LISTENING host={socket.gethostname()} port={self.port}")
        idle_since = time.monotonic()
        # Always on a timeout, so `stop()` is acted on within half a second rather
        # than whenever the next client happens to connect.
        listener.settimeout(0.5)
        try:
            while not self._stop.is_set():
                try:
                    sock, addr = listener.accept()
                except socket.timeout:
                    if (self.exit_when_idle
                            and time.monotonic() - idle_since > self.exit_when_idle):
                        self.log(f"no client for {self.exit_when_idle:.0f}s -- exiting")
                        return
                    continue
                self.log(f"client {addr[0]}:{addr[1]} connected")
                try:
                    self.serve_client(sock)
                except (OSError, protocol.ProtocolError) as exc:
                    self.log(f"client dropped: {type(exc).__name__}: {exc}")
                finally:
                    try:
                        sock.close()
                    except OSError:
                        pass
                    # A client that goes away leaves the run where it was, and
                    # stopped: nobody is watching it, and an A100 integrating for
                    # an empty socket is the most expensive no-op available.
                    self.playing = False
                    idle_since = time.monotonic()
                    self.log("client gone; simulation held")
        finally:
            listener.close()

    # ---- one client ---------------------------------------------------------

    def _authenticate(self, sock):
        # On a timeout for the handshake only; cleared by serve_client once the
        # client is known. It must NOT stay on afterwards: the same socket is read
        # by the control thread, which is idle for minutes at a time between slider
        # movements, and a timeout there would read as a dead client.
        sock.settimeout(HANDSHAKE_TIMEOUT)
        try:
            header, _ = protocol.recv_message(sock)
        except (TimeoutError, socket.timeout):
            self.log(f"a connection said nothing in {HANDSHAKE_TIMEOUT:.0f}s -- "
                     f"dropped")
            return None
        if header is None:
            return None
        if header.get("t") != "hello":
            sock.sendall(protocol.pack({"t": "error", "msg": "expected hello"}))
            return None
        if header.get("version") != protocol.VERSION:
            sock.sendall(protocol.pack(
                {"t": "error", "msg": f"protocol version mismatch: client "
                                      f"{header.get('version')}, server "
                                      f"{protocol.VERSION}. Both ends must run the "
                                      f"same lammps_live."}))
            return None
        # compare_digest, not ==, because a plain comparison on a shared cluster
        # network leaks the token one byte at a time to anything that can time it.
        if not hmac.compare_digest(str(header.get("token", "")), self.token):
            self.log("rejected a client with a bad token")
            sock.sendall(protocol.pack({"t": "error", "msg": "bad token"}))
            return None
        return header

    def switch_playground(self, ref):
        """Point this server at a different playground, throwing away the one it is
        holding. True if anything changed.

        THIS IS WHAT MAKES ONE ALLOCATION SERVE SEVERAL DEMOS. Getting a GPU is the
        expensive, queued, prompt-answering part; building a playground on one that
        is already ours is tens of seconds and no queue at all. So a client that
        connects naming a playground other than the one loaded gets the loaded one
        closed -- which frees its LAMMPS instance, and with it the GPU memory it was
        holding -- and the named one built in its place, on the same node, through
        the same tunnel, with the same job id. Switching back later is the same move
        in reverse (see ui/remote_panel.py, which is what asks).

        AND THE RUN IS PRESERVED, which it did not use to be. The comment here used
        to call losing it "the honest cost of the trade"; it was only honest while
        rebuilding cost a minute. Now the state is put aside before the instance is
        closed and goes straight back in if this playground is asked for again (see
        `_park_current`, and the resume in `build`), so a talk can cycle between
        two demos without either of them starting over. What a switch costs now is a
        second of build and the frames in flight.

        Closing here is safe for the one reason that matters: the server serves one
        client at a time, and this runs during a handshake, so the serve loop has
        returned and nothing is inside `lmp.command("run ...")` (see `close`).

        THE TOKEN IS THE BOUNDARY. `ref` reaches `registry.load`, which will import
        a path as well as a bundled name -- but only a client holding the session
        token gets this far, and such a client can already drive the deck through
        every control message there is. This adds no capability it did not have.
        """
        if not ref or ref == self.playground_ref:
            return False
        self.log(f"client asked for {ref}, holding {self.playground_ref} -- "
                 f"switching")
        self._park_current()
        self.close()
        self.playground_ref = ref
        # The new run starts stopped and from sequence zero, which is how the client
        # recognises it as a different run rather than a jump in the old one (see
        # RemoteSystem._is_new_run).
        self.playing = False
        self.seq = 0
        self._fault = None
        return True

    def serve_client(self, sock):
        protocol.set_socket_options(sock)
        header = self._authenticate(sock)
        if header is None:
            return
        sock.settimeout(None)
        # Asked for before anything is said about building, so the message below
        # names the playground that is actually about to be built.
        self.switch_playground(header.get("playground"))
        if self.system is None:
            # SAY SO FIRST. A build is `plugin load`, the placement and LAMMPS' own
            # setup, plus whatever relaxation the scenario asks for -- and all of it
            # lands before the welcome can be sent, with the client sitting in a
            # blocking read with a handshake timeout on it. It used to be tens of
            # seconds and is now around one on the assembly decks (see
            # RandomFill.build) and several on a bonded one that settles, which is
            # still long enough to matter: the client used to give up at 15 seconds,
            # drop the socket, and retry -- which made the server throw the half-built
            # simulation away and start again, so the retry could not succeed either.
            # This message is what turns that wait into a wait: the client stops
            # counting against the handshake timeout and says what is happening (see
            # FrameLink.connect).
            sock.sendall(protocol.pack(
                {"t": "building",
                 "msg": f"building {self.playground_ref} -- this takes a moment"}))
            try:
                self.build()
            except Exception as exc:
                sock.sendall(protocol.pack(
                    {"t": "error", "msg": f"could not build the simulation: "
                                          f"{type(exc).__name__}: {exc}"}))
                raise
        system = self.system
        sock.sendall(protocol.pack(self._welcome()))

        control = ControlChannel(sock)
        next_send = time.monotonic()
        while not control.closed.is_set() and not self._stop.is_set():
            stop = self._apply_control(control.drain(), sock)
            if stop:
                return
            if self.playing:
                if self.free_run:
                    # Keep integrating until the next send is due. One chunk is
                    # always taken, so a slow chunk cannot starve the frame.
                    while True:
                        system.step(self.steps_per_frame)
                        if time.monotonic() >= next_send:
                            break
                else:
                    system.step(self.steps_per_frame)
            self._flush_fault(sock)
            self._send_frame(sock, system)
            if self.fps > 0:
                next_send = max(next_send + 1.0 / self.fps, time.monotonic() - 0.25)
                delay = next_send - time.monotonic()
                if delay > 0:
                    # Interruptible by the client going away, so a paused server
                    # does not sit in a sleep for a quarter of a second past the
                    # disconnect.
                    control.closed.wait(delay)
            else:
                next_send = time.monotonic()

    def _welcome(self):
        system = self.system
        box = system.box
        profile = hosts.get(self.profile_name)
        return {
            "t": "welcome",
            "version": protocol.VERSION,
            "playground": self.playground_ref,
            "natoms": int(system.natoms),
            "box": {"lo": list(box.lo), "hi": list(box.hi),
                    "periodic": [bool(p) for p in box.periodic]},
            "timestep": system.scenario.timestep,
            "steps_per_frame": self.steps_per_frame,
            "codec": self.codec,
            "free_run": self.free_run,
            "fps": self.fps,
            "coeff_values": self.coeff_values,
            "host": socket.gethostname(),
            "profile": profile.name,
            "lammps_args": list(profile.lammps_args),
            "atom_style": system.force_field.atom_style,
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "sim_time": system.get_sim_time(),
        }

    def _apply_control(self, messages, sock=None):
        """Apply what the client asked for. Returns True if it asked to end.

        Every one of these is the same call the local app makes on its own system
        in `App._tick` -- the control channel is a remote procedure call onto the
        MDSystem interface, not a second way of doing things.
        """
        system = self.system
        for msg in messages:
            kind = msg.get("t")
            try:
                if self._apply_one(msg, kind, system, sock):
                    return True
            except Exception as exc:                   # noqa: BLE001 -- reported
                # A SLIDER MUST NOT COST THE ALLOCATION. Anything that reaches
                # LAMMPS can be refused by it, and letting that propagate ends this
                # process -- which runs `scancel` on its own job on the way out, so
                # one bad value would take the A100 with it and the next attempt
                # costs a fresh queue wait and a one-time code. `reset` recovers on
                # its own (PlaygroundSystem._rebuild); everything else is reported
                # and skipped.
                self._fault = Fault.from_error(
                    exc, fatal=(self.system is None or self.system.lmp is None))
                self.log(f"{kind} failed: {self._fault.line()}")
                if self._fault.fatal:
                    # There is no simulation left to send frames from. Say so and
                    # drop the client; the next one to connect gets a fresh build.
                    if sock is not None:
                        sock.sendall(protocol.pack(
                            {"t": "error",
                             "msg": f"the simulation could not be rebuilt: "
                                    f"{self._fault.detail}"}))
                    self.system = None
                    return True
        return False

    def _apply_one(self, msg, kind, system, sock):
        """One control message. True if the client asked to end the session."""
        if kind == "set":
            system.set_extra_param(msg["key"], float(msg["value"]))
        elif kind == "temp":
            system.set_target_temp(float(msg["value"]))
        elif kind == "play":
            self.playing = True
        elif kind == "pause":
            self.playing = False
        elif kind == "reset":
            # TIMED, AND SAID AFTERWARDS AS WELL AS BEFORE. This call is the whole
            # of a LAMMPS rebuild and it blocks this thread -- no frames go out and
            # no control message is answered until it returns -- so the log line
            # that opens it used to be the last thing anybody heard for as long as
            # it took. The pair of lines is what tells a rebuild that is merely slow
            # apart from one that never came back, in the log the panel copies.
            self.log("reset: rebuilding from a fresh random state")
            t0 = time.monotonic()
            system.reset()
            self.log(f"reset: rebuilt in {time.monotonic() - t0:.1f}s")
            self.playing = False
            self.seq = 0
        elif kind == "config":
            if "fps" in msg:
                self.fps = float(msg["fps"])
                # The stride follows, so that changing how OFTEN frames are sent
                # does not also change how fast the simulation runs. Skipped when
                # the stride was pinned explicitly (--steps-per-frame), which is
                # the one case where somebody has said what they want.
                self.steps_per_frame = self._stride_for(self.fps)
            if "codec" in msg and msg["codec"] in protocol.CODECS:
                self.codec = msg["codec"]
            if "energies" in msg:
                self.want_energies = bool(msg["energies"])
            if "steps_per_frame" in msg:
                # An explicit stride pins it: it survives later fps changes, the
                # same way --steps-per-frame does.
                self.steps_override = max(1, int(msg["steps_per_frame"]))
                self.steps_per_frame = self.steps_override
            if "free_run" in msg:
                self.free_run = bool(msg["free_run"])
        elif kind == "ping":
            # Answered on this thread, which is also the one that sends
            # frames, so the round trip the client measures includes any time
            # spent waiting behind a frame -- which is the number that matters.
            if sock is not None:
                sock.sendall(protocol.pack({"t": "pong", "id": msg.get("id")}))
        elif kind == "bye":
            self.log("client said goodbye")
            return True
        return False

    def _flush_fault(self, sock):
        """Send the next fault, if there is one, as a message of its own.

        NOT A FIELD ON THE FRAME HEADER, which is where this started and where it
        was wrong: the client keeps only the LATEST frame and drops the rest (see
        client.py), so on a link fast enough to drop frames -- which is the point of
        the A100 -- the one frame carrying the event would usually be the one thrown
        away. An event that is only sometimes delivered is worse than none.

        Two sources, one channel: this loop catches what a control message raised,
        and the simulation itself latches what a chunk or a rebuild did (a blow-up,
        or the parameters a rebuild had to put back). The client cannot tell them
        apart and should not have to.
        """
        fault = self._fault
        self._fault = None
        if fault is None and self.system is not None:
            fault = self.system.take_fault()
        if fault is None:
            return
        self.log("fault: " + fault.line())
        sock.sendall(protocol.pack({"t": "fault", **fault.as_message()}))

    def _send_frame(self, sock, system):
        state = system.current_state()
        energies = system.get_bead_energies() if self.want_energies else None
        energy_range = system.spec.render_style.energy_range
        manifest, payload = protocol.encode_frame(
            state, system.box, energies=energies, energy_range=energy_range,
            codec=self.codec)
        self.seq += 1
        header = {
            "t": "frame",
            "seq": self.seq,
            "n": int(len(state.positions)),
            "codec": self.codec,
            "arrays": manifest,
            "sim_time": system.get_sim_time(),
            # The sim time this frame advanced, which is what the client's
            # smoothing filter needs to set its own weight (see smoothing.py).
            "dt": system._last_step_dt if self.playing else 0.0,
            "thermo": [float(v) for v in system.get_thermo_state()],
            "playing": self.playing,
            "unstable": system.unstable,
            "wall": time.time(),
        }
        sock.sendall(protocol.pack(header, payload))
        self._sent_frames += 1
        self._sent_bytes += len(payload)
        if self.verbose and self._sent_frames % 600 == 0:
            self.log(f"{self._sent_frames} frames, "
                     f"{self._sent_bytes / 1e6:.1f} MB sent, "
                     f"sim time {system.get_sim_time():.1f} tau")


def _read_token(args):
    """The shared secret, which never appears in a command line.

    On a shared cluster every user can read every other user's `ps` output, so a
    `--token` argument would publish it to the whole login node; the same goes for
    the environment on a system where /proc is not hidden. The session sends it
    down the SSH channel's stdin instead, which nothing else can see. `--token` is
    kept for driving the server by hand.
    """
    if args.token_stdin:
        line = sys.stdin.readline()
        if not line.strip():
            raise SystemExit("--token-stdin: no token arrived on stdin")
        return line.strip()
    if args.token:
        return args.token
    if os.environ.get("LAMMPS_LIVE_TOKEN"):
        return os.environ["LAMMPS_LIVE_TOKEN"]
    raise SystemExit("no token: pass --token-stdin, --token, or set "
                     "LAMMPS_LIVE_TOKEN. The server refuses to run open, because "
                     "its control channel can re-issue LAMMPS commands.")


def build_parser():
    p = argparse.ArgumentParser(prog="python -m lammps_live.remote.server",
                                description=__doc__.splitlines()[0])
    p.add_argument("--playground", default="mesomem_remote",
                   help="playground key or path to serve (default: mesomem_remote)")
    p.add_argument("--profile", default="cluster-gpu",
                   choices=sorted(hosts.PROFILES),
                   help="which LAMMPS build this is (default: cluster-gpu)")
    p.add_argument("--port", type=int, default=protocol.DEFAULT_PORT)
    p.add_argument("--bind", default="0.0.0.0",
                   help="interface to listen on. The default is reachable from the "
                        "login node, which is where the tunnel lands; the token is "
                        "what keeps that safe")
    p.add_argument("--token", default="", help="shared secret (prefer --token-stdin)")
    p.add_argument("--token-stdin", action="store_true",
                   help="read the shared secret from the first line of stdin")
    p.add_argument("--fps", type=float, default=60.0,
                   help="cap on frames sent per second; 0 for uncapped")
    p.add_argument("--steps-per-frame", type=int, default=None,
                   help="MD steps per frame (default: the scenario's own cadence)")
    p.add_argument("--codec", default=protocol.DEFAULT_CODEC, choices=protocol.CODECS)
    p.add_argument("--free-run", action="store_true",
                   help="integrate flat out between sends instead of one chunk per "
                        "frame -- the same physics, fast-forwarded")
    p.add_argument("--coeff-values", type=int, default=None,
                   help="truncate pair_coeff to this many values (8 for the "
                        "authors' original style, 9 for the patched one). The "
                        "connect flow fills this in from the probe")
    p.add_argument("--exit-when-idle", type=float, default=0.0, metavar="SECONDS",
                   help="exit after this long with no client connected, so an "
                        "abandoned allocation releases itself (0 = never)")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    server = FrameServer(
        playground=args.playground, profile=args.profile, port=args.port,
        bind=args.bind, token=_read_token(args), fps=args.fps,
        steps_per_frame=args.steps_per_frame, codec=args.codec,
        free_run=args.free_run, coeff_values=args.coeff_values,
        exit_when_idle=args.exit_when_idle, verbose=not args.quiet)
    # A clean end on scancel / kill: Slurm sends SIGTERM before SIGKILL, and the
    # handler runs on this thread between chunks, so LAMMPS is closed while nothing
    # is inside it. Without this the process is killed mid-`run`, which is harmless
    # for the simulation but loses the last log lines the GUI is showing.
    def _terminate(_signum, _frame):
        server.log("signal received -- stopping")
        server.stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _terminate)
        except (ValueError, OSError):
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.log("interrupted")
    finally:
        server.close()
        server.log("stopped")
    return 0


if __name__ == "__main__":
    status = main()
    # os._exit, NOT sys.exit, and only on this path -- a `main()` called in-process
    # (the loopback tests) must still return normally.
    #
    # WHY: a Kokkos/CUDA build leaves a static `CudaInternal` singleton whose
    # destructor runs during interpreter shutdown, by which time the CUDA runtime is
    # already unloading. `cudaStreamSynchronize` then fails, the destructor throws,
    # and std::terminate turns whatever we were reporting into SIGABRT and a core
    # dump -- which is how a one-line "requires zeta >= 1" arrived from the cluster
    # under sixty lines of backtrace. Skipping the C++ static destructors costs
    # nothing here: every LAMMPS instance has already been closed by `server.close()`
    # and both streams are flushed on the next two lines.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status or 0)
