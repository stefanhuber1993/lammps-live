"""Getting a GPU, putting the server on it, and giving it back afterwards.

Six steps, run on a worker thread so the app keeps drawing at 60 fps throughout,
each one reported to whatever is watching (the connect panel, or a terminal):

    1. LOGIN      one SSH master connection, authenticated once
    2. DEPLOY     ship this package to the cluster
    3. PROBE      ask that machine whether it can run the server at all
    4. ALLOCATE   salloc --no-shell, then wait for the node
    5. LAUNCH     srun the server inside the allocation
    6. TUNNEL     a second SSH, ending ON the node, carrying a local port to it

WHY ONE MASTER CONNECTION. Every step above needs the cluster, and each `ssh`
would authenticate again -- which with a one-time code means typing a fresh one
five times. So the first connection is a ControlMaster and every later command
rides on it over the same socket, instantly and without a prompt.

WHY THE TUNNEL IS TWO HOPS. The obvious one-hop form,
`ssh -L <local>:<node>:<port> <login>`, ends its session on the LOGIN node: sshd
there decrypts the stream and opens a separate, plain TCP connection onward to the
compute node. So the frames cross the cluster's internal network in clear text, and
the server has to listen on the node's external interface where every other user on
that network can reach it.

Two hops instead: an SSH session that terminates ON THE COMPUTE NODE, carried
through the login node as an opaque stream it cannot read. The forward's far end is
then `localhost` as seen from the node, so the server binds 127.0.0.1 and the port
simply does not exist for the rest of the cluster. The jump hop rides the master
connection (`ProxyCommand ssh -S <ctl> -W %h:%p`), so it costs no extra
authentication -- verified on Snellius: `gcn12` does not ask for a second one-time
code. `RemoteTarget.tunnel = "forward"` keeps the one-hop form for a site where
ssh to a compute node is not permitted.

AND EVERY HOP SAYS WHAT IT NEEDS, rather than inheriting it. Both ssh commands here
disable connection multiplexing explicitly (`ControlPath=none` on the tunnel,
`ControlPersist=no` on the master), because a `~/.ssh/config` that turns it on for
the cluster -- a sensible thing for a human to have -- changes what `ssh -N -L`
does: it backgrounds the master and exits 0, and the forward goes with it. That cost
a debugging session; the long version is at the tunnel command below.

WHY `salloc --no-shell`. The obvious `ssh host salloc ... bash` holds the
allocation inside an interactive shell on a pipe, and then everything -- the node
name, the job's lifetime, the teardown -- hangs off that pipe staying alive. With
`--no-shell` the allocation is created, the job id is printed, and the command
returns: the allocation exists on its own, `squeue` says where it is, `srun
--jobid=` puts work on it and `scancel` ends it. Nothing depends on a pipe.

WHAT `--no-shell` DOES NOT DO is return before the allocation is GRANTED. On a
full GPU partition that is however long the queue is, so salloc is started as a
process this watches rather than a command this waits on: Slurm prints "Pending
job allocation N" at submission, and that id is enough to poll `squeue` for the
state and the reason, show them, and let Cancel end the wait (step 4, below).

HOW THE PASSWORD AND THE ONE-TIME CODE GET IN. ssh will not read a secret from a
pipe -- deliberately, and no amount of arranging changes that. What it will do is
run an SSH_ASKPASS helper and use what it prints. So this writes a small helper
into a private temporary directory, points ssh at it with
SSH_ASKPASS_REQUIRE=force, and the helper asks THIS process over a unix socket.
Every prompt ssh raises therefore arrives here as text -- "Password:",
"Verification code:", a host-key fingerprint question, a key passphrase -- is shown
verbatim, and the answer is passed back. Nothing is stored, nothing is logged, and
nothing appears in a command line or an environment variable. It also means this
code does not have to know what your cluster asks for or in what order.

GIVING THE GPU BACK. Three independent backstops, because an allocated A100 that
nobody is using is the one failure with a bill attached:

    the app       cancels the job when the window closes (`shutdown`)
    the server    cancels its OWN allocation when it exits, including after
                  --exit-when-idle, so a hard-killed app costs minutes not hours
    Slurm         --time ends it regardless

Run this file directly to test the whole flow without the GUI, which is the way to
debug the SSH leg:

    python -m lammps_live.remote.session --playground mesomem_remote
"""
import json
import os
import re
import secrets
import shlex
import shutil
import socket
import stat
import subprocess
import tempfile
import textwrap
import threading
import time
from collections import deque

from .client import FrameLink, LinkClosed

# The states, in order. `FAILED` and `DOWN` are the two resting places.
DOWN = "down"
LOGIN = "login"
DEPLOY = "deploy"
PROBE = "probe"
ALLOCATE = "allocate"
LAUNCH = "launch"
TUNNEL = "tunnel"
READY = "ready"
FAILED = "failed"
CLOSING = "closing"
# Moving an allocation we already hold onto a different playground: not one of the
# six numbered steps, because it skips every one of them -- the login, the queue,
# the launch and the tunnel are all still standing, and only the far side's
# simulation is rebuilt. See `switch_playground`.
SWITCH = "switch"

_STEP_ORDER = (LOGIN, DEPLOY, PROBE, ALLOCATE, LAUNCH, TUNNEL, READY)

# What the helper script asks this process over its unix socket. Written to a file
# with a shebang because that is what SSH_ASKPASS needs: an executable.
_ASKPASS_SOURCE = '''#!{python}
"""Bridge between ssh's password prompt and the app showing it. Written at run
time by lammps_live/remote/session.py; safe to delete."""
import os, socket, sys

prompt = sys.argv[1] if len(sys.argv) > 1 else "Password:"
path = os.environ["LAMMPS_LIVE_ASKPASS_SOCK"]
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(600)
try:
    s.connect(path)
    s.sendall(prompt.encode("utf-8", "replace") + b"\\n")
    answer = b""
    while not answer.endswith(b"\\n"):
        chunk = s.recv(4096)
        if not chunk:
            break
        answer += chunk
except Exception as exc:
    sys.stderr.write("askpass bridge failed: %s\\n" % exc)
    sys.exit(1)
if not answer.strip():
    sys.exit(1)          # tells ssh the prompt was cancelled
sys.stdout.write(answer.decode("utf-8", "replace").rstrip("\\n"))
'''


class SessionError(Exception):
    """A step failed, with a message worth showing the user."""


def _indented(text, width):
    """`text` wrapped to fit the report's second column, continuation lines aligned.

    The one-line-per-fact shape is what makes the report skimmable, and the error is
    the one fact that can be three hundred characters long.
    """
    lines = textwrap.wrap(str(text), 88 - width) or [""]
    return ("\n" + " " * width).join(lines)


def _same_host(a, b):
    """Whether two names denote the same machine.

    `squeue` prints bare node names, `socket.gethostname()` on the node may print
    an FQDN, and neither is wrong -- so compare what is in front of the first dot,
    case-insensitively. A missing name compares equal to anything, because "we do
    not know yet" must not read as "a different machine".
    """
    if not a or not b:
        return True
    return a.split(".")[0].lower() == b.split(".")[0].lower()


class PromptBridge:
    """The unix-socket end of the askpass helper.

    One connection per prompt: read the question, publish it, wait for the answer,
    write it back. The socket lives in a 0700 directory that only this user can
    traverse, and the answer exists only in memory.
    """

    def __init__(self, on_prompt):
        self.on_prompt = on_prompt
        self.dir = tempfile.mkdtemp(prefix="lammps-live-ask-")
        os.chmod(self.dir, stat.S_IRWXU)
        self.socket_path = os.path.join(self.dir, "ask.sock")
        self.script_path = os.path.join(self.dir, "askpass.py")
        import sys
        with open(self.script_path, "w") as fh:
            fh.write(_ASKPASS_SOURCE.format(python=sys.executable))
        os.chmod(self.script_path, stat.S_IRWXU)
        self._answer = None
        self._answered = threading.Event()
        self._cancelled = False
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(self.socket_path)
        self._listener.listen(4)
        self._listener.settimeout(0.5)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="askpass-bridge",
                                        daemon=True)
        self._thread.start()

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _addr = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with conn:
                try:
                    conn.settimeout(600)
                    prompt = conn.makefile("rb").readline().decode(
                        "utf-8", "replace").strip()
                    answer = self._ask(prompt)
                    if answer is None:
                        continue          # closing without an answer cancels it
                    conn.sendall(answer.encode("utf-8") + b"\n")
                except OSError:
                    continue

    def _ask(self, prompt):
        self._answer = None
        self._answered.clear()
        self.on_prompt(prompt)
        while not self._answered.wait(0.25):
            if self._stop.is_set() or self._cancelled:
                return None
        return self._answer

    def answer(self, text):
        self._answer = text
        self._answered.set()

    def cancel(self):
        self._cancelled = True
        self._answered.set()

    def env(self):
        """The environment ssh needs to use the helper.

        SSH_ASKPASS_REQUIRE=force is the important one: without it ssh prefers a
        terminal, and the app has none it can prompt on. DISPLAY is set because
        older OpenSSH refuses to run an askpass helper without one, and is
        harmless on a version that no longer cares.
        """
        env = dict(os.environ)
        env["SSH_ASKPASS"] = self.script_path
        env["SSH_ASKPASS_REQUIRE"] = "force"
        env["LAMMPS_LIVE_ASKPASS_SOCK"] = self.socket_path
        env.setdefault("DISPLAY", ":0")
        return env

    def close(self):
        self._stop.set()
        self.cancel()
        try:
            self._listener.close()
        except OSError:
            pass
        shutil.rmtree(self.dir, ignore_errors=True)


class RemoteSession:
    """One remote demo session: allocate, launch, connect, tear down.

    Drive it from anywhere: `start()` runs the whole flow on a worker thread,
    `answer()` feeds it a login prompt's answer, `link` is the connected
    `FrameLink` once it reaches READY, and `shutdown()` gives the GPU back.
    """

    LOG_LINES = 200

    def __init__(self, target, playground_ref="mesomem_remote", on_log=None,
                 log_lines=None):
        self.target = target.resolved()
        # Which playground the server is told to build. Both ends must name the
        # same one or they would disagree about how many beads there are.
        self.playground_ref = playground_ref
        # And what was ASKED for, before `_deploy_playground_file` rewrote a local
        # path to where it landed on the cluster. Kept because it is the name the
        # caller knows a playground by -- the app hands back a bundled key or the
        # path the user typed, never the far side's copy -- and `serves` has to
        # answer "is that the one you are running?" in the caller's terms.
        self.playground_asked = playground_ref
        self.state = DOWN
        self.reached = DOWN                 # the furthest step this session got to
        self.detail = "not connected"
        self.error = None
        self.prompt = None                  # the question ssh is waiting on
        self.job_id = None
        self.node = None
        self.local_port = None
        self.link = None
        self.probe_report = None
        self.coeff_values = None
        self.log = deque(maxlen=log_lines or self.LOG_LINES)
        self._on_log = on_log
        self._token = secrets.token_hex(24)
        self._bridge = None
        self._master = None                 # the ControlMaster ssh process
        self._server_proc = None            # the srun holding the server
        self._salloc_proc = None            # the salloc waiting in the queue
        self._control_path = None
        self._tmpdir = None
        self._thread = None
        self._cancel = threading.Event()
        self._forwarded = None               # one-hop: the -L spec to cancel
        self._tunnel_proc = None             # two-hop: the ssh holding the tunnel
        self._remote_port = None             # what the server said it is listening on
        self._server_host = None             # and which host it said it from
        self._multi_node = False             # does the allocation span more than one?
        # Teardown can be reached from two threads at once -- the worker giving up
        # on a failed step, and the app closing the window -- so it takes the lock
        # and empties each field before acting on it. Without that, the second
        # caller finds a half-cleared session and trips over a None.
        self._teardown_lock = threading.Lock()

    # ---- reporting ----------------------------------------------------------

    def _say(self, message, state=None):
        if state:
            self.state = state
            # The furthest step reached, kept separately: `state` becomes FAILED and
            # the one thing the report most needs -- which step it died on -- would
            # be gone with it.
            if state in _STEP_ORDER:
                self.reached = state
        self.detail = message
        stamped = f"{time.strftime('%H:%M:%S')} {message}"
        self.log.append(stamped)
        if self._on_log:
            self._on_log(stamped)

    def _log_line(self, line):
        """A line of output from the far side."""
        line = line.rstrip()
        if not line:
            return
        self.log.append(line)
        if self._on_log:
            self._on_log(line)

    @property
    def busy(self):
        return self.state not in (DOWN, READY, FAILED)

    @property
    def connected(self):
        return self.state == READY and self.link is not None

    @property
    def holds_allocation(self):
        """Is there a Slurm job of ours still out there?

        The question the connect panel asks before deciding whether Connect means
        "get a GPU" or "move the one we have" -- and the reason it is a job id and
        not the state is that a link can die, or a switch can fail, with the
        allocation perfectly intact behind it. Cleared by `_teardown`, which is the
        one place the job is given back.
        """
        return self.job_id is not None and self.state not in (FAILED, CLOSING)

    def serves(self, playground_ref):
        """Whether this session's server is (or is being) built for that playground.

        Compared against `playground_asked`, not `playground_ref`: the latter has
        been rewritten to the far side's copy for a path, and the caller only ever
        knows the name it gave.
        """
        return str(self.playground_asked) == str(playground_ref)

    def progress(self):
        """(step index, total) for a progress readout."""
        if self.state in _STEP_ORDER:
            return _STEP_ORDER.index(self.state) + 1, len(_STEP_ORDER)
        if self.state == SWITCH:
            # Everything but the last step is already done and staying done, so the
            # bar sits where it genuinely is rather than resetting to empty.
            return len(_STEP_ORDER) - 1, len(_STEP_ORDER)
        return (len(_STEP_ORDER), len(_STEP_ORDER)) if self.state == READY else (0, len(_STEP_ORDER))

    def diagnostics(self):
        """Everything known about this session, as text somebody else can read.

        This is what the panel's Copy button puts on the clipboard, and it is
        deliberately more than the panel can show: the whole log rather than the
        last fourteen lines, untruncated rather than clipped to the card's width,
        and the settings that produced it -- because the first question anyone
        asked about a failure was "which node, which port, which tunnel mode?" and
        the answer was off the bottom of the card.

        THE TOKEN IS NOT IN HERE. It is the one secret the session holds, this text
        is meant to be pasted into a chat window, and a live token plus a node name
        is enough for anyone on the cluster to take over the stream.
        """
        target = self.target
        lines = [
            f"lammps-live remote session report  {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"state       {self.state}",
            # The detail line is the FAILED message repeated verbatim once a step
            # has thrown, so it is only worth a line while it says something else.
            *([] if str(self.detail).startswith("FAILED:")
              else ["doing       " + _indented(self.detail, 12)]),
            f"step        {self.reached} "
            f"({_STEP_ORDER.index(self.reached) + 1 if self.reached in _STEP_ORDER else 0}"
            f"/{len(_STEP_ORDER)})",
            "error       " + _indented(self.error or "-", 12),
            "",
            "playground  " + (
                str(self.playground_asked) if self.playground_ref == self.playground_asked
                else f"{self.playground_asked} (shipped to {self.playground_ref})"),
            f"login       {target.destination}",
            f"allocation  {target.partition}, "
            + (f"{target.gres}, " if target.gres
               else f"{target.gpus} gpu, " if target.gpus else "cpus only, ")
            + f"{target.cpus_per_task} cores, {target.time}"
            + (f", account {target.account}" if target.account else ""),
            f"job         {self.job_id or '-'} on node {self.node or '-'}",
            f"tunnel      {target.tunnel}: 127.0.0.1:{self.local_port or target.local_port}"
            f" -> {self.node or '<node>'}:{self._remote_port or target.port}"
            f" (server binds {target.server_bind})",
            f"far side    {target.remote_dir}, env {target.env_script or '<none>'}, "
            f"python {target.python}, profile {target.profile}",
            f"deploy      {target.deploy_dir}",
            f"stream      {target.codec} at {target.fps} fps"
            + (", free-run" if target.free_run else "")
            + f", idle timeout {target.exit_when_idle:.0f}s",
            "",
            "processes   " + ", ".join(
                f"{name}={self._proc_state(proc)}" for name, proc in (
                    ("master", self._master), ("server", self._server_proc),
                    ("tunnel", self._tunnel_proc))),
            f"local port  {self._describe_local_port()}",
        ]
        if self.link is not None:
            frames, mbs = self.link.rates()
            lines += ["", f"link        {self.link.received} frames in, "
                          f"{self.link.dropped} dropped, {frames:.0f} fps, "
                          f"{mbs:.2f} MB/s, error {self.link.error or '-'}"]
        if self.probe_report:
            lines += ["", "probe report"]
            lines += ["  " + line for line in
                      (self.probe_report.get("summary") or ["<empty>"])]
        lines += ["", f"log ({len(self.log)} lines)", ""]
        lines += list(self.log)
        return "\n".join(lines)

    def save_report(self, path=None):
        """Write `diagnostics()` to a file and return its path, or None.

        Next to the clipboard rather than instead of it: a file can be attached to
        a mail, it survives the next thing that gets copied, and on a machine with
        no clipboard command at all it is the only way the report gets out.
        """
        path = path or os.path.join(
            tempfile.gettempdir(),
            f"lammps-live-remote-{time.strftime('%Y%m%d-%H%M%S')}.txt")
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self.diagnostics().rstrip() + "\n")
        except OSError:
            return None
        return path

    @staticmethod
    def _proc_state(proc):
        """How a child process is doing, in one word, for the report."""
        if proc is None:
            return "none"
        code = proc.poll()
        return "running" if code is None else f"exited({code})"

    def _describe_local_port(self):
        """Whether this end of the tunnel is still listening, right now.

        The one fact that splits the two failures that look identical from the
        client's side: "the tunnel died" and "the tunnel is fine, the far end is
        not answering".
        """
        port = self.local_port or self.target.local_port
        return (f"127.0.0.1:{port} accepts connections" if self._port_in_use(port)
                else f"127.0.0.1:{port} is NOT listening")

    # ---- the flow -----------------------------------------------------------

    def start(self):
        """Begin connecting. Returns immediately."""
        if self.busy:
            return
        self._cancel.clear()
        self.error = None
        self.prompt = None
        # Set here, not in the worker: a caller that checks `busy` on the next line
        # (the GUI's own frame, a test's wait loop) must see that something is
        # happening rather than race the thread's first step.
        self.state = LOGIN
        self.detail = "starting"
        self._thread = threading.Thread(target=self._run, name="remote-session",
                                        daemon=True)
        self._thread.start()

    def _run(self):
        try:
            self._tmpdir = tempfile.mkdtemp(prefix="lammps-live-ssh-")
            self._control_path = os.path.join(self._tmpdir, "ctl")
            self._open_master()
            self._deploy()
            self._probe()
            self._allocate()
            self._probe_node()
            self._launch()
            self._tunnel()
            self._connect()
            self._say(f"streaming from {self.node} (job {self.job_id})", READY)
        except SessionError as exc:
            self.error = str(exc)
            self._say(f"FAILED: {exc}", FAILED)
            self._teardown()
        except Exception as exc:                      # noqa: BLE001 -- reported
            self.error = f"{type(exc).__name__}: {exc}"
            self._say(f"FAILED: {self.error}", FAILED)
            self._teardown()
        finally:
            self.prompt = None

    def answer(self, text):
        """Answer whatever ssh is currently asking."""
        if self._bridge is not None:
            self.prompt = None
            self._bridge.answer(text)

    def reopen_link(self):
        """Open a fresh link over the tunnel this session already holds.

        This is what makes switching to another playground and back cheap. The
        server keeps its simulation between clients -- it holds the run where it
        was and waits for the next one to connect (see server.serve_forever) -- so
        coming back is one socket through a tunnel that never closed, not another
        allocation and another queue wait. The GPU is still ours the whole time;
        the server's own `--exit-when-idle` is what eventually gives it back if
        nobody returns.

        Returns the new link, or None if there is nothing to come back to (the job
        ended, the tunnel died, this session never got there) -- in which case the
        session is marked DOWN with the reason, and the caller should start a new
        one.
        """
        if self.state != READY or self.local_port is None:
            return None
        try:
            # This one is not a connection failure but a bereavement: a process of
            # ours has exited, and the server's own `scancel` means the allocation
            # went with it. Handled separately below, because the difference decides
            # whether `holds_allocation` may still be believed.
            self._check_still_alive()
            self.link = FrameLink.connect("127.0.0.1", self.local_port,
                                          self._token, timeout=15.0,
                                          on_notice=self._say,
                                          playground=self.playground_ref)
        except SessionError as exc:
            self.link = None
            self.error = str(exc)
            self._say(f"FAILED: {exc}", FAILED)
            self._teardown()
            return None
        except (LinkClosed, OSError) as exc:
            # Both ends still look alive and the socket did not work out: the GPU is
            # very probably still ours, so it is NOT given back on the strength of
            # one refused connection. The panel offers another go, and Disconnect.
            self.link = None
            self.note_link_lost(str(exc))
            return None
        self._say(f"reconnected to {self.node} (job {self.job_id})", READY)
        return self.link

    def switch_playground(self, ref):
        """Put a DIFFERENT playground on the GPU this session already holds.

        The whole point of the exercise, and the thing that makes two remote demos
        practical to stand in front of: getting the allocation is the part that
        queues, prompts for a one-time code and takes minutes, and it is not
        repeated. The login, the deployed package, the job, the server process and
        the tunnel all stay exactly as they are; the far side closes the simulation
        it was holding and builds this one instead, on the same node (see
        server.FrameServer.switch_playground). What that costs is about a second of
        rebuild: the run being left is parked over there and comes back where it was
        when this playground is asked for again.

        Returns True if a switch was started. It runs on a worker thread, because
        the far side's rebuild is LAMMPS' own setup on tens of thousands of beads --
        tens of seconds during which this end must keep drawing at 60 fps, and
        during which `state` is SWITCH and the panel says so.

        A FAILED SWITCH DOES NOT GIVE THE GPU BACK. That is the one place this
        departs from every other step in this file, and deliberately: the allocation
        is still ours and still good, so the useful thing to offer is another go at
        the switch, not another hour in the queue. The state goes to DOWN with the
        reason, `holds_allocation` stays true, and the panel's Connect comes back
        here rather than to `start` (see connect_playground).
        """
        if self.serves(ref):
            return False                # already the one loaded; resume instead
        return self._start_relink(ref)

    def _start_relink(self, ref):
        """Run `_run_switch` on a worker. Also the reconnect path: with `ref` the
        playground already loaded it asks the server for nothing and just opens a
        fresh socket, which is what a link that died on its own needs."""
        if self.busy or not ref or self.local_port is None:
            return False
        if not self.holds_allocation:
            return False
        self._cancel.clear()
        self.error = None
        self.prompt = None
        self.state = SWITCH
        self.detail = (f"reconnecting to {ref}" if self.serves(ref)
                       else f"switching the GPU to {ref}")
        self._thread = threading.Thread(target=self._run_switch, args=(ref,),
                                        name="remote-switch", daemon=True)
        self._thread.start()
        return True

    def _run_switch(self, ref):
        previous, previous_asked = self.playground_ref, self.playground_asked
        try:
            # Our own socket first: the server serves one client at a time, so it
            # has to see this one close before it will accept the hello that asks
            # for the new playground.
            if self.link is not None:
                self.link.close()
                self.link = None
            self.playground_ref = self.playground_asked = ref
            self._say(f"asking {self.node} for {ref} on job {self.job_id}", SWITCH)
            # A playground given as a path has to be re-shipped: `_deploy` only ran
            # for the one this session started with. A bundled name came over with
            # the package and this returns at once.
            self._deploy_playground_file()
            self._check_still_alive()
            self.link = FrameLink.connect("127.0.0.1", self.local_port,
                                          self._token, timeout=15.0,
                                          on_notice=self._say,
                                          playground=self.playground_ref)
            self._say(f"streaming {ref} from {self.node} (job {self.job_id})",
                      READY)
        except SessionError as exc:
            # A process of ours has exited (see _check_still_alive), which for the
            # server means its own `scancel` has already ended the allocation. There
            # is nothing left to switch onto, so give the rest back and let the panel
            # start a fresh session -- the same thing every other step does when it
            # fails, and for the same reason.
            self.playground_ref, self.playground_asked = previous, previous_asked
            self.error = str(exc)
            self._say(f"FAILED: could not switch to {ref}: {exc}", FAILED)
            self._teardown()
        except (LinkClosed, OSError) as exc:
            self.playground_ref, self.playground_asked = previous, previous_asked
            self.link = None
            self.error = str(exc)
            self.state = DOWN
            self._say(f"could not switch to {ref}: {exc} -- the allocation "
                      f"(job {self.job_id}) is still held")
        except Exception as exc:                      # noqa: BLE001 -- reported
            self.playground_ref, self.playground_asked = previous, previous_asked
            self.link = None
            self.error = f"{type(exc).__name__}: {exc}"
            self.state = DOWN
            self._say(f"could not switch to {ref}: {self.error}")
        finally:
            self.prompt = None

    def connect_playground(self, ref):
        """What the panel's Connect button means, whichever situation it is in:
        get a GPU and put `ref` on it, or move the GPU we already hold to `ref`.

        One method rather than two buttons, because from where the user stands they
        are one action -- "run this one" -- and which of the two it takes is a fact
        about the session, not a decision they should have to make.
        """
        # A held allocation covers three cases and one answer: streaming another
        # playground, holding a link to this one that has died, or sitting there
        # after a switch that failed. All three are "open a socket, naming what we
        # want", which is what `_start_relink` does -- on a worker, because the far
        # side may have to build.
        if self.holds_allocation and self._start_relink(ref):
            return True
        if self.busy:
            return False
        self.playground_ref = self.playground_asked = ref
        self.start()
        return True

    def note_link_lost(self, reason):
        """Record that the link died on its own -- the job hit its time limit, the
        node failed, the network dropped. Called by whatever notices (the connect
        panel), because the session's own steps have all completed by then and it
        has nothing left watching."""
        if self.state != READY:
            return
        self.state = DOWN
        self.detail = reason or "the link closed"
        self._say(f"link lost: {self.detail}")

    def cancel(self):
        """Give up on a connection attempt in progress."""
        self._cancel.set()
        if self._bridge is not None:
            self._bridge.cancel()

    # ---- 1. the master connection -------------------------------------------

    def _ssh_base(self):
        return ["ssh", "-S", self._control_path,
                "-o", "BatchMode=no",
                "-o", "StrictHostKeyChecking=accept-new"]

    def _open_master(self):
        self._say(f"connecting to {self.target.destination}", LOGIN)
        self._bridge = PromptBridge(self._on_prompt)
        cmd = ["ssh", "-M", "-S", self._control_path, "-N", "-T",
               # ControlPersist=no: the master lives exactly as long as this
               # process holds it, so a crashed app cannot leave an
               # authenticated connection to the cluster lying around.
               "-o", "ControlPersist=no",
               "-o", "ControlMaster=yes",
               "-o", "BatchMode=no",
               "-o", "NumberOfPasswordPrompts=3",
               "-o", "StrictHostKeyChecking=accept-new",
               "-o", "ServerAliveInterval=30",
               "-o", "ServerAliveCountMax=4",
               self.target.destination]
        self._master = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, env=self._bridge.env(),
            start_new_session=True)
        threading.Thread(target=self._drain, args=(self._master.stdout, "ssh: "),
                         daemon=True).start()
        # The master is up when it will answer a check. Nothing else tells us: the
        # process starts before it has authenticated and stays running after.
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if self._cancel.is_set():
                raise SessionError("cancelled")
            if self._master.poll() is not None:
                raise SessionError(
                    "the SSH connection closed before it was ready -- see the log "
                    "above (a wrong password or code is the usual reason)")
            check = subprocess.run(self._ssh_base() + ["-O", "check",
                                                       self.target.destination],
                                   capture_output=True, text=True)
            if check.returncode == 0:
                self._say("logged in")
                return
            time.sleep(0.5)
        raise SessionError("timed out waiting for the SSH login")

    def _on_prompt(self, prompt):
        self.prompt = prompt
        self._say(f"waiting for: {prompt}")

    def _drain(self, stream, prefix=""):
        for line in iter(stream.readline, ""):
            self._log_line(prefix + line.rstrip())

    # ---- running things on the far side --------------------------------------

    def _login_shell(self, command):
        """`command`, wrapped so it survives the trip to the far side's shell.

        THE TRAP THAT COST A REAL DEBUG ROUND: `ssh host bash -lc 'a && b'` does NOT
        pass an argv array. ssh joins all of its arguments with spaces and hands the
        single resulting string to the remote login shell, which then parses it
        itself. So that invocation arrives as

            bash -lc a && b

        and since `-c` takes exactly ONE argument, the far side runs `a` with `b` as
        a positional parameter -- or, in the case that found this, ran a bare `mkdir`
        with its `-p` as $0 and reported "missing operand". Quoting the command into
        a single shell word is the fix; there is no argv-preserving mode to switch
        on. A login shell (`-l`) is still wanted, because that is what puts the
        cluster's module system and `env.sh`'s dependencies on PATH.
        """
        return "bash -lc " + shlex.quote(command)

    def _remote(self, command, timeout=120, check=True, login_shell=True):
        """Run one command over the master connection and return it completed."""
        if self._cancel.is_set():
            raise SessionError("cancelled")
        argv = self._ssh_base() + [self.target.destination]
        argv += [self._login_shell(command)] if login_shell else [command]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            raise SessionError(f"remote command timed out after {timeout}s: "
                               f"{command[:80]}") from None
        if check and proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            raise SessionError(f"remote command failed ({proc.returncode}): "
                               + (" / ".join(detail[-3:]) or "no output")
                               + f"  [{command[:70]}]")
        return proc

    def _env_prefix(self):
        """`cd` into the build and source its environment, in one string.

        Sourced in a login shell so the cluster's module system is available, which
        is what `env.sh` almost certainly needs.
        """
        parts = [f"cd {self.target.remote_dir}"]
        if self.target.env_script:
            parts.append(f"source {self.target.env_script}")
        return " && ".join(parts)

    # ---- 2. deploy -----------------------------------------------------------

    def _deploy(self):
        """Ship this package to the cluster.

        Sent as a tar over the master connection rather than installed: no pip, no
        virtualenv, nothing on the far side modified. The server runs with
        PYTHONPATH pointing at the unpacked directory. Compiled artifacts are
        excluded -- a macOS .dylib is no use there, and the cluster's pair style is
        in its own LAMMPS build anyway.
        """
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        parent = os.path.dirname(package_dir)
        name = os.path.basename(package_dir)
        self._say(f"shipping {name} to {self.target.deploy_dir}", DEPLOY)
        tar = subprocess.Popen(
            ["tar", "czf", "-", "-C", parent,
             "--exclude", "__pycache__", "--exclude", "*.pyc",
             "--exclude", "*.so", "--exclude", "*.dylib", "--exclude", "*.dll",
             "--exclude", "*.o", "--exclude", "*.build.json",
             "--exclude", "_mpi_stub", "--exclude", "_obj",
             name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        unpack = (f"mkdir -p {self.target.deploy_dir} && "
                  f"tar xzf - -C {self.target.deploy_dir}")
        remote = subprocess.Popen(
            self._ssh_base() + [self.target.destination,
                                self._login_shell(unpack)],
            stdin=tar.stdout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True)
        tar.stdout.close()
        out, _ = remote.communicate(timeout=300)
        tar.wait(timeout=10)
        if remote.returncode != 0:
            raise SessionError("could not unpack the package on the cluster: "
                               + ((out or "").strip()[-300:] or "no output")
                               + f"  [{unpack}]")
        self._deploy_playground_file()
        self._say("package in place")

    def _deploy_playground_file(self):
        """Ship a playground given as a PATH, and point the server at the copy.

        A bundled key needs nothing -- it came over with the package. But
        `--playground ./my_idea.py` is the whole point of the registry accepting a
        path, and it would otherwise fail on the far side with a file-not-found for
        a directory that only exists on this laptop.
        """
        ref = self.playground_ref
        if not (os.sep in ref or ref.endswith(".py")):
            return
        if ref.startswith(self.target.deploy_dir.rstrip("/") + "/"):
            # ALREADY SHIPPED: this is where a previous call put it, so `ref` is a
            # path on the FAR side. Re-sending it would look for it under that path
            # on this machine -- which on a real cluster does not exist, and on the
            # single-machine tests is the deployed copy itself, which `cat >` would
            # truncate before reading. Reachable because `switch_playground` can be
            # asked for a ref that has been through here before.
            return
        local = os.path.abspath(os.path.expanduser(ref))
        if not os.path.isfile(local):
            raise SessionError(f"no playground file at {local}")
        remote_path = f"{self.target.deploy_dir}/{os.path.basename(local)}"
        with open(local, "rb") as fh:
            proc = subprocess.run(
                self._ssh_base() + [self.target.destination,
                                    self._login_shell(f"cat > {remote_path}")],
                stdin=fh, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise SessionError("could not copy the playground file: "
                               + (proc.stderr or "").strip()[-200:])
        self.playground_ref = remote_path
        self._log_line(f"playground file -> {remote_path}")

    # ---- 3. probe ------------------------------------------------------------

    def _probe_command(self, level):
        return (f"{self._env_prefix()} && PYTHONPATH={self.target.deploy_dir} "
                f"{self.target.python} -m lammps_live.remote.probe --json "
                f"--profile {self.target.profile} --level {level}")

    def _probe(self):
        """Ask the LOGIN node what a login node can answer.

        Which is less than it first appears, and the difference cost a debugging
        round. The interpreter, numpy, and whether the `lammps` module and its
        shared library are where they should be: all of that is checkable here, for
        free, before any queue time is spent. But OPENING that library is not. A
        Kokkos/CUDA liblammps links against `libcuda.so.1` -- the NVIDIA driver --
        which is installed on GPU nodes and nowhere else, so the load fails on a
        login node with an error that looks exactly like a broken build and is not
        one. The real build check therefore runs on the node, inside the allocation
        (see `_probe_node`).
        """
        self._say("checking python and numpy on the login node", PROBE)
        proc = self._remote(self._probe_command("light"), timeout=600, check=False)
        report = None
        for line in proc.stdout.splitlines():
            if line.startswith("{"):
                try:
                    report = json.loads(line)
                except ValueError:
                    pass
        if report is None:
            raise SessionError(
                "the probe produced no report -- is python and the LAMMPS module "
                "reachable after sourcing " + str(self.target.env_script) + "? "
                + ((proc.stderr or proc.stdout).strip().splitlines() or [""])[-1])
        self.probe_report = report
        for line in report.get("summary") or []:
            self._log_line("probe: " + line)
        if not report.get("ok"):
            raise SessionError("the far side cannot run the server -- see the probe "
                               "lines above")
        self._say("python and numpy are fine")

    def _probe_node(self):
        """The real build check, on the GPU node, inside the allocation.

        This is where the library is actually opened, which is the only place it
        can be. It also answers the one question the server NEEDS rather than
        merely likes: how many values this build's `pair_coeff` takes, which
        decides the commands every slider will issue.

        A failure here is fatal but cheap in the only sense that matters: the
        allocation is already ours, so it is released on the way out.
        """
        self._say(f"checking the build on {self.node}", LAUNCH)
        # login_shell=False because the srun flags must reach the remote shell as
        # plain words; the probe command inside them is quoted by _login_shell.
        proc = self._remote(
            f"srun --jobid={self.job_id} --ntasks=1 "
            + "".join(f"{a} " for a in self.target.gpu_request())
            + f"--cpus-per-task={self.target.cpus_per_task} "
            + self._login_shell(self._probe_command("full")),
            timeout=900, check=False, login_shell=False)
        report = None
        for line in (proc.stdout + proc.stderr).splitlines():
            if line.startswith("{"):
                try:
                    report = json.loads(line)
                except ValueError:
                    pass
        if report is None:
            raise SessionError(
                "the build check on the node produced no report: "
                + ((proc.stderr or proc.stdout).strip().splitlines() or [""])[-1])
        self.probe_report = report
        for line in report.get("summary") or []:
            self._log_line("node: " + line)
        if not report.get("ok"):
            raise SessionError("this build cannot run the server on the node -- see "
                               "the lines above")
        self.coeff_values = (report.get("coeff") or {}).get("values")
        self._say("build checks out on the node")

    # ---- 4. allocate ---------------------------------------------------------

    def _allocate(self):
        target = self.target
        want = (f"{target.gres}" if target.gres else
                f"{target.gpus} GPU" if target.gpus else "cpus only")
        self._say(f"asking for {want} on {target.partition} "
                  f"for {target.time}", ALLOCATE)
        # NOTHING IS ASKED FOR AFTER THE SESSION HAS BEEN TOLD TO STOP. Every other
        # step reaches the cluster through `_remote`, which refuses once `_cancel`
        # is set; this one starts its own process, so it checks for itself. Without
        # it, a Disconnect landing here submits the allocation anyway -- and the
        # teardown that Disconnect ran has already been and gone.
        if self._cancel.is_set():
            raise SessionError("cancelled")
        self._submit_allocation()
        self._say(f"job {self.job_id} allocated; waiting for the node")
        self._await_node()

    def _submit_allocation(self):
        """Start `salloc --no-shell` and set `job_id` as soon as it names one.

        WATCHED AS A PROCESS, NOT WAITED ON AS A CALL, and that is the whole point:
        `--no-shell` returns as soon as the allocation is GRANTED, which on a busy
        GPU partition is not two minutes -- it is however long the people ahead of
        you take. Running it under a fixed-timeout `_remote` therefore failed the
        connect with "remote command timed out after 120s" whenever the queue was
        anything but empty, which is the ordinary case rather than the exceptional
        one.

        Slurm names the job the moment it is submitted ("salloc: Pending job
        allocation N"), long before it grants it, so reading the output live gets
        the id out of a still-queued request -- and an id is what turns the wait
        into something with a state, a reason and a Cancel behind it (`_await_node`)
        rather than an ssh sitting silently on a timeout.
        """
        argv = self._ssh_base() + [
            self.target.destination,
            self._login_shell(" ".join(self.target.salloc_args()))]
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)
        self._salloc_proc = proc
        named = threading.Event()
        found = {}
        lines = []

        def watch():
            for line in iter(proc.stdout.readline, ""):
                line = line.rstrip()
                if not line:
                    continue
                lines.append(line)
                self._log_line("salloc: " + line)
                match = re.search(r"(?:Pending|Granted) job allocation (\d+)", line)
                if match and "id" not in found:
                    found["id"] = match.group(1)
                    # PUBLISHED FROM THE WATCHER, not from the return value, and
                    # that ordering is the whole point: from the instant Slurm names
                    # a job there is something that can hold a GPU, and `_teardown`
                    # cancels whatever `job_id` holds. Setting it only once
                    # `_submit_allocation` returned left a window -- short, but
                    # exactly the one a Disconnect during the queue wait lands in --
                    # where the allocation existed and the session did not know its
                    # number, so the teardown found nothing to cancel and the job
                    # ran on alone.
                    self.job_id = match.group(1)
                    named.set()

        reader = threading.Thread(target=watch, name="salloc-log", daemon=True)
        reader.start()
        deadline = time.monotonic() + self.target.queue_wait
        while not named.wait(0.5):
            if self._cancel.is_set():
                raise SessionError("cancelled")
            if proc.poll() is not None:
                # Drain first: a salloc that failed outright exits within
                # milliseconds of printing why, and the reader thread may not have
                # been scheduled yet. Reporting "no output" there would throw away
                # the only line that says what went wrong.
                reader.join(timeout=5.0)
                if named.is_set():
                    break
                raise SessionError("salloc did not grant an allocation: "
                                   + (lines[-1] if lines else "no output"))
            if time.monotonic() > deadline:
                raise SessionError(
                    f"salloc never named a job in "
                    f"{self.target.queue_wait / 60:.0f} minutes")

    def _await_node(self):
        """Poll the queue until the job is running and has a node.

        A queued job is normal and can sit for the best part of an hour, so this
        reports Slurm's own state and reason -- "queued: PENDING Resources", with
        the minutes so far once there are any -- rather than spinning silently. The
        poll eases off after the first minute: a busy queue is answered by waiting,
        not by asking more often, and every ask is an ssh round trip on the shared
        master connection.
        """
        deadline = time.monotonic() + self.target.queue_wait
        started = time.monotonic()
        last = None
        while time.monotonic() < deadline:
            if self._cancel.is_set():
                raise SessionError("cancelled")
            # Explicitly separated fields: the padded --Format output is
            # ambiguous when a pending job has no node list to print.
            proc = self._remote(f"squeue -h -j {self.job_id} -o '%T|%N|%r'",
                                timeout=60, check=False)
            line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
            if not line:
                raise SessionError(f"job {self.job_id} is no longer in the queue")
            fields = (line.split("|") + ["", "", ""])[:3]
            state, node, reason = (f.strip() for f in fields)
            if state == "RUNNING" and node and not node.startswith("("):
                self.node = self._single_hostname(node)
                self._say(f"got {self.node}")
                return
            waited = time.monotonic() - started
            status = f"queued: {state} {reason}".strip()
            if waited >= 60:
                status += f" ({waited / 60:.0f} min)"
            if status != last:
                self._say(status)
                last = status
            time.sleep(3.0 if waited < 60 else 15.0)
        raise SessionError(
            f"the allocation never started within "
            f"{self.target.queue_wait / 60:.0f} minutes -- the queue is busy; "
            f"the job has been cancelled")

    def _single_hostname(self, nodelist):
        """One real hostname from what squeue printed.

        A one-node allocation prints a plain name, which is the whole story 99% of
        the time. But Slurm's `%N` is a nodelist, and a nodelist can be a bracketed
        range -- which is not something `ssh` can connect to, and would fail with a
        name-resolution error that says nothing about why. `scontrol show hostnames`
        is the canonical expansion, so ask it rather than guess.
        """
        if "[" not in nodelist:
            return nodelist
        proc = self._remote(f"scontrol show hostnames {nodelist}", timeout=60,
                            check=False)
        names = [n.strip() for n in proc.stdout.split() if n.strip()]
        if not names:
            raise SessionError(f"could not expand the node list {nodelist!r}")
        if len(names) > 1:
            self._multi_node = True
            self._log_line(f"allocation spans {len(names)} nodes; the server runs "
                           f"on whichever one srun picks")
        return names[0]

    # ---- 5. launch -----------------------------------------------------------

    def _server_command(self):
        target = self.target
        server = (
            f"{target.python} -u -m lammps_live.remote.server "
            f"--playground {self.playground_ref} "
            f"--profile {target.profile} "
            f"--port {target.port} --bind {target.server_bind} --token-stdin "
            f"--fps {target.fps} --codec {target.codec} "
            f"--exit-when-idle {target.exit_when_idle}"
        )
        if target.free_run:
            server += " --free-run"
        if self.coeff_values:
            server += f" --coeff-values {self.coeff_values}"
        # The server cancels its own allocation on the way out, whatever took it
        # out: a finished session, --exit-when-idle after an abandoned one, or a
        # crash. That is the backstop that does not depend on this laptop still
        # being alive to run scancel.
        return (f"{self._env_prefix()} && "
                f"PYTHONPATH={target.deploy_dir} {server}; "
                f"rc=$?; echo \"[server] exited with $rc\"; "
                f"scancel $SLURM_JOB_ID; exit $rc")

    def _launch(self):
        target = self.target
        self._say(f"starting the server on {self.node}", LAUNCH)
        argv = self._ssh_base() + [
            target.destination,
            "srun", f"--jobid={self.job_id}", "--unbuffered",
            f"--ntasks={target.ntasks}", *target.gpu_request(),
            f"--cpus-per-task={target.cpus_per_task}",
            # The srun flags are plain words and survive being joined; the command
            # itself has to be one quoted word (see _login_shell).
            self._login_shell(self._server_command())]
        self._server_proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, start_new_session=True)
        # The token goes down stdin, which srun forwards to the task, so it never
        # appears in a command line that the rest of the cluster can read.
        self._server_proc.stdin.write(self._token + "\n")
        self._server_proc.stdin.flush()

        listening = threading.Event()
        port_seen = {}

        def watch():
            for line in iter(self._server_proc.stdout.readline, ""):
                self._log_line(line.rstrip())
                match = re.search(r"LISTENING host=(\S+) port=(\d+)", line)
                if match:
                    port_seen["host"] = match.group(1)
                    port_seen["port"] = int(match.group(2))
                    listening.set()

        threading.Thread(target=watch, name="server-log", daemon=True).start()
        # Ten minutes is generous on purpose: it covers LAMMPS starting and Kokkos
        # initialising the GPU, which is the part that does not get faster (the
        # placement, which used to dominate this wait, now does not -- see
        # RandomFill.build). But the srun is watched while we wait, because the
        # common failures here (a step that cannot be created, a module that is not
        # loaded, a syntax error in env.sh) exit in seconds and there is no reason to
        # sit out the full timeout for them.
        deadline = time.monotonic() + 600
        while not listening.wait(1.0):
            if self._cancel.is_set():
                raise SessionError("cancelled")
            if self._server_proc.poll() is not None:
                raise SessionError(
                    f"the server exited (status {self._server_proc.returncode}) "
                    f"before it was listening -- see the log above")
            if time.monotonic() > deadline:
                raise SessionError("the server did not come up within 10 minutes")
        self._remote_port = port_seen["port"]
        self._server_host = port_seen["host"]
        self._say(f"server listening on {self._server_host}:{self._remote_port}")
        # WHERE THE SERVER ACTUALLY IS, versus where we are about to tunnel to.
        # `squeue` gave us the allocation's node list and we took the first name;
        # `srun --ntasks=1` is free to place that one task on ANY node in the
        # allocation. When those disagree the tunnel goes to a machine where nothing
        # is listening, and in "jump" mode nothing else would ever notice, because
        # the server binds loopback on ITS node and the forward is simply refused at
        # the far end -- which reaches the user as "the server did not answer" and
        # says nothing about why.
        #
        # Only a multi-node allocation gets its route changed. A single node that
        # reports a different name is reporting a NAME, not a different machine:
        # `socket.gethostname()` there may be an FQDN, or an interface alias that
        # `ssh` cannot resolve from the login node, and preferring it over the
        # canonical Slurm name would break a route that works. So say so and keep
        # going; the mismatch is in the log and in the report either way.
        if not _same_host(self._server_host, self.node):
            if self._multi_node:
                self._say(f"the server came up on {self._server_host}, not "
                          f"{self.node}; the tunnel will go there instead")
                self.node = self._server_host
            else:
                self._log_line(f"note: the node calls itself {self._server_host}, "
                               f"Slurm calls it {self.node}; tunnelling to "
                               f"{self.node}")

    # ---- 6. tunnel and connect ----------------------------------------------

    def _tunnel(self):
        """Bring a local port to the server, by whichever route the target asks for.

        Both routes end with `self.local_port` being a port on this machine that
        reaches the server, and that is all the client knows about it.
        """
        if self.target.tunnel == "jump":
            self._tunnel_two_hop()
        else:
            self._tunnel_one_hop()

    def _free_local_ports(self):
        """Candidate local ports, starting at the requested one, skipping any that
        are already listening -- a second app, or a forward left over from a session
        that did not shut down cleanly."""
        wanted = self.target.local_port
        return [p for p in range(wanted, wanted + 20) if not self._port_in_use(p)]

    def _tunnel_two_hop(self):
        """A second SSH whose session ENDS ON THE COMPUTE NODE (the default).

        `-W %h:%p` makes the jump hop a stdio pipe over the connection that is
        already authenticated, so nothing new is typed. Everything after that is an
        ordinary SSH session to the node, and `-L <local>:127.0.0.1:<port>` asks it
        to carry the port -- `127.0.0.1` here means loopback ON THE NODE, which is
        the whole point.

        Host keys go in a session-local file rather than the user's `known_hosts`.
        Compute nodes are reimaged and their keys change, so pinning them in the
        real file turns a routine node reallocation into a hard failure and a scary
        warning weeks later; and a per-session file still pins the key for the
        lifetime of the session. The MITM this gives up on is the login node itself,
        which already has the user's home directory and Slurm credentials.
        """
        self._say(f"opening the tunnel to {self.node}", TUNNEL)
        # The ProxyCommand is handed to /bin/sh by ssh, so the control path is
        # quoted: TMPDIR is not guaranteed to be free of spaces. `%h:%p` must stay
        # unquoted-looking, and does -- ssh substitutes those tokens itself before
        # the shell ever sees the string.
        proxy = (f"ssh -S {shlex.quote(self._control_path)} "
                 f"-o ControlMaster=no -W %h:%p "
                 f"{self.target.destination}")
        known_hosts = os.path.join(self._tmpdir, "node_known_hosts")
        destination = self.target.node_destination(self.node)
        candidates = self._free_local_ports()
        if not candidates:
            raise SessionError(
                f"no free local port near {self.target.local_port} -- another "
                f"session may still be running")
        last = ""
        for candidate in candidates:
            spec = f"{candidate}:127.0.0.1:{self._remote_port}"
            cmd = ["ssh", "-N", "-T",
                   "-o", f"ProxyCommand={proxy}",
                   # NO MULTIPLEXING ON THIS HOP. The cost of getting this wrong was
                   # a whole debugging session, so at length:
                   #
                   # A `~/.ssh/config` that sets `ControlMaster auto` and
                   # `ControlPersist` for the cluster's compute nodes -- which is a
                   # sensible thing to have, and the reason `gcn*` is in there is to
                   # make ssh-ing to your own job's node cheap -- changes what
                   # `ssh -N -L ...` MEANS. With ControlPersist and no remote
                   # command, ssh puts the master in the BACKGROUND and the process
                   # we started exits immediately, status 0. So:
                   #
                   #   - the local port is bound for a moment, so the forward looks
                   #     like it came up, and
                   #   - the process holding it is gone, and the forward with it, so
                   #     the connect a fraction of a second later is refused --
                   #     reported as "the tunnel is open but the server did not
                   #     answer", which is true and useless.
                   #
                   # `ControlPath=none` disables multiplexing for this connection
                   # outright: no master to join, no ControlPersist, nothing
                   # backgrounds itself, and `-N` means what this code assumes it
                   # means -- a process that stays in the foreground for exactly as
                   # long as the forward exists. The login master does the same
                   # thing for the same reason (see `_open_master`); this hop was
                   # missing it.
                   "-o", "ControlMaster=no",
                   "-o", "ControlPath=none",
                   # Fail loudly if the port cannot be bound, rather than sitting
                   # there with a connection that carries nothing.
                   "-o", "ExitOnForwardFailure=yes",
                   "-o", f"UserKnownHostsFile={known_hosts}",
                   "-o", "StrictHostKeyChecking=accept-new",
                   "-o", "ServerAliveInterval=30",
                   "-o", "ServerAliveCountMax=4",
                   "-L", spec, destination]
            # LAMMPS_LIVE_SSH_VERBOSE=1 puts ssh's own trace in the session log.
            # Worth having as a switch rather than always on: `-v` explains a
            # channel that could not be opened and a session the node closed --
            # the two failures that otherwise reach the user as nothing but
            # "the server did not answer" -- at the cost of fifty lines of noise
            # in front of every successful connect.
            if os.environ.get("LAMMPS_LIVE_SSH_VERBOSE"):
                cmd.insert(1, "-v")
            # The askpass bridge is passed along even though Snellius does not ask
            # again: a site where the node DOES prompt then shows its question in
            # the panel instead of hanging on a terminal that is not there.
            proc = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, env=self._bridge.env(),
                start_new_session=True)
            threading.Thread(target=self._drain, args=(proc.stdout, "tunnel: "),
                             daemon=True).start()
            if self._await_forward(proc, candidate):
                self._tunnel_proc = proc
                self.local_port = candidate
                if candidate != self.target.local_port:
                    self._say(f"local port {self.target.local_port} was busy; "
                              f"using {candidate}")
                self._say(f"tunnel up: 127.0.0.1:{candidate} -> "
                          f"{self.node}:{self._remote_port}")
                return
            last = f"the tunnel to {destination} exited (status {proc.returncode})"
            if proc.returncode == 0:
                # The signature of an ssh that backgrounded itself rather than one
                # that failed: a clean exit from a process that was asked to stay.
                last += (" without failing -- something is putting that connection "
                         "in the background. Check for ControlMaster / "
                         "ControlPersist settings this hop is not already "
                         "overriding")
            try:
                proc.kill()
            except OSError:
                pass
        raise SessionError(last or "could not open the tunnel -- see the log above")

    def _await_forward(self, proc, port, timeout=60.0):
        """True once `port` is listening AND `proc` is still the one holding it.

        Polling the port is the only honest signal that the forward exists: `ssh -N`
        reports success by continuing to run, and it is already running before the
        forward is set up.

        But a listening port on its own is not enough, which is the whole lesson of
        the ControlPersist trap above: an ssh that backgrounds itself binds the port
        and exits, and for a moment both "the port answers" and "the process is
        gone" are true. So the process is checked at the same time as the port, and
        an exited one fails this candidate however healthy the port looks.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._cancel.is_set():
                raise SessionError("cancelled")
            exited = proc.poll() is not None
            if self._port_in_use(port) and not exited:
                return True
            if exited:
                return False
            time.sleep(0.2)
        return False

    def _tunnel_one_hop(self):
        """`-O forward` on the login-node master: the fallback.

        Adds the forward to the connection that is already authenticated, so there
        is nothing new to type -- but the login node terminates the session and
        makes its own plain-TCP connection onward, which is why this mode also makes
        the server bind 0.0.0.0 (see RemoteTarget.server_bind) and why it is not the
        default.
        """
        self._say("opening the tunnel (one hop, via the login node)", TUNNEL)
        candidates = self._free_local_ports()
        if not candidates:
            raise SessionError(
                f"no free local port near {self.target.local_port} -- another "
                f"session may still be running")
        last_error = ""
        for candidate in candidates:
            spec = f"{candidate}:{self.node}:{self._remote_port}"
            proc = subprocess.run(
                self._ssh_base() + ["-O", "forward", "-L", spec,
                                    self.target.destination],
                capture_output=True, text=True)
            if proc.returncode == 0:
                self.local_port = candidate
                self._forwarded = spec
                if candidate != self.target.local_port:
                    self._say(f"local port {self.target.local_port} was busy; "
                              f"using {candidate}")
                return
            last_error = (proc.stderr or proc.stdout).strip()
        raise SessionError(f"could not open a local forward: {last_error}")

    @staticmethod
    def _port_in_use(port):
        with socket.socket() as probe:
            probe.settimeout(0.2)
            return probe.connect_ex(("127.0.0.1", port)) == 0

    def _connect(self):
        self._say(f"connecting to 127.0.0.1:{self.local_port}")
        # The first attempt can beat the forward into place by a few milliseconds.
        last = None
        for attempt in range(10):
            if self._cancel.is_set():
                raise SessionError("cancelled")
            # Both ends of the thing we are connecting through, checked every time
            # round. Retrying nine more times against a process that has already
            # exited only delays the real message -- and one of these dying is the
            # commonest way this step fails: the server's own `scancel` on the way
            # out ends the allocation, and Slurm then kills the ssh session on the
            # node that the tunnel is made of, so the local port stops listening a
            # moment after the server goes.
            self._check_still_alive()
            try:
                self.link = FrameLink.connect("127.0.0.1", self.local_port,
                                              self._token, timeout=15.0,
                                              on_notice=self._say)
                return
            except LinkClosed as exc:
                # Each DISTINCT reason, once: ten identical lines say nothing, but a
                # refusal that turns into a handshake failure on the fourth try is
                # the whole story and used to be invisible.
                if str(exc) != str(last):
                    self._log_line(f"connect attempt {attempt + 1}: {exc}")
                last = exc
                time.sleep(0.5)
        raise SessionError(f"the tunnel is open but the server did not answer: "
                           f"{last} -- {self._connect_postmortem()}")

    def _check_still_alive(self):
        """Raise if the server or the tunnel has exited under us."""
        server = self._server_proc
        if server is not None and server.poll() is not None:
            raise SessionError(
                f"the server exited (status {server.returncode}) after saying it "
                f"was listening -- see the [server] lines in the log. Its own "
                f"`scancel` then ends the allocation, which is why the tunnel goes "
                f"down with it")
        tunnel = self._tunnel_proc
        if tunnel is not None and tunnel.poll() is not None:
            raise SessionError(
                f"the tunnel ssh exited (status {tunnel.returncode}) after opening "
                f"the forward -- see the `tunnel:` lines in the log. A job that has "
                f"ended takes the session on the node with it; "
                f"LAMMPS_LIVE_SSH_VERBOSE=1 adds ssh's own trace")

    def _connect_postmortem(self):
        """Which end broke, established while the evidence is warm.

        "The server did not answer" is three different failures wearing one
        message, and a client socket cannot tell them apart:

            nothing listening on this end   the ssh holding the forward has exited,
                                            so the connect is refused locally and
                                            the far side was never involved
            listening, refused at the far   the forward is up but the server is not
                                            on that node or not on that port
            connected, handshake failed     both ends are alive and disagree about
                                            the protocol version or the token

        The first two are indistinguishable from the outside -- both arrive as a
        failed `connect()` -- which is why this asks directly rather than guessing.
        """
        facts = [self._describe_local_port()]
        for name, proc in (("the tunnel ssh", self._tunnel_proc),
                           ("the srun holding the server", self._server_proc),
                           ("the login connection", self._master)):
            if proc is None:
                continue
            code = proc.poll()
            facts.append(f"{name} is still running" if code is None
                         else f"{name} has exited (status {code})")
        if self._server_host and self.node and not _same_host(self._server_host,
                                                              self.node):
            facts.append(f"the server reported host {self._server_host} but the "
                         f"tunnel goes to {self.node}")
        return ("; ".join(facts) + ". The log has both ends' own output; "
                "LAMMPS_LIVE_SSH_VERBOSE=1 adds ssh's trace to it.")

    # ---- teardown ------------------------------------------------------------

    def shutdown(self):
        """Give everything back. Safe to call from anywhere, more than once."""
        if self.state in (DOWN, CLOSING):
            self._teardown()
            return
        self._say("closing down", CLOSING)
        self._cancel.set()
        self._teardown()
        self.state = DOWN
        self.detail = "not connected"

    def _teardown(self):
        """Undo everything, in the order that leaves nothing stranded.

        Stop drawing, stop the job, stop the forward, close the login. The link
        goes first because cancelling underneath it would leave the client blocked
        on a socket that will never answer again; the job goes next, ahead of the
        tunnel and the login, because it is the only one of these that costs
        anything to still be holding a minute from now -- and the tunnel's kill
        alone is worth ten seconds of waiting that the GPU should not be behind.

        Each field is taken and cleared under the lock before it is used, so two
        threads arriving here at once (the worker abandoning a failed step, the
        window closing) cannot trip over each other's cleanup.
        """
        with self._teardown_lock:
            link, self.link = self.link, None
            server, self._server_proc = self._server_proc, None
            salloc, self._salloc_proc = self._salloc_proc, None
            tunnel, self._tunnel_proc = self._tunnel_proc, None
            job_id, self.job_id = self.job_id, None
            forwarded, self._forwarded = self._forwarded, None
            master, self._master = self._master, None
            bridge, self._bridge = self._bridge, None
            tmpdir, self._tmpdir = self._tmpdir, None
            control = self._control_path
            self.node = None
            self.local_port = None

        if link is not None:
            try:
                link.close()
            except Exception:                          # noqa: BLE001 -- best effort
                pass
        if server is not None:
            # Killing the srun ends the step; the remote shell's own `scancel`
            # then ends the allocation. The explicit scancel below is what
            # actually guarantees it -- this is just the quick way.
            try:
                server.terminate()
            except Exception:                          # noqa: BLE001
                pass
        if salloc is not None and salloc.poll() is None:
            # Still queued, then -- a granted salloc --no-shell has long since
            # exited. Killing it is what gives up the place in the queue: a
            # pending request that nobody is waiting for would otherwise be
            # granted later and hold a GPU with nothing on it. Dropping the ssh
            # hangs up on the far side's salloc, which cancels its own pending
            # request; the scancel below is the guarantee, this is the immediate
            # one and it does not need another round trip to be sure of.
            try:
                salloc.terminate()
            except Exception:                          # noqa: BLE001
                pass
        if job_id and not self._release_job(job_id, control):
            # NOT CONFIRMED GONE, so it is still ours: put the id back rather than
            # forgetting it. A second teardown then tries again, the report the
            # panel copies names the job that has to be cancelled by hand, and the
            # one failure with a bill attached stops being silent.
            self.job_id = job_id
        if tunnel is not None:
            # The two-hop tunnel is a process of its own; killing it takes the
            # forward and the session on the node with it.
            try:
                tunnel.terminate()
                tunnel.wait(timeout=10)
            except subprocess.TimeoutExpired:
                tunnel.kill()
            except OSError:
                pass
        if forwarded and control and os.path.exists(control):
            self._control_op(control, ["-O", "cancel", "-L", forwarded])
        if master is not None:
            self._control_op(control, ["-O", "exit"])
            try:
                master.wait(timeout=10)
            except subprocess.TimeoutExpired:
                master.kill()
        if bridge is not None:
            bridge.close()
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
            self._control_path = None

    # How long the release may spend on any one ssh. Short on purpose: this runs
    # while the window is closing, and a laptop being shut must not sit out a
    # two-minute timeout on a connection that is already gone. Two routes are tried
    # inside this budget, so the whole release is bounded by about four times it.
    RELEASE_TIMEOUT = 15

    # What `squeue` prints for a job that still has the GPU. Anything else -- no
    # line at all, or a state that has stopped -- means the scancel took. CANCELLED
    # and COMPLETING both appear for a few seconds after a successful scancel and
    # must NOT be read as a failure, or every clean teardown would report one.
    HOLDING_STATES = ("PENDING", "RUNNING", "CONFIGURING", "SUSPENDED", "RESIZING")

    def _release_job(self, job_id, control):
        """Cancel `job_id`, and check with Slurm that it actually went.

        TWO ROUTES, BECAUSE THE CONTROL MASTER IS NOT A GUARANTEE. It is the first
        thing a dropped network takes and the last thing a teardown can lean on,
        and there are ordinary ways to reach here without one: a session torn down
        twice (the second teardown has already removed the socket), a connect
        thread that got its job id a moment after the window closed, an ssh master
        that died while the allocation lived on. Every one of those used to end in
        the same place -- the `scancel` quietly skipped, and an A100 held until
        Slurm's own `--time` ran out an hour later.

        So the master is tried first (it costs nothing and needs no authentication)
        and a connection of its own second. That one is BatchMode: with no
        interactive fallback it fails in seconds against a cluster that would ask
        for a password, rather than stopping the shutdown on a prompt that nobody
        is there to answer.

        FIRING IT IS NOT THE SAME AS IT HAVING WORKED, which is why each route ends
        with a `squeue`. Returns True only when Slurm agrees the job has stopped.
        """
        routes = []
        if control and os.path.exists(control):
            routes.append(["ssh", "-S", control])
        routes.append(["ssh", "-o", "BatchMode=yes",
                       "-o", "StrictHostKeyChecking=accept-new",
                       "-o", f"ConnectTimeout={self.RELEASE_TIMEOUT}"])
        for prefix in routes:
            self._release_over(prefix, f"scancel {job_id}")
            state = self._job_state(prefix, job_id)
            if state is not None and state not in self.HOLDING_STATES:
                self._log_line(f"scancel {job_id}: released"
                               + (f" ({state})" if state else ""))
                return True
        self._log_line(
            f"scancel {job_id} COULD NOT BE CONFIRMED -- the job may still be "
            f"holding a GPU. Check it with `squeue -j {job_id}` and cancel it by "
            f"hand if it is still there.")
        return False

    def _release_over(self, prefix, command):
        """One teardown command down one route. Never raises: a route that does not
        work is the reason the next one is tried."""
        try:
            return subprocess.run(prefix + [self.target.destination,
                                            self._login_shell(command)],
                                  capture_output=True, text=True,
                                  timeout=self.RELEASE_TIMEOUT)
        except (OSError, subprocess.SubprocessError):
            return None

    def _job_state(self, prefix, job_id):
        """Slurm's state for `job_id`: "" if it is no longer in the queue at all,
        None if the question could not be asked (which is not an answer, and must
        not be mistaken for one)."""
        proc = self._release_over(prefix, f"squeue -h -j {job_id} -o %T")
        if proc is None:
            return None
        text = (proc.stdout or "").strip()
        if proc.returncode != 0:
            # `squeue` refuses an id it has never heard of ("Invalid job id
            # specified"), which is the strongest possible confirmation: Slurm has
            # forgotten the job entirely. Any other failure is the ssh's, and says
            # nothing about the job.
            if "invalid job id" in (proc.stderr or "").lower():
                return ""
            return None
        # The fields are separated for the caller elsewhere; take the first
        # whatever the separator, so this reads the same output `_await_node` does.
        return text.replace("|", " ").split()[0] if text else ""

    def _control_op(self, control, args):
        """One `ssh -O ...` control operation, ignoring its outcome -- during
        teardown there is nothing useful to do about a failure."""
        if not control:
            return
        subprocess.run(["ssh", "-S", control] + args + [self.target.destination],
                       capture_output=True, text=True)


# --- driving it from a terminal -----------------------------------------------

def main(argv=None):
    """Run the whole flow with the prompts on the terminal.

    This is how to debug the SSH and Slurm half without the GUI in the way: it
    shows every step, asks for the password / one-time code on the terminal, and
    then streams frames and prints what it is receiving until interrupted.
    """
    import argparse
    import getpass

    from ..playground import registry

    parser = argparse.ArgumentParser(
        prog="python -m lammps_live.remote.session",
        description=main.__doc__.splitlines()[0])
    parser.add_argument("--playground", default="mesomem_remote")
    parser.add_argument("--seconds", type=float, default=20.0,
                        help="how long to stream before tearing down")
    parser.add_argument("--play", action="store_true",
                        help="run the simulation, rather than only connecting")
    args = parser.parse_args(argv)

    playground = registry.load(args.playground)
    if playground.remote is None:
        raise SystemExit(f"{args.playground} declares no remote target")
    session = RemoteSession(playground.remote, playground_ref=args.playground,
                            on_log=lambda line: print(line, flush=True))
    session.start()
    try:
        while session.busy:
            if session.prompt:
                prompt = session.prompt
                session.prompt = None
                session.answer(getpass.getpass(prompt.rstrip() + " "))
            time.sleep(0.2)
        if session.state != READY:
            # The whole report, on disk, before the teardown in `finally` clears the
            # facts it is made of. The terminal has the log already; the file is
            # what gets attached to the mail to whoever runs the cluster.
            path = session.save_report()
            raise SystemExit(f"\nfailed: {session.error}"
                             + (f"\nreport: {path}" if path else ""))
        link = session.link
        print(f"\nconnected: {link.welcome}\n")
        if args.play:
            link.send({"t": "play"})
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline and not link.closed.is_set():
            frame = link.take_frame(timeout=1.0)
            if frame is None:
                continue
            header, payload = frame
            fps, mbs = link.rates()
            print(f"frame {header['seq']:5d}  n={header['n']}  "
                  f"t={header['sim_time']:8.2f} tau  "
                  f"{len(payload) / 1024:6.1f} kB  {fps:4.0f} f/s  "
                  f"{mbs:5.2f} MB/s  rtt {link.rtt_ms or 0:.0f} ms", flush=True)
            link.ping()
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        print("\ntearing down (this cancels the job)...")
        session.shutdown()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
