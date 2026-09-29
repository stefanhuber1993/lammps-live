"""Write the fake cluster's executables into a directory, and return it."""
import os
import stat
import sys
import textwrap

# --- the fake ssh -------------------------------------------------------------
# The invocations session.py makes, in the order it makes them:
#   ssh -M -S ctl -N -T ... dest                 the master: prompt, then hold
#   ssh -S ctl -O check dest                     is the master up yet?
#   ssh -S ctl dest bash -lc CMD                 run something over it
#   ssh -S ctl dest srun --jobid=.. bash -lc CMD run the server "on a node"
#   ssh -S ctl -O forward -L a:b:c dest          add a port forward
#   ssh -S ctl -O cancel -L a:b:c dest           remove it
#   ssh -S ctl -O exit dest                      close the master
_SSH = r'''#!@PYTHON@
import os, socket, subprocess, sys, threading, time

argv = sys.argv[1:]
def opt(name):
    return argv[argv.index(name) + 1] if name in argv else None

control = opt("-S")
prompts = os.environ.get("FAKE_SSH_PROMPTS", "Password:|Verification code:")
expected = os.environ.get("FAKE_SSH_ANSWERS", "")
log = open(os.environ["FAKE_SSH_LOG"], "a")

def note(*bits):
    print(*bits, file=log, flush=True)

# --- the master connection ---------------------------------------------------
if "-M" in argv:
    note("master", " ".join(argv))
    askpass = os.environ.get("SSH_ASKPASS")
    if not askpass:
        print("fake ssh: no SSH_ASKPASS was set", file=sys.stderr)
        sys.exit(1)
    answers = []
    for prompt in prompts.split("|"):
        if not prompt:
            continue
        proc = subprocess.run([askpass, prompt], capture_output=True, text=True)
        if proc.returncode != 0 or not proc.stdout.strip():
            print("fake ssh: Permission denied (prompt cancelled)", file=sys.stderr)
            sys.exit(255)
        answers.append(proc.stdout.strip())
    if expected and "|".join(answers) != expected:
        print("fake ssh: Permission denied, please try again.", file=sys.stderr)
        note("bad answers", answers)
        sys.exit(255)
    note("authenticated", answers)
    # `-O exit` has to be able to end us, as it ends a real master -- a fake that
    # ignored it made every teardown sit out the full reaping grace, which hid how
    # long the real one takes. SIGTERM unwinds through the `finally`, so the
    # control socket goes with the process exactly as ssh removes its own.
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    open(control + ".pid", "w").write(str(os.getpid()))
    open(control, "w").write("up\n")
    print("fake ssh: master connected", flush=True)
    try:
        while True:
            time.sleep(0.2)
    finally:
        try:
            os.unlink(control)
        except OSError:
            pass

# --- the two-hop tunnel ------------------------------------------------------
# `ssh -N -T -o ProxyCommand=... -L <local>:127.0.0.1:<remote> user@node`. There is
# no -S here: this is a session to the NODE, whose jump hop is the ProxyCommand
# riding the master. It holds the forward for as long as it runs, so this process
# becomes the proxy and stays alive until it is killed -- exactly as the real one
# does.
if "-N" in argv and "-L" in argv and "-M" not in argv:
    proxied = [argv[i + 1] for i, a in enumerate(argv) if a == "-o"]
    note("tunnel", opt("-L"), "|", " ".join(proxied))
    local, host, remote = opt("-L").split(":")
    os.execv(sys.executable,
             [sys.executable,
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "proxy.py"),
              local, host, remote])

# A CONNECTION OF ITS OWN, with no master to ride. This is the route the teardown
# falls back to when the control socket has gone, and whether it works is a
# property of the cluster rather than of the app: it needs a key the agent can
# offer, because `BatchMode=yes` means there is no prompt to fall back on. So it
# is off unless the test says this cluster has one, and off is the honest default
# -- a fake that always let a keyless ssh through would prove nothing about the
# machine that asks for a one-time code.
if control is None:
    if os.environ.get("FAKE_SSH_DIRECT") != "1":
        note("direct refused", " ".join(argv))
        print("fake ssh: Permission denied (publickey,keyboard-interactive).",
              file=sys.stderr)
        sys.exit(255)
elif not os.path.exists(control):
    print("fake ssh: no control connection", file=sys.stderr)
    sys.exit(255)

# --- control operations ------------------------------------------------------
if "-O" in argv:
    action = opt("-O")
    note("control", action, opt("-L") or "")
    if action == "check":
        sys.exit(0)
    if action == "forward":
        local, host, remote = opt("-L").split(":")
        # A real forward lives inside the master connection; here it is a child
        # process that proxies until it is cancelled, which is close enough to
        # exercise everything on this side of it.
        pid_path = control + ".fwd." + local
        # devnull, not the inherited pipes: the caller reads our output until EOF,
        # and a detached grandchild holding the write end would hang it forever.
        # (It did. That is what this comment is for.)
        proxy = subprocess.Popen(
            [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "proxy.py"), local, host, remote],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)
        open(pid_path, "w").write(str(proxy.pid))
        # Wait for the listener, so a connect straight after this cannot lose a race.
        for _ in range(100):
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", int(local))) == 0:
                    break
            time.sleep(0.05)
        sys.exit(0)
    if action == "cancel":
        local = opt("-L").split(":")[0]
        pid_path = control + ".fwd." + local
        if os.path.exists(pid_path):
            try:
                os.kill(int(open(pid_path).read()), 15)
            except OSError:
                pass
            os.unlink(pid_path)
        sys.exit(0)
    if action == "exit":
        try:
            os.kill(int(open(control + ".pid").read()), 15)
        except (OSError, ValueError):
            pass
        sys.exit(0)
    sys.exit(1)

# --- running a command on "the cluster" --------------------------------------
# FAITHFULLY: real ssh does NOT pass an argv array to the far side. It JOINS every
# argument after the destination with single spaces and hands the one resulting
# string to the remote user's shell, which parses it itself. An earlier version of
# this fake took argv[-1] as the command, which quietly made `ssh host bash -lc 'a
# && b'` work here and fail on the real cluster (bash -c takes one argument, so the
# far side ran `a` alone). A stand-in that is more forgiving than the real thing
# hides exactly the bugs it exists to catch, so this joins like ssh does.
FLAG_WITH_VALUE = {"-S", "-o", "-O", "-L", "-R", "-D", "-i", "-p", "-F", "-J", "-l",
                   "-c", "-E", "-I", "-m", "-Q", "-w", "-W", "-b"}
i, dest = 0, None
while i < len(argv):
    if argv[i] in FLAG_WITH_VALUE:
        i += 2
        continue
    if argv[i].startswith("-"):
        i += 1
        continue
    dest = i
    break
remote = argv[dest + 1:] if dest is not None else []
if remote:
    joined = " ".join(remote)
    note("run", joined)
    # -c, not -lc: a login shell here would source the developer's own profile.
    sys.exit(subprocess.run(["bash", "-c", joined]).returncode)
note("unhandled", " ".join(argv))
sys.exit(2)
'''

_PROXY = r'''#!@PYTHON@
"""localhost:LOCAL -> HOST:REMOTE, until killed. Stands in for an SSH -L forward."""
import socket, sys, threading

local, host, remote = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])

def pump(src, dst):
    try:
        while True:
            chunk = src.recv(65536)
            if not chunk:
                break
            dst.sendall(chunk)
    except OSError:
        pass
    finally:
        for s in (src, dst):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("127.0.0.1", local))
listener.listen(4)
while True:
    client, _ = listener.accept()
    try:
        upstream = socket.create_connection((host, remote), timeout=10)
    except OSError:
        client.close()
        continue
    for a, b in ((client, upstream), (upstream, client)):
        threading.Thread(target=pump, args=(a, b), daemon=True).start()
'''

_SRUN = r'''#!@PYTHON@
"""srun: ignore the resource requests, set the job environment, run the command.

Everything before the command is a `--flag` of ours; what is left is the command,
and stdin is inherited straight through -- which is what carries the server's token.
"""
import os, subprocess, sys
argv = sys.argv[1:]
open(os.environ["FAKE_SLURM_LOG"], "a").write("srun " + " ".join(argv) + "\n")
command = [a for a in argv if not a.startswith("--")]
env = dict(os.environ)
env["SLURM_JOB_ID"] = "4242"
env["SLURM_JOB_NODELIST"] = "localhost"
sys.exit(subprocess.run(command, env=env).returncode)
'''

_SALLOC = r'''#!@PYTHON@
"""salloc --no-shell: create an allocation, print its id, return.

FAITHFUL ABOUT THE QUEUE, which is the thing worth standing in for here: the real
salloc names the job as soon as it is submitted and then does NOT return until the
allocation is granted -- for as long as the partition is full. FAKE_SALLOC_QUEUE
seconds of that is what makes a test able to tell "waited in the queue" apart from
"timed out", and the default of 0 keeps every other test instant.
"""
import os, sys, time
open(os.environ["FAKE_SLURM_LOG"], "a").write("salloc " + " ".join(sys.argv[1:]) + "\n")
print("salloc: Pending job allocation 4242", file=sys.stderr, flush=True)
time.sleep(float(os.environ.get("FAKE_SALLOC_QUEUE", 0)))
print("salloc: Granted job allocation 4242", file=sys.stderr, flush=True)
'''

_SQUEUE = r'''#!@PYTHON@
"""squeue: pending for the first call, then running on localhost -- and gone once
the job has been cancelled.

THE CANCELLED HALF MATTERS AS MUCH AS THE QUEUE HALF. The teardown does not trust
its own `scancel` to have worked; it asks Slurm afterwards. A fake that answered
RUNNING forever would make every clean release report a failure, and one that
answered nothing would make a failed release look clean.
"""
import os, sys
log = os.environ["FAKE_SLURM_LOG"]
argv = sys.argv[1:]
job = argv[argv.index("-j") + 1] if "-j" in argv else ""
if not job:
    # A listing (`squeue -u me -n name`), which the deploy makes to spot a job an
    # earlier session left behind. Answered from FAKE_STALE_JOBS, in the format it
    # asked for, and WITHOUT touching the call counter below: that counter is the
    # queue simulation for the job this session submits, and a listing is not a
    # question about it.
    fmt = argv[argv.index("-o") + 1] if "-o" in argv else "%i %T"
    for stale in os.environ.get("FAKE_STALE_JOBS", "").split():
        print(fmt.replace("%i", stale).replace("%T", "RUNNING"))
    sys.exit(0)
cancelled = set()
if os.path.exists(log + ".cancelled"):
    cancelled = set(open(log + ".cancelled").read().split())
if job and job in cancelled:
    sys.exit(0)                      # not in the queue: no line at all
path = log + ".squeue"
calls = 0
if os.path.exists(path):
    calls = int(open(path).read() or 0)
open(path, "w").write(str(calls + 1))
if calls < 1:
    print("PENDING||Priority")
else:
    print("RUNNING|localhost|None")
'''

_SCANCEL = r'''#!@PYTHON@
import os, sys
log = os.environ["FAKE_SLURM_LOG"]
open(log, "a").write("scancel " + " ".join(sys.argv[1:]) + "\n")
ids = [a for a in sys.argv[1:] if not a.startswith("-")]
open(log + ".cancelled", "a").write(" ".join(ids) + "\n")
'''


def build(directory):
    """Write the fake executables into `directory` and return its path."""
    os.makedirs(directory, exist_ok=True)
    for name, source in (("ssh", _SSH), ("proxy.py", _PROXY), ("srun", _SRUN),
                         ("salloc", _SALLOC), ("squeue", _SQUEUE),
                         ("scancel", _SCANCEL)):
        path = os.path.join(directory, name)
        with open(path, "w") as fh:
            # A plain placeholder, not str.format: these scripts contain braces
            # (dicts, sets, f-strings) and format() would read them as fields.
            fh.write(textwrap.dedent(source).replace("@PYTHON@", sys.executable))
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IRUSR)
    return directory
