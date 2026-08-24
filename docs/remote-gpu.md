# The remote-GPU demo, explained

How `mesomem_remote` works: 10,000 beads integrated on a Snellius A100 and drawn
on the laptop at 60 fps, with every slider still live. This is the companion to
[a100-plan.md](a100-plan.md) (which is the *plan* and its measurements) and
[snellius/README.md](snellius/README.md) (which is *how to run it*). This file is
the *why* -- every decision, in order, with the networking spelled out for someone
who does not spend their time in it. For the code itself -- the framing, the
codec, the threads, the socket options, line by line -- see
[remote-networking.md](remote-networking.md).

Built 2026-08-18. **The SSH and Slurm half has never run against the real
Snellius** -- only against a fake cluster (see [Testing](#8-testing-a-cluster-that-fits-on-one-machine)) --
because a real run needs a one-time code typed by hand. Everything else is
verified end to end.

---

## 1. The idea, in one paragraph

The app is already two halves that barely talk: LAMMPS integrates, and everything
else reads arrays out of it and draws them. Every readout already hands back a
*copy* of an id-ordered array (that was forced by the threading rule in
[stepper.py](../lammps_live/stepper.py)), and every slider already produces a
plain LAMMPS command *string*. So the seam was already there. All the remote demo
does is put a socket in it: the arrays go one way, the command strings go the
other, and nothing above `MDSystem3D` -- not the control loop, not the renderer --
knows which side of the socket its simulation is on.

```
   YOUR LAPTOP                          SNELLIUS
   ===========                          ========

   lammps-live                          login node        gpu node (gcn12)
   +----------------------+             +--------+        +---------------------+
   | RemoteSystem         |             |        |        | server              |
   |  . decodes and draws |             |  ssh   |        |  . LAMMPS on the    |
   |  . runs the analysis |             |        |        |    A100             |
   |  . owns the sliders  |             |        |        |  . 20 steps / frame |
   +----------------------+             +--------+        +---------------------+
        127.0.0.1:5723 <---- ONE TUNNEL, ONE LOGIN ----> 127.0.0.1:5723
                              (the login node relays
                               a stream it cannot read)

     <---- frames     100 kB each, 60 per second      (server to client)
     ----> commands   ~50 bytes, only when you move something
```

That asymmetry -- megabytes per second one way, a few bytes the other -- is why all
the care goes into the frame encoding and none into the control channel.

### Where the code is

| file | what it is |
|---|---|
| `lammps_live/playgrounds/mesomem_remote.py` | the demo itself: 10k beads, and where it runs |
| `lammps_live/playgrounds/mesomem_polymer.py` | the second one on the same target: a vesicle with a ring-polymer melt in it |
| `lammps_live/remote/protocol.py` | the wire: message framing and the frame codec |
| `lammps_live/remote/server.py` | the headless half -- runs on the GPU node |
| `lammps_live/remote/client.py` | `RemoteSystem`: an `MDSystem3D` fed by a socket |
| `lammps_live/remote/session.py` | SSH + Slurm: allocate, deploy, launch, tunnel, tear down |
| `lammps_live/remote/probe.py` | one command: can this node run the server? |
| `lammps_live/remote/hosts.py` | what differs about the cluster's LAMMPS build |
| `lammps_live/remote/target.py` | the login, the queue request, the ports |
| `lammps_live/ui/remote_panel.py` | the connect card (**N**) |
| `tests/fake_cluster/` | a fake `ssh`/`salloc`/`squeue`/`scancel` to test all of it |

Each of those files opens with a docstring covering its own decisions; this
document is the tour.

---

## 2. The networking, from the bottom

Skip this section if `-L`, `ControlMaster` and `SSH_ASKPASS` are already familiar.

**A socket is a phone line between two programs.** One program *listens* on a
**port** (a number, like an extension: ours is 5723) and the other *connects* to
it. Once connected, either side can write bytes and the other reads them, in order,
until somebody hangs up. That is the whole model. TCP guarantees the bytes arrive
in order and none are lost -- which is why the wire format below can be a plain
sequence of messages with no sequence numbers or retries in it.

**`localhost` / `127.0.0.1` means "this machine".** A program listening on
127.0.0.1:5723 can only be reached from the same computer.

**The problem: the GPU node is not on the internet.** `gcn12` is on Snellius'
internal network. Nothing on your laptop can open a socket to it -- there is no
route, and there is a firewall in the way, and both of those are correct and
should stay that way.

**The fix: an SSH tunnel.** SSH is already allowed to reach the login node, and
the login node *can* reach `gcn12`. So we ask SSH to carry the socket for us. There
are two ways to do that, and the difference between them is **where the SSH session
ends**. It matters more than it looks.

### One hop -- the obvious way, and not what we use

```
ssh -N -L 5723:gcn12:5723  stefanh@snellius.surf.nl
          ^^^^ ^^^^^ ^^^^
           |    |     |
           |    |     +-- ... to port 5723 on that machine
           |    +------- ... forward it to gcn12 (as seen from the login node)
           +------------ anything that connects to 127.0.0.1:5723 on my laptop ...
```

Your session **terminates on the login node**. Its sshd decrypts your bytes, then
opens a *separate, plain TCP* connection onward to `gcn12:5723`, and re-encrypts on
the way back. Two consequences:

- the frames cross the cluster's internal network **in clear text**, and the login
  node sees the plaintext;
- the server must listen on the node's **external interface** (`--bind 0.0.0.0`),
  because a forward arriving from the login node cannot reach the node's own
  loopback. That port is then reachable by anything on the internal network.

### Two hops -- the default

```
ssh -N -J stefanh@snellius.surf.nl  stefanh@gcn12  -L 5723:localhost:5723
          ^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^
          jump through here         session ENDS      ... and localhost
          (a pipe, not an           here, on the      means the NODE
          endpoint)                 compute node      itself
```

Your session **terminates on `gcn12`**. The login node relays an opaque stream it
cannot read -- a pipe, not a man in the middle -- and because the far end of the
forward is `localhost` *as seen from the node*, the server can bind
**`127.0.0.1`**. What that buys:

| | one hop | two hops |
|---|---|---|
| where the session ends | login node | **the compute node** |
| frames on the internal network | plain TCP | **encrypted end to end** |
| what the login node can read | the plaintext | **nothing** |
| server listens on | `0.0.0.0` | **`127.0.0.1`** |
| who can reach the port | anything on the cluster network (token-protected) | **only processes on that node** |
| the token is | the only thing in the way | defence in depth |

The last row is the real prize. The one thing about the one-hop design that was
genuinely uncomfortable is that `--bind 0.0.0.0` put a channel that can re-issue
LAMMPS commands on a shared network, with nothing but a shared secret in front of
it. Two hops removes the exposure rather than guarding it.

**And it costs no extra login.** `-J` would normally authenticate twice -- once to
the login node for the jump, once to `gcn12` -- which with a one-time code would
mean typing a second one, at the worst possible moment (the node's name only exists
*after* the queue wait, so the prompt would arrive minutes in). Both halves of that
are avoided:

- the jump hop rides the master connection that is already authenticated, as a
  stdio pipe: `-o ProxyCommand="ssh -S <ctl> -W %h:%p <login>"`;
- the node hop, on Snellius, **does not prompt at all** (tested: `ssh -J login gcn12
  hostname` while an allocation is running answers straight away, because Slurm's
  PAM stack lets the owner of a running job in).

So the whole flow still costs exactly one one-time code.

### Two things two hops does *not* fix

- **A shared node is still shared.** A `gpu_a100` node holds four A100s, so
  `--gpus=1` means up to three other jobs on the same machine -- and `127.0.0.1` is
  perfectly reachable by their processes. Loopback shuts out the rest of the
  cluster, not your node-mates. **The token stays**, for exactly this reason.
  (Allocate the whole node and loopback does become airtight.)
- **It is a site policy, not a law.** `ssh` to a compute node needs sshd running
  there and a PAM stack that admits the owner of a running job. Where that is not
  allowed, the failure is a bare `Permission denied` that reads like an auth
  problem -- which is why the one-hop form is kept, as
  `RemoteTarget(tunnel="forward")`, and why `server_bind` is derived from the tunnel
  mode rather than configured separately. The two can never disagree.

Performance is not a consideration either way: two hops double-encrypt on the
laptop (inner session inside the outer jump), and AES-NI runs at gigabytes per
second against the 6 MB/s this needs -- 60 MB/s even at 100k beads. The login node
does the same amount of work in both designs.

Everything else below is bookkeeping around that one line.

### 2.1 One login, not five

The flow needs the cluster six times: deploy, probe, allocate, wait for the node,
launch, tunnel. Six separate `ssh` commands means six authentications, which with a
one-time code means typing six codes. Unusable.

SSH has the answer built in: **ControlMaster**. The first connection is told to
open a little **control socket** -- an ordinary file on the laptop, in a private
temp directory -- and every later `ssh` command is pointed at that file instead of
at the network. Those later commands do not authenticate at all; they open a new
*channel* inside the connection that is already up.

```
ssh -M -S /tmp/.../ctl -N -T  stefanh@snellius.surf.nl    the master: authenticates
                                                          once, then just sits there
ssh -S /tmp/.../ctl  stefanh@...  bash -lc 'squeue ...'   instant, no prompt
ssh -o ProxyCommand='ssh -S /tmp/.../ctl -W %h:%p ...' \
    -L 5723:localhost:5723  stefanh@gcn12                 the tunnel: jumps through
                                                          the master, ends on the node
ssh -S /tmp/.../ctl -O exit  stefanh@...                  hangs up everything
```

The flags, since they all earn their place:

| flag | why |
|---|---|
| `-M` | be the master (accept other commands through the control socket) |
| `-S <path>` | where the control socket lives -- a 0700 temp dir, removed on exit |
| `-N` | do not run a remote command; this connection exists only to be shared |
| `-T` | no terminal; nothing is being typed into it |
| `-o ControlPersist=no` | the master dies with this process. A crashed app must not leave an authenticated connection to the cluster lying around |
| `-o ServerAliveInterval=30` | notice a dead network in 2 minutes instead of never |
| `-o StrictHostKeyChecking=accept-new` | a first-time host key is accepted rather than turning into a prompt nobody expected. A *changed* key is still refused, which is the case that matters |
| `-O forward` / `-O cancel` / `-O exit` / `-O check` | operate on the existing master: add a forward (one-hop mode only), remove it, hang up, or ask whether it is up yet |
| `-W %h:%p` | on the *jump* hop: turn that ssh into a plain stdio pipe to `host:port`, which is what lets the two-hop tunnel reuse the authenticated master |
| `-o ExitOnForwardFailure=yes` | on the tunnel: fail loudly if the local port cannot be bound, instead of sitting there carrying nothing |
| `-o UserKnownHostsFile=<session temp>` | the node's host key is pinned for the session but not written into the real `known_hosts`: compute nodes get reimaged, and pinning them there turns a routine reallocation into a hard failure weeks later. What this gives up on is a MITM by the login node, which already holds the user's home directory and Slurm credentials |

### The trap in that, which cost the first real run

**ssh does not pass an argv array to the far side.** Everything after the
destination is joined with single spaces into one string, which the remote shell
then parses itself. So this, which looks obviously correct:

```python
subprocess.run(["ssh", "-S", ctl, dest, "bash", "-lc", "mkdir -p ~/d && tar xzf -"])
```

arrives on the cluster as

```
bash -lc mkdir -p ~/d && tar xzf -
```

and `bash -c` takes **exactly one** argument. So the far side ran a bare `mkdir`
(with `-p` as its `$0`) and answered `mkdir: missing operand`, the `&&` short-
circuited, and the tar never happened. That is the first thing the real Snellius
run reported.

The fix is to quote the command into a single shell word -- there is no
argv-preserving mode to switch on -- which is what `_login_shell()` does, in one
place, for every remote command:

```python
return "bash -lc " + shlex.quote(command)
```

The reason the test suite did not catch it is worth more than the bug: the fake
`ssh` in `tests/fake_cluster/` took `argv[-1]` as the command, so it *preserved*
the argv that real ssh flattens. **A stand-in that is more forgiving than the real
thing hides exactly the bugs it exists to catch.** It now joins arguments the way
ssh does, and there is a unit test asserting that a command with `&&` in it
survives being flattened.

`-O check` is how the code knows the login succeeded, and there is no better
signal: the `ssh -M` process starts *before* it has authenticated and keeps running
*after*, so its being alive tells you nothing. The flow polls `-O check` until it
answers, and watches for the process exiting (which is what a wrong password looks
like).

### 2.2 The one-time code, and why it cannot just be piped in

The obvious idea -- write the password into ssh's standard input -- does not work,
by design. SSH deliberately reads secrets from the *terminal*, not from a pipe,
precisely so that a script cannot feed it one silently. No amount of arranging
changes that.

What SSH *will* do is run a helper program and use whatever the helper prints. That
is the `SSH_ASKPASS` mechanism, normally used by desktop environments to pop up a
password box. So:

1. On connect, the app writes a tiny helper script into a private temp directory.
2. It starts `ssh` with `SSH_ASKPASS=<that script>` and
   `SSH_ASKPASS_REQUIRE=force` (the "force" is essential: without it SSH prefers a
   terminal, and the app hasn't got one it can prompt on).
3. SSH needs an answer, so it runs the helper, passing the question as its first
   argument -- `Password:`, `Verification code:`, a host-key fingerprint, a key
   passphrase, whatever this login happens to ask.
4. The helper connects to a **unix socket** (a socket that is a file on disk rather
   than a network address, reachable only by this user) and passes the question
   through to the running app.
5. The app shows the question **verbatim** in the connect panel. You type; the
   answer goes back down the unix socket; the helper prints it; SSH consumes it.

```
   ssh  --runs-->  askpass.py  --unix socket-->  the app  --shows-->  you
     ^                  |                                               |
     |                  +--------- the answer <-------- you type --------+
     +-- prints it -----+
```

Three things this buys, none of them accidental:

- **The code never touches disk, a command line, or an environment variable.** Only
  the memory of two processes and a socket in a directory nobody else can enter.
  (On a shared cluster, anything in a command line is readable by every other user
  via `ps` -- see [2.3](#23-the-token).)
- **The app does not need to know what your login asks for**, or in what order, or
  how many times. It relays questions. If SURF adds a step, or you switch to a key
  with a passphrase, or the host key changes, the panel shows that instead and the
  code needs no change.
- **Cancelling works.** The helper exiting without printing is how SSH is told
  "the user dismissed the prompt", so the Cancel button is a real cancel rather
  than a hang.

### 2.3 The token

With the two-hop tunnel the server binds `127.0.0.1`, so the port does not exist as
far as the rest of the cluster is concerned. The token is still there, for two
reasons that are not paranoia:

- **A GPU node is shared.** Four A100s per node, and `--gpus=1` means up to three
  other jobs on the same machine. Their processes can reach that node's loopback
  perfectly well. Loopback shuts out the cluster, not your node-mates.
- **The one-hop fallback exists**, and there the port really is on the internal
  network. A secret that is only present in one of two modes is a secret somebody
  will eventually forget to turn on.

The control channel can re-issue LAMMPS commands, so it is not something to leave
open in either mode. Every connection therefore has to present a **24-byte random
secret** (48 hex characters), generated fresh per session, and:

- it is sent to the server **on its standard input**, not as `--token abc` on the
  command line, because on a shared cluster every user can read every other user's
  command lines -- and node-mates are exactly the people loopback does not keep out.
  `srun` forwards stdin to the task, so the secret travels inside the SSH connection
  and is never visible in `ps`;
- the server compares it with `hmac.compare_digest`, not `==`. A plain string
  comparison stops at the first wrong byte, and something that can time the reply
  can learn the secret one byte at a time. This is cheap paranoia -- the comparison
  costs nothing -- but it is the correct habit;
- the server **refuses to start without one**. There is no "open" mode to
  accidentally leave on.

---

## 3. Slurm: getting the GPU and giving it back

You would normally do this by hand:

```
srun --partition=gpu_a100 --gpus=1 --ntasks=1 --cpus-per-task=18 --pty bash
```

which drops you on a node inside a shell. That is exactly what a GUI cannot use:
the allocation lives only as long as that interactive shell, so everything -- the
node name, the job's lifetime, the teardown -- would hang off a pipe staying alive.
Pipes die.

**`salloc --no-shell` is the trick.** It creates the allocation, prints
`salloc: Granted job allocation 9182733`, and *returns*. The allocation then exists
on its own, independent of any connection:

```
salloc --no-shell --partition=gpu_a100 --gpus=1 --ntasks=1 \
       --cpus-per-task=18 --time=01:00:00 --job-name=mesomem-live
squeue -h -j 9182733 -o '%T|%N|%r'      → RUNNING|gcn12|None
srun --jobid=9182733 --unbuffered ... python -m lammps_live.remote.server ...
scancel 9182733
```

`squeue` is polled until the state is `RUNNING` and a node name has appeared;
while it is `PENDING` the reason (`Priority`, `Resources`, ...) goes straight into
the panel, so a queue wait looks like a queue wait rather than a hang. The `%T|%N|%r`
format is explicitly separated by `|` because the pretty column format is ambiguous
when a pending job has no node list to print.

**What `--no-shell` does not do is return before the allocation is granted.** On a
full `gpu_a100` it blocks for however long the queue is, which found the app as
`remote command timed out after 120s` -- a connect failing for the most ordinary
reason there is. So `salloc` is started as a process the session *watches* rather
than a command it waits on:

```
salloc: Pending job allocation 9182733     ← the id, at submission
salloc: job 9182733 queued and waiting for resources
                                           ← polling starts here, not after this
salloc: Granted job allocation 9182733
```

Slurm names the job when it is *submitted*, so the id is read off the `Pending`
line and the `squeue` loop above takes over immediately -- with the state, the
reason, the minutes so far, and a Cancel that works. The wait is bounded by
`RemoteTarget.queue_wait`, an hour by default (`--time` bounds the allocation once
it starts; `queue_wait` bounds getting one at all), and the poll eases from every
3 s to every 15 s after the first minute, because a busy queue is answered by
waiting rather than by asking more often. Cancelling kills the `salloc`, which gives
up the place in the queue -- otherwise the request would be granted later and hold a
GPU with nobody connected to it.

### The three ways the allocation is released

An A100 held by a forgotten allocation is the one failure mode with a bill attached,
so this is deliberately over-engineered:

| when | mechanism |
|---|---|
| you close the window, or press Disconnect | the app runs `scancel` (`RemotePanel.release`) |
| the app dies without running anything -- crash, `kill -9`, laptop lid | the *server* runs `scancel $SLURM_JOB_ID` when it exits, and it exits by itself after `--exit-when-idle` (default **15 minutes** with no client) |
| everything above fails | Slurm's own `--time` (default **1 hour**) |

**The first row is the one that had holes in it**, found after leftover jobs turned
up in `squeue` long after the app was gone. Four of them, and all four came from the
same assumption -- that the shutdown path would be reached, in one piece, with the
SSH connection it started with:

| the hole | what now closes it |
|---|---|
| the top-level `finally` was five bare statements, and `scancel` was the fourth. Anything raising before it -- the stepper re-raising the error that ended the run, a joystick whose device had gone -- skipped it | `App._shutdown` guards each step separately and releases FIRST. A second Ctrl-C on a teardown that looks slow (it is running an ssh) no longer aborts it either |
| a `kill`, a closed terminal or a logout ends the process without unwinding anything, so no `finally` runs at all | SIGTERM/SIGHUP/SIGINT are caught and raised as `SystemExit`, which unwinds through that `finally`. Plus an `atexit` hook. SIGKILL still cannot be caught -- that one is what the middle row above is for |
| the `scancel` was skipped in silence whenever the SSH control socket had gone: a second teardown (the first removed it), a connect thread that got its job id just after the window closed, a master that died while the allocation lived on | `RemoteSession._release_job` tries the master, then a connection of its own (`BatchMode=yes`, so it fails in seconds rather than stopping the shutdown on a prompt), and asks `squeue` afterwards whether it actually took. A job it cannot confirm is kept, not forgotten, and said so in the log |
| Disconnect during the probe -- one ssh that can sit for a minute -- ran a whole teardown that correctly found nothing to cancel, and the flow then walked on into `salloc` and asked for a GPU nothing was left to release | `_allocate` checks `_cancel` for itself. It is the one step that does not reach the cluster through `_remote`, which had that check all along |

**Firing a `scancel` is not the same as the job having stopped**, which is why the
release ends with a `squeue`. `CANCELLED` and `COMPLETING` both appear for a few
seconds after a successful one and must not be read as failures; anything still
`PENDING` or `RUNNING` is a real one, and gets said in the log the panel copies
rather than swallowed.

**Switching playground does NOT release it** (it used to). Going off to show the
membrane patch and coming back is a normal thing to do mid-talk, and it should not
cost another queue wait: the session, the tunnel and the server all stay up, the
server holds its simulation between clients, and coming back is one fresh socket
through the same tunnel (`RemoteSession.reopen_link`, `RemotePanel._resume`). What
protects the A100 while nobody is looking at it is the middle row above -- the
server's own idle timeout, which is exactly the mechanism designed for "the client
is gone and may not come back". The panel says the GPU is still held, in amber,
while another playground is on screen.

The middle one is the interesting one: the remote command is
`... python -m ...server ...; rc=$?; scancel $SLURM_JOB_ID; exit $rc`, so the job cancels
itself no matter *how* the server ends. Nothing depends on the laptop still being
alive.

---

## 4. Splitting the app in two

### One interface, two implementations

`registry.build()` looks at the playground: if it declares a `remote=` target it
returns a `RemoteSystem` instead of a `PlaygroundSystem`. Both satisfy
`MDSystem3D`, so `app.py` and `renderer.py` are untouched by any of this. The
substitutions:

| `PlaygroundSystem` (local) | `RemoteSystem` (client) |
|---|---|
| `step(n)` runs `lmp.command("run 20")` | `step(n)` waits for the next frame off the socket |
| `set_extra_param` issues `pair_coeff ...` | sends `{"t":"set","key":"k_tilt","value":33.0}` |
| `frame_state()` gathers LAMMPS arrays | decodes the arrays out of the last message |
| `reset()` rebuilds LAMMPS here | sends `{"t":"reset"}`; the server rebuilds there |

Reset is the one of those with a wait in the middle of it: the rebuild happens on
the far side, during which no frames come back at all. It used to be 44 seconds for
50,000 beads — essentially all of it `create_atoms random ... overlap`, LAMMPS
placing particles on the host and rejecting the ones that land too close — and it is
half a second now that the placement is a bulk numpy pass (`RandomFill.build`). A
scenario with a relaxation in it still pays for the relaxation. Two things follow
from there being a wait at all, and both were found the hard way. The client latches `_resetting` and the HUD says *rebuilding
from a fresh state* until a frame from the new run arrives (the server restarts its
sequence at 0, which is how that is recognised) -- otherwise the picture sits on the
last frame of the old run, which is exactly what a Reset that did nothing looks
like. And `App._reset_simulation` waits for the simulation thread first: Reset
arrives from the event handler, which runs *between* frames, which is precisely when
a step is in flight -- so without the wait it replaces the analysis and clears the
smoother while the stepper thread is using them.

The server, meanwhile, **is** a `PlaygroundSystem` -- the same class, built from the
same playground file. That is the decision the whole design rests on: there is one
definition of what this demo is, and both ends read it. The alternative (a
hand-written LAMMPS input deck on the cluster, like the benchmark decks in
`docs/snellius/`) means two definitions that have to be kept in step by hand, and
they *will* drift.

### Who does what

| | server (A100) | client (laptop) |
|---|---|---|
| integration, thermostat, housekeeping | ✔ | |
| `Analysis`: pair list, energy panels, observables | | ✔ |
| RDF, trajectory smoothing, camera, rendering | | ✔ |
| Play / Pause / Reset, sliders | applies them | owns them |

**Why the analysis runs on the client**, when the server is the one with 18 cores:
it is single-threaded Python either way, so putting it on the server would throttle
the integrator it is supposed to be observing -- and the machine that draws the
panels is the machine with cycles going spare, since all it otherwise does is issue
GPU calls. The wire carries state, not conclusions. (`Analysis` grew an `enabled`
flag for this; the server constructs it switched off.)

### Where the waiting hides

`step()` is called on the **stepper thread** -- the worker that already exists so
that LAMMPS and pygame overlap (see [stepper.py](../lammps_live/stepper.py)). So
the network wait *and* the analysis that follows it both happen while the previous
frame is being drawn. The frame costs `max(analysis, render)` rather than the sum.
Same trick as the local demo, used for a different expensive thing.

### Frames are dropped, never queued

The reader thread keeps only the **newest** frame; an unread one is discarded and
counted. If the laptop hitches -- a window resize, a slow analysis frame -- the next
`step()` picks up where the simulation actually *is*, not where it was three frames
ago. A queue would trade latency for smoothness, and over a link the renderer
cannot flow-control, the latency would only ever grow. The dropped count is on the
HUD so it is visible rather than mysterious.

### Two states worth naming

- **Disconnected is normal.** Selecting the playground builds a `RemoteSystem` with
  no connection: empty scene, an explanatory HUD, every readout safe. That is what
  makes the connect panel possible *inside* the running app, and what makes the
  whole client testable with no cluster. (It also found a renderer bug -- see
  [§9](#9-two-real-bugs-fell-out).)
- **Paused is the server's business too.** `set_playing` is pushed down into the
  system, so Pause stops the *far end* integrating. Otherwise an A100 would keep
  computing frames for a socket nobody is reading -- the most expensive no-op
  available.

---

## 5. The wire format

### The frame budget

Positions and directors as plain `float32` is 24 bytes per bead: at 10k and 60 fps
that is 14 MB/s (115 Mbit/s), and at 100k it is 144 MB/s. So the frames are
quantised, to **10 bytes per bead**:

| field | encoding | bytes |
|---|---|---|
| position | 3 × 12 bits across the cell, packed in pairs | 4.5 |
| director | 2 × uint8, octahedral | 2 |
| per-bead energy | 1 × uint8 over the colour range | 1, and only on request |

At 10k that is **65 kB/frame, 3.9 MB/s, 31 Mbit/s** -- comfortable on any real
link. The bigger lever is the frame RATE: the default wire is 20 fps, not 60,
which costs another factor of three and nothing in simulation speed (the server
takes a proportionally longer stride) or in smoothness (the client fills in
between frames). See remote-networking.md §8.

### Quantising positions

A `uint16` holds 0...65535. The cell is 37.4σ across, so we store
`round((x - lo) / (hi - lo) × 65535)` and multiply back on the other side. The
error is at most half a step: **0.0003σ**, against a bead radius of 0.5σ and a
thermal rattle three orders of magnitude larger. Invisible, and provably so
(`tests/test_remote_protocol.py` asserts it).

Two details that are not obvious:

- **The range is the cell plus 1σ of padding.** LAMMPS remaps atoms into the
  periodic cell when it rebuilds neighbour lists, not on every step, so a
  coordinate can legitimately sit slightly outside the box it belongs to. Padding
  costs 3% of the resolution and avoids a pile-up of beads pinned to the wall.
- **Out-of-range values are clamped, never wrapped.** A wrap would put a bead on
  the *opposite side of the cell* -- the one error that could never be mistaken for
  noise.

### Quantising directors: the octahedral map

A director is a unit vector: three numbers, but only two degrees of freedom, so
storing three components wastes a third of the bytes. The standard fix, and the one
used here:

1. Squash the sphere onto the surface of an octahedron: divide by
   `|x| + |y| + |z|`, so every vector lands on `|x| + |y| + |z| = 1`.
2. The top half of that octahedron is already a flat square in `(x, y)`. Fold the
   bottom half *outward* into the surrounding square by reflecting it across the
   diagonals.
3. Now the whole sphere is a unit square, and a square is two numbers. Store them.

It is continuous, nearly area-preserving (so no direction is much worse encoded
than any other), and about six vector operations each way.

**16 bits per component, not 8** -- 10 bytes/bead rather than 8, one deliberate step
away from the plan's table. Eight bits gives 0.87° of worst-case error, invisible in
the shading, and the plan therefore recommended it. But the client does not only
*draw* those directors: it **measures** `nematic_S` from them, and that is the
number the k_tilt transition shows up in. An order parameter should not carry a
codec's error. Sixteen bits costs two bytes and takes the worst case to
**0.0037°** (mean 0.0013°).

### Energies on request

Per-bead potential energy is a tenth of the bytes and the server has to gather it
from a per-atom compute -- but it is only *used* when the bead colouring is set to
ENERGY. So the client asks for it based on whether `get_bead_energies()` is
actually being called, with a couple of frames of hysteresis (an edge-triggered
version was fragile: how many times `step()` runs per drawn frame is not something
the client gets to assume). Toggle the colouring and the request follows a frame or
two later; leave it off and the bytes are never sent.

The third colouring -- CLUSTER, which paints each connected aggregate its own
colour -- asks for nothing at all. It is a fact about geometry the client is
already holding, so the client runs the labelling itself
(`lammps_live/playground/clustering.py`) on the positions off the wire. That is
the right side of the link for it: the far end is the scarce resource, and the
answer would cost as many bytes to send as it costs to compute. What it does cost
is *here*, and it is the one readout whose price scales with the bead count this
playground exists to show off -- ~90 ms at 50k with a thousand aggregates in the
cell, which is why the labelling is recomputed every 33 frames rather than every
frame (aggregates coarsen over seconds; there is nothing in a per-frame labelling
that was not in the last one).

### What is deliberately not built

The plan's table continues below the codec's 6.5 B/bead using temporal deltas
and zlib. Not
implemented, on purpose: both need a **stateful decoder**, where a frame only
decodes if the previous one arrived -- and this client *drops frames by design*.
Every frame here stands alone. At 10k the link is not the bottleneck; when it is,
[§11](#11-what-to-do-next-to-100k-and-beyond) says what to add and in what order.

### The message shape

Two length-prefixed blocks: a JSON header and, optionally, raw array bytes.

```
[ uint32 json_len ][ uint32 payload_len ][ JSON ][ raw arrays ]
```

The header lists what is in the payload -- `[["pos","<u2",[10000,3]], ...]` -- so the
decoder needs no built-in knowledge of which fields a given frame chose to carry,
and a new codec is a new name in one dict. Control messages are the same envelope
with an empty payload: `{"t":"set","key":"wc","value":1.8}`, `{"t":"play"}`,
`{"t":"reset"}`, `{"t":"ping","id":42}`.

Two small things worth knowing if you read the code:

- **`recv` is allowed to return fewer bytes than you asked for.** This is the
  classic socket mistake, and it only shows up under load -- which is exactly when
  the frames are biggest. `_recv_exactly` loops until the buffer is full.
- **`TCP_NODELAY` is set.** Otherwise the OS holds a small write for a round trip
  hoping to combine it with the next one, which on a slider drag is felt directly.

### Clamped values, not slider values

`set_extra_param` sends the value **after** the force field's declared clamp -- `wc`
is capped at `rc` -- because the client computes the energy panels locally and the
far end must be running the same coefficients the local decomposition is using.
Sending the raw slider value would make the panel quietly describe a different
simulation than the one on screen.

---

## 6. The cluster's LAMMPS is not our LAMMPS

Four differences, all collected in `remote/hosts.py` as a `HostProfile` rather
than scattered through the force field:

| | local (pip wheel) | Snellius (Kokkos build) |
|---|---|---|
| pair style | compiled on demand, `plugin load mesomem.dylib` | compiled in, plus `mesomem/kk` |
| atom style | `hybrid sphere dipole` | `dipole_sphere_angle` |
| mass | per type: `mass 1 1.0` | per atom: `set type 1 mass 1.0` |
| `pair_coeff` | 9 numeric values (patched: `splay_symmetry`) | 8 or 9 -- **probed**, not assumed |
| command line | — | `-k on g 1 -sf kk` |

The mass row is the one that cost a real run
([§9](#9-seven-real-bugs-fell-out)): the `rmass` field that makes
`dipole_sphere_angle` a finite-size-particle style is exactly what makes the
per-type `mass` command illegal on it. Same number, two spellings, and which one is
legal is a property of the build, not of the physics -- so the profile rewrites the
command and the force field goes on saying "each bead weighs one".

A profile **adapts a force-field instance in place** (`ff.atom_style = ...`), which is
safe because `PlaygroundSystem` builds a fresh force field per system, so the
instance attribute shadows the class attribute for that one object and nothing else
in the process is affected. The alternative -- registering a second `MesoMem`
subclass under another name -- would have to be named in the playground file, which
would put a cluster's build details into a file that is meant to describe physics,
and would make the local loopback test unrunnable.

Two traps, both found the hard way. Truncating `pair_coeff` has to wrap **both**
`pair_commands` and `coeff_commands`. The base class defines the second as the
first minus its `pair_style` line, but a force field may write it out itself, and
MesoMem does. Wrapping only one truncates the initial build and leaves every
slider re-issuing the untruncated form -- a bug that works at startup and fails the
first time you touch `k_tilt`.

And the probe has to issue **the commands the server will issue**, not equivalent
ones. Its two-atom snippet reached for `set group all density 1.0` where the force
field emits `mass 1 1.0`; both give a bead a mass, only one of them was going to be
rejected on the node, and the probe was testing the wrong one. It now issues the
profile's own mass command and reports which spelling the build accepted.

### The probe, and why it runs twice

`python -m lammps_live.remote.probe` answers, in one command, every question that
has bitten this port already. It runs at **two levels, on two machines**, and the
reason is the single most confusing error this port can produce:

```
OSError: libcuda.so.1: cannot open shared object file: No such file or directory
```

A Kokkos/CUDA `liblammps.so` **links against the NVIDIA driver**, and the driver is
installed on GPU nodes and nowhere else. So `import lammps` succeeds on a login node
(it is pure Python) while `lammps(...)` -- the `CDLL` that loads the library -- fails
there, with an error that looks exactly like a broken build and is not one. The
original design probed only the login node, which for this build could never have
worked. Now:

| level | where | answers |
|---|---|---|
| `--level light` | login node, before `salloc` | which interpreter, numpy, and whether the `lammps` module and `liblammps.so` are where they should be. Free, and catches the most common failures |
| `--level full` | the GPU node, inside the allocation | everything that needs the library OPEN: liblammps version, which styles are registered, and how many values `pair_coeff` takes |

The second one is not optional politeness -- the `pair_coeff` arity is something the
server *needs*, because it decides the commands every slider issues. It could only
ever have been answered on the node.

What each level checks:

- **Does `import lammps` work?** This is the fatal one. The `lmp` binary alone
  cannot be driven frame by frame, so without the Python module there is no demo.
  Fix: rebuild with `-DBUILD_SHARED_LIBS=yes -DPKG_PYTHON=yes`, then
  `make install-python`.
- **Is `mesomem` registered? `mesomem/kk`?** A missing Kokkos variant means `-sf kk`
  silently falls back to the host style: the run works and is slow, which is the
  worst thing to have to diagnose from a frame rate.
- **`nve/sphere/kk`?** This is `snellius/README.md` point 2 -- without it the
  integrator copies positions, dipoles and torques back to the host every step and
  the pair kernel's gain goes with it. Reported as a warning, not a failure.
- **8 or 9 `pair_coeff` values?** Answered by handing the pair style nine on two
  atoms and watching for an error. The answer is passed to the server, so the
  coefficients the client sends always match what that build accepts.
- **numpy present? scipy?** numpy is required; scipy is *not*, because the server
  runs no analysis. Worth reporting, since a cluster python with numpy and no scipy
  is common and would otherwise look like a risk.

**"scipy is not needed" is a constraint on this codebase, not just an observation
about the probe.** Everything the server touches -- a scenario's `build`, the force
field's deck, the per-frame read-out -- has to run on numpy alone; scipy belongs to
the client, where the analysis is. It is an easy invariant to break silently,
because locally scipy is always there: `mesomem_polymer` broke it with one
`cKDTree` inside `icosphere_spacing`, and the bill came due as a
`ModuleNotFoundError` *after* Slurm had allocated an A100 -- a server that says
LISTENING, dies on the first build, and takes the allocation down with it. The
guard is `test_the_build_runs_with_no_scipy_installed`, which builds the scenario
with the import blocked.

**Every LAMMPS question is asked in a subprocess.** A LAMMPS error normally raises a
Python exception -- but a build without exceptions enabled calls `MPI_Abort`, which
takes the whole process with it, and a probe that can kill the thing it is probing
is useless.

The connect flow runs the light level first and **refuses to allocate a GPU** for an
interpreter that cannot even import numpy; then, once the allocation exists, runs the
full level on the node with one `srun` before starting the server. A failure there
costs a queue slot, which is unavoidable and is why the cheap checks come first.

---

## 7. The connect panel

A modal card over the sim view, shown whenever the remote playground is selected and
not streaming. **N** toggles it.

It exists in the app, rather than as a script you run first, because the demo is a
thing you stand in front of: allocating from a terminal, then starting the app, then
reconnecting by hand when the tunnel drops, is three ways for a demo to fail in
public. And the *teardown* matters more than the setup -- the thing that ends the
session has to be the thing you close.

Details that took a moment's thought:

- **Nothing blocks.** Every step runs on the session's worker thread; the panel
  reads its state once per frame. The app keeps drawing at 60 fps through an SSH
  login, a queue wait and a LAMMPS build on the far side.
- **It is modal while a prompt is pending**, and only then. A one-time code is
  digits, and digits are the app's playground shortcuts -- without this, typing
  `424242` would switch playgrounds four times mid-login.
- **The prompt is shown verbatim**, because only the thing that asked knows what it
  is asking for.
- **The card is sized from its content**, and the far side's own output is
  interleaved with this end's steps in one log -- which is what makes a failure
  diagnosable without going to look for a log file on a node that no longer exists.
- **A link that dies on its own reopens the panel** with the reason, instead of
  leaving a frozen picture and no explanation.
- **The server says it is BUILDING before it builds.** The welcome cannot be sent
  until LAMMPS' setup is done -- tens of seconds at 10k beads -- and the client sits
  in a blocking read for all of it. It used to give up at 15 s, drop the socket and
  retry, which made the server throw the half-built simulation away and start over,
  so the retry could not succeed either; the symptom was `connect attempt 1:
  handshake failed: timed out` followed by a session that eventually worked for no
  visible reason. The `building` message resets that clock to a much longer one
  (`FrameLink.BUILD_TIMEOUT`) and puts the wait in the panel's log, and the
  distinction it keeps is the one that matters: a SILENT server is broken and should
  be given up on quickly, a server that has said it is working should not.
- **Switching playground and back resumes the session rather than replacing it**
  (`RemotePanel.attach_system`). Streaming -> reconnect over the tunnel and carry
  on; still connecting -> leave it working and hand the link over when it lands (a
  queue wait that is nearly done must not be cancelled by a keystroke); the job
  gone -> a new session and the card. A link that arrived *while the playground was
  off screen* is adopted rather than reconnected: the server serves one client at a
  time, so a second socket would sit in the backlog behind the one we already hold.
- **One allocation serves both remote playgrounds.** See 7.1.
- It is drawn *inside* `renderer.draw()`, because in GL mode every 2D surface has
  to be on `self.screen` before it is composited -- and because `draw()` is what
  flips the display.
- **C copies the whole report**, in every state, to the clipboard *and* to a file
  in the temp directory. See below.

### 7.1 One GPU, several demos

There are two remote playgrounds now -- `mesomem_remote` and `mesomem_polymer` --
and asking for a GPU twice in one talk is asking for the queue twice. So there is
one session for the whole app, not one per playground, and the split it rests on is:

| | |
|---|---|
| expensive, done once | the login and its one-time code, the deployed package, the probe, the **allocation**, the server process, the tunnel |
| cheap, done per demo | closing one `PlaygroundSystem` on the far side and building another |

Which gives three behaviours, and the third is the only one that costs anything:

- **`Tab` to the other remote playground.** Nothing happens on the cluster. The run
  you left is still integrating (or rather still held -- `serve_forever` stops
  integrating the moment nobody is connected), the job is still yours, and the card
  comes up saying so: *GPU held: job 4242 on gcn12, running mesomem_remote*.
- **`Tab` back.** One socket through the tunnel that never closed, and you are
  looking at the same box you left. This is `reopen_link`, and it predates the rest.
- **Connect on the other one.** The button says *Move GPU here*, because that is
  what it does. The client's `hello` now carries a `playground` field; a server
  holding a different one closes it -- freeing the LAMMPS instance and the GPU memory
  with it -- and builds the named one in its place
  (`FrameServer.switch_playground`). Same node, same port, same job id, no queue, no
  second code.

**The run you leave is parked, not thrown away.** This used to be the honest price
of one GPU serving two demos, and the reason moving it was a button press rather
than something `Tab` did behind your back: the coarsened box you had been watching
for four minutes was gone. It stopped being a price worth paying once building an
instance became cheap. `FrameServer._park_current` takes a snapshot -- positions,
velocities, directors, angular velocities, the cell, the simulated time and the
parameters, in stable **atom-id** order, because LAMMPS' local ordering is
emphatically not the same on the other side of a rebuild -- and `_resume_parked`
scatters it into the freshly built instance
(`PlaygroundSystem.snapshot_state` / `restore_state`). A parked 50k run is under
5 MB of arrays; four are kept, oldest evicted.

Two consequences of it being cheap:

- **The build that is about to be overwritten does not relax.** Passing
  `settle=False` builds the deck exactly as usual -- every fix installed, every
  command validated, `run 0` where the scenario asked for a relaxation -- and skips
  only the integration, because relaxing a configuration that is about to be
  scattered over is the most expensive pointless thing in the flow. Measured on
  `mesomem_polymer`: 7.2 s to build, 0.4 s to build and resume. The `run` commands
  are what is dropped and nothing else, because a settle list also *installs* fixes
  and on some scenarios one of them outlives the settle (the rod's barostat).
- **A snapshot that no longer fits is refused, and then the build is redone
  properly.** Different bead count means someone edited the playground between
  switches; scattering it in anyway would produce a scene that is neither run, and
  serving the no-settle build would mean serving the scenario's made-up starting
  geometry.

Three details worth knowing:

- **The switch runs on the session's worker thread** (`RemoteSession._run_switch`),
  because the far side's rebuild is LAMMPS' own setup on 50,000 beads. The state is
  `SWITCH` while it goes, the progress bar sits at 6/7 -- everything but the last
  step really is still done -- and the app keeps drawing.
- **A failed switch does not give the GPU back.** This is the one place that departs
  from the rest of `session.py`, where every failed step tears down. The distinction
  is whether a process of *ours* has exited: if `_check_still_alive` raises, the
  server is gone and its own `scancel` has already ended the allocation, so there is
  nothing to hold and the session tears down as usual. If both ends are alive and
  the socket merely did not work out, the allocation is still good and the useful
  thing to offer is another go -- so the state goes to `DOWN`, `holds_allocation`
  stays true, and the card keeps both *Move GPU here* and *Disconnect*. "Give the
  GPU back" has to be one click from every state that is holding one.
- **A link landing for the other playground is not handed over.** Switch to B, then
  `Tab` back to A before it lands, and B's link would otherwise be attached to A's
  system -- one simulation's beads drawn into another's scene. `RemotePanel.update`
  checks `session.serves(playground_key)` first.

A client that names **no** playground (the `--remote HOST:PORT` CLI path) leaves the
server on whatever its own `--playground` said. That keeps "connect to what is
running" and "put this on the GPU" as two different things.

### Reading a failure: the report

The card can show fourteen log lines, clipped to its width. The log holds two
hundred, and the settings that produced them are not on the card at all -- so the
first real failure was debugged off a photograph of a screen, badly.

**C**, or the *Copy report* button, puts `RemoteSession.diagnostics()` on the
clipboard and writes the same text to `/tmp/lammps-live-remote-<stamp>.txt`. It is
the state, the step, the error, the login, the allocation, the job id, the node,
both ends of the tunnel, the far-side paths, whether each of the three child
processes (login master, `srun`, tunnel `ssh`) is still running, whether this end
of the tunnel is still listening *right now*, the probe's summary, and then the
whole log untruncated. The token is deliberately **not** in it: the report is meant
to be pasted into a chat window, and a live token plus a node name is enough for
anyone on the cluster to take over the stream.

The terminal path writes the same file on failure:

```
python -m lammps_live.remote.session --playground mesomem_remote
...
failed: the tunnel is open but the server did not answer: ...
report: /tmp/lammps-live-remote-20260819-142233.txt
```

### "The tunnel is open but the server did not answer"

That message is three different failures wearing one sentence, because the client
socket genuinely cannot tell them apart -- two of them arrive as nothing but a
failed `connect()` to `127.0.0.1:<local_port>`:

| what the report says | what it means |
|---|---|
| `127.0.0.1:5723 is NOT listening` | the ssh holding the forward has gone; the far side was never involved |
| `... accepts connections` + a handshake error | the forward is up and the server is not there: wrong node, wrong port, or it died |
| `... accepts connections` + `bad token` / version mismatch | both ends alive, disagreeing -- the deployed copy is stale |

The first row has two known causes, and neither is a network problem.

**Your `~/.ssh/config`, if the tunnel hop is not defending itself.** This is what it
actually was the first time (see [§9](#9-six-real-bugs-fell-out)): a
`ControlMaster auto` + `ControlPersist` stanza matching the compute nodes makes
`ssh -N -L` background itself and exit 0, taking the forward with it. Fixed in the
code -- that hop passes `ControlPath=none` now -- and the tell, if anything ever
overrides it again, is a tunnel that exits **status 0**: a clean exit from a process
that was asked to stay is not a failure, it is a process that thinks its job is
done elsewhere.

**A chain reaction from the far end.** The server exits for its own reasons, its
`scancel $SLURM_JOB_ID` on the way out ends the allocation (that is deliberate --
it is what stops an abandoned A100 costing money), Slurm then kills every session
on the node including the ssh the tunnel is made of, and the local port stops
listening a moment later. The *symptom* is at this end; the *cause* is in the
`[server]` lines above it. So the connect step checks both child processes on every
retry and names whichever one died, rather than retrying nine more times and then
blaming the tunnel.

Two knobs when the report is not enough:

- `LAMMPS_LIVE_SSH_VERBOSE=1` adds `ssh -v` to the tunnel hop, so its trace
  (channel-open failures, a session the node closed, the ProxyCommand's own
  troubles) lands in the same log. Off by default: fifty lines of noise in front of
  every *successful* connect.
- `LAMMPS_LIVE_REMOTE_TUNNEL=forward` falls back to the one-hop form. If two hops
  fail and one hop works, the problem is ssh-to-the-compute-node -- a site policy
  or a PAM stack, not this code.

And the same bisect by hand, with the job id and node from the report:

```
ssh snellius.surf.nl squeue -j <job>            # is the allocation still there?
ssh -J snellius.surf.nl <node> "ss -ltnp | grep 5723"   # is anything listening?
ssh -v -N -L 5723:127.0.0.1:5723 -J snellius.surf.nl <node>   # does the hop hold?
```

---

## 8. Testing: a cluster that fits on one machine

The SSH and Slurm half cannot be exercised casually against the real thing -- every
attempt costs a one-time code and a queue wait. So `tests/fake_cluster/` stands in
for it: a fake `ssh` that understands the handful of invocations the session
actually makes, plus fake `salloc`, `squeue` and `scancel` on `PATH`.

It is a **stand-in, not a mock**. The commands the session builds are *run*, not
inspected: the tar is really unpacked, the probe really runs, the server really
starts, the token really arrives on stdin, the `-O forward` really starts a proxy,
and frames really come back through it. Only the machine boundary is fake. The
fake `ssh` even calls `SSH_ASKPASS` twice, with two different prompts, and checks
the answers -- so the OTP bridge is tested, including the wrong-answer path and
cancelling.

Alongside it: a **loopback** test that runs a real LAMMPS server over a real socket
on this machine (`--profile local`), which is where the codec, the sliders, the
play/pause/reset semantics, the frame dropping and the disconnect handling are
verified. 136 tests, ~2 minutes.

You can use that path yourself, with no cluster at all:

```
python -m lammps_live.remote.server --playground mesomem_remote --profile local \
       --token dev --port 5723
lammps-live --playground mesomem_remote --remote 127.0.0.1:5723 --token dev
```

---

## 9. Eight real bugs fell out

Two were pre-existing and unreachable before this; the third was mine (see
[§2.1](#the-trap-in-that-which-cost-the-first-real-run)); the fourth came out of
writing down why `--bind 0.0.0.0` felt wrong; the last one took a third real
Snellius run and a diagnostics report to see at all:

- **Closing a LAMMPS instance while its own thread is inside `run` segfaults the
  interpreter.** Found from a test fixture tearing the server down while it was
  integrating. Stopping is now a two-step move -- stop the serve loop, *then* close
  what it was using (`FrameServer.stop()` / `.close()`), and the server installs a
  `SIGTERM` handler so `scancel` ends it between chunks rather than mid-`run`.
- **The renderer indexed `pts[0]` of an empty array.** The puller-anchored overlays
  assumed at least one particle. A disconnected remote playground is a scene with
  zero particles, which is a real state, not a degenerate one.
- **The build check ran where it could not possibly succeed.** A CUDA
  `liblammps.so` needs the NVIDIA driver, so it only opens on a GPU node -- and the
  probe ran on the login node, by a deliberate decision to spend no queue time on a
  broken build. Correct instinct, wrong consequence: the check now runs at two
  levels, and the login-node one stops before the `dlopen`. Found by the second real
  Snellius run.
- **A connection that said nothing wedged the server permanently.** It serves one
  client at a time by design, and the handshake read had no timeout -- so a single
  silent connection (a stray port scan on a shared network is enough) meant nobody
  could ever connect again. There is now a 20-second handshake timeout, cleared
  once the client is known: it must NOT stay on afterwards, because the same socket
  is read by the control thread, which is legitimately idle for minutes between
  slider movements. One socket, two threads, opposite timeout needs.
- **Every remote command was passed to ssh as separate argv words**, which ssh
  flattens into one string, so `bash -lc 'a && b'` ran only `a`. Found by the first
  real connection to Snellius, at the deploy step. Fixed in one place
  (`_login_shell`), and the fake cluster now flattens arguments the same way real
  ssh does -- because it previously did not, which is why the tests passed.
- **The user's own `~/.ssh/config` changed what `ssh -N -L` means.** A
  `ControlMaster auto` + `ControlPersist 4h` stanza matching `gcn*` -- a sensible
  thing to have, and there precisely to make ssh-ing to your own job's node cheap --
  makes ssh put the master in the *background* and the process we started exit
  immediately, status 0. The local port is bound for a moment on the way out, so
  the forward looks like it came up; the connect a fraction of a second later is
  refused. It surfaced as `the tunnel is open but the server did not answer: could
  not reach 127.0.0.1:5723`, which is true, useless, and points at the wrong end of
  the link.

  The login master already defended itself against this (`ControlPersist=no`, its
  own `ControlPath`) with a comment explaining why. The tunnel hop, written later,
  did not -- so it now passes `ControlMaster=no` and `ControlPath=none`, which
  disables multiplexing for that connection outright. And `_await_forward` no longer
  believes a listening port on its own: it checks the port *and* the process holding
  it in the same breath, because for a moment during that failure both "the port
  answers" and "the process is gone" were true.

  The general lesson is worth more than the fix: **every ssh this code runs is
  subject to the user's config**, and a demo that has to work in front of people
  cannot inherit settings it did not ask for. Each hop now says what it needs.
- **The cluster's atom style takes its mass per particle, so `mass 1 1.0` was
  invalid there.** With the tunnel fixed, the fourth run got all the way to the
  server building the simulation, which failed on the very first command the force
  field emits:

  ```
  ERROR: Cannot set per-type atom mass for atom style dipole_sphere_angle/kk
  Last input line: mass 1 1.0
  ```

  `hybrid sphere dipole` accepts the per-type spelling; `dipole_sphere_angle` stores
  `rmass` per atom and rejects it, so the same number has to be written
  `set type 1 mass 1.0`. `HostProfile` already existed for exactly this kind of
  difference and its docstring already listed `rmass` among the fields -- the
  conclusion just had not been drawn. It now carries `per_atom_mass` and rewrites
  that one command, the way `coeff_values` rewrites `pair_coeff`.

  **Why nothing caught it.** The probe builds two atoms and runs the pair style, and
  it reached for `set group all density 1.0` -- the *per-atom* idiom -- so the
  command the server would actually issue was never tried anywhere. The probe now
  issues the profile's own mass command, so a wrong declaration costs ten seconds on
  the node instead of an allocation, and says which spelling the build wanted.
- **A deck that failed took the process down with SIGABRT and a core dump.** The
  half-built LAMMPS instance was left to the garbage collector, so its destructor
  ran during interpreter shutdown with CUDA already unloading; Kokkos threw from
  that destructor (`cudaErrorCudartUnloading`) and `std::terminate` buried the real
  error under sixty lines of backtrace. `PlaygroundSystem._setup` now closes the
  instance on any failure before re-raising, which finalises Kokkos while CUDA is
  still alive -- so the error arrives as itself, exit status 1, and the client's
  "could not build the simulation: ..." message is the whole story.
- **One slider value cost the whole session.** With the demo finally streaming --
  20,735 frames off the A100 -- dragging `zeta` below 1 and pressing Reset killed
  the server, its allocation and the tunnel:

  ```
  ERROR: mesomem/kk requires zeta >= 1 (src/KOKKOS/pair_mesomem_kokkos.cpp:585)
  Last input line: run 0
  ```

  Three separate things were wrong, and the middle one is the interesting one:

  1. `zeta < 1` is legal to the CPU pair style and rejected by the Kokkos one, so
     it is a *build* difference like the others in [§6](#6-the-clusters-lammps-is-not-our-lammps).
  2. **It streamed for six minutes before failing.** Coefficients are validated at
     `init_one`, which runs on a full `run` setup -- and the per-chunk `run ... pre
     no` deliberately skips that (it is worth 5-14% of a chunk). So the value was
     live and unvalidated until Reset did a real build. A parameter can therefore be
     *accepted, applied, and fatal later*, which is a shape of failure worth knowing
     about.
  3. `PlaygroundSystem.step` already survived a blow-up -- catch, latch, tell the
     HUD, wait for a rebuild -- but `reset()` had no such protection, so the one
     action that is supposed to be the way OUT of a broken simulation was the action
     that killed the process. And on the cluster the process exiting means
     `scancel`, so the GPU went with it.

  Now: a rebuild falls back (current values, then the ones that last built, then the
  playground's own), the event is reported once as a red card with a plain sentence
  and the raw error, the sliders follow whatever the simulation settled on, and a
  setting that destroys every fresh state gets one automatic recovery and then stops.
  See `playground/faults.py`, `ui/alert.py`, and the README section.

  **A trap inside the fix, caught by a test.** The fault first travelled as a field
  on the frame header, which is exactly wrong here: the client keeps only the latest
  frame and drops the rest, so on a link fast enough to drop frames -- the point of
  the A100 -- the one frame carrying the event is usually the one thrown away. It is
  its own message now. An event that is only sometimes delivered is worse than one
  that is never sent, because you stop looking for the bug.

---

## 10. What I fixed on the way: the analysis scheduler

Re-measuring §3 of the plan on a real *coarsened* 10k configuration (22.8
neighbours per bead, not the 11.5 of the initial gas the old extrapolation assumed)
turned up something better than a slow function -- a wrong schedule:

| | measured at 10k |
|---|---|
| `build_pairs` (cKDTree, rc = 2.5) | **21.7 ms** |
| `energy_terms` over 113k pairs | 9.7 ms |
| all three observables together | 0.8 ms |
| one full `Analysis.update` | 31.7 ms |
| throttled average, **before** | 17.4 ms/frame |
| throttled average, **after** | **6.8 ms/frame** |

Two changes, both free:

1. **The observables' phases were deliberately staggered** so they would not land
   on the same frame. With a *shared* pair list that is backwards: the list is the
   expensive part, and three every-4-frames observables spread over three frames
   rebuilt it three times instead of once. They are now aligned.
2. **Two of the three never look at the pair list** (`nematic_S`, `thickness`).
   They now declare `needs_pairs=False` and cannot trigger a build. The flag
   defaults to `False`, so a new observable that forgets to declare it gets an
   empty pair list -- an obviously wrong answer rather than a quietly expensive
   right one.

The 2.6× is real but the peak is unchanged: **31.7 ms on one frame in eight**, a
visible dip to ~31 fps at 7.5 Hz. That is the first thing on the list below.

### The numbers as they stand

| | measured |
|---|---|
| GL render, 10k beads | 2.6 ms/frame |
| frame decode | 0.4 ms |
| trajectory smoothing | 0.4 ms |
| client analysis, average | 6.8 ms/frame |
| client analysis, peak (1 frame in 8) | 31.7 ms |
| wire (`q12`) | 6.5 B/bead → 65 kB/frame, 31 Mbit/s at 60 fps |
| position error | 0.005σ (0.2 px windowed, 0.33 fullscreen) |
| director error | 0.94° worst case; moves `nematic_S` by 8e-5 |
| sim, A100, 10k, 20 steps | ~0.7 ms/frame (estimated from plan §4; the card is idling) |

---

## 11. What to do next: to 100k and beyond

At 10k the client is the bottleneck and the A100 is at a few percent of capacity.
The order below is by payoff per unit of work, and the first two items are what
actually stand between here and 100k.

### 1. Move the analysis off the frame path entirely -- **DONE**

This was the biggest win and it is in: `remote/client.py`'s `FrameAnalysis` takes
whatever the newest frame is, measures it on a thread and a clock of its own, and
publishes finished results for the drawing thread to read. Nothing on the frame
path waits for any of it. What stays on the frame path is the decode and the
rattle's amplitude, ~4 ms at 50k.

**What forced it.** At 50,000 beads, on a *coarsened* configuration rather than the
opening gas:

| | measured at 50k | ran |
|---|---|---|
| decode + rattle | 4 ms | every frame -- stayed |
| RDF sample | 5 ms | every frame |
| `Analysis.update` (sampled pair list) | 16 ms | one frame in 4 |
| **cluster labelling** | **185 ms** | one frame in 33 |

The labelling is the one that showed: 185 ms is eleven frames the window did not
draw, arriving every 33rd frame of a 20 fps wire, i.e. **a visible hitch every 1.6
seconds** -- which is exactly what it looked like in the demo. It had already been
moved from the readout to the stepper thread, and that could not fix it: `App._tick`
joins the stepper before it draws, so the stepper is the frame path with one frame
of slack, not an escape from it.

Measured after, driving the client at 60 Hz against a 20 fps synthetic wire at 50k
with the cluster colouring on:

| per-tick cost | before | after |
|---|---|---|
| median | 1.9 ms | 1.9 ms |
| p99 | 30.7 ms | 10.2 ms |
| worst | 40+ ms, every 1.25 s | 21.8 ms, warm-up only |
| what the frame waited for (`wait` p99) | 30.3 ms | 0.1 ms |

The GIL warning below turned out to be the right thing to watch and a survivable
one: the labelling is vectorised numpy, which releases it in the inner loops, so
what is left of a 185 ms pass on the drawing thread is a couple of frames at ~20 ms
instead of eleven frames of nothing. The `--debug` breakdown reads this correctly
with no change: `analysis` is charged only what a frame waited for -- zero, now --
and the wall figure after the sum still says what the pass cost.

Watch for, if this is revisited: the GIL. Python threads only genuinely overlap
where the work releases it, which numpy and scipy mostly do. (With the *CPU*
renderer the contention is severe: the same analysis measured 6.8 ms standalone and
30 ms inside a frame that spent 520 ms in software rendering.)

### 2. Stop building a pair list on this machine at all (the real 100k answer)

Every expensive thing in the analysis exists because of one 21.7 ms KD-tree. All
three of its consumers can get their answer from somewhere better:

| quantity | today | instead |
|---|---|---|
| `coordination` | mean over the client's pair list | LAMMPS `compute coord/atom cutoff rc` on the server, reduced to one float. **Free** -- the neighbour list already exists on the GPU |
| the three energy terms | Python pass over 113k pairs (9.7 ms; ~100 ms at 100k) | accumulate three running sums *inside the pair style's `compute()`* and expose them as a global vector. It is your C++, and the loop is already running on the GPU. **Three floats on the wire instead of a Python pass** |
| `nematic_S`, `thickness` | O(N) numpy, no pairs | subsample 10k of the 100k beads. Both are intensive quantities; a random subsample is unbiased and the rolling display averages the extra noise away |

Do those three and the client's per-frame Python cost goes to roughly nothing at any
N, and the wall moves to the wire. This is the same conclusion as the plan's
"analysis off, or from GPU-side computes → ~125,000 beads" row, made concrete.

Cheaper stopgap if you want the peak gone this afternoon:
`cKDTree.query_ball_point(..., workers=-1)` is multithreaded where `query_pairs` is
not -- worth perhaps 3× on the laptop's cores, at the cost of assembling the pair
arrays from per-point lists in Python.

### 3. The wire at 100k

6.5 B/bead × 100k × 60 fps = **39 MB/s, 312 Mbit/s** (it was 480 before the codec
went to 12-bit positions and 8-bit directors -- see remote-networking.md §4).
Fine on a wired research network, hopeless on a hotel connection. In order of
payoff:

1. **Send fewer frames** -- DONE, and it is the biggest single lever: the wire
   defaults to 20 fps, so the numbers above are already 3x lower in practice
   (13 MB/s, 104 Mbit/s at 100k). Two things had to come with it, and neither is
   the interpolation this list originally called for. The server derives its
   stride from the send rate so the demo's pace does not change; and the client
   synthesises the thermal rattle between frames, because measurement showed
   interpolation costs a whole wire frame of latency and EXTRAPOLATION IS WORSE
   THAN FREEZING THE PICTURE (the motion between frames is uncorrelated noise, so
   a velocity estimate is a random number). remote-networking.md §8 has the
   measured table.
2. **Temporal delta + zstd** (~4.3 B/bead against the *old* 10 B baseline, so
   less headroom than it looks now that the codec is 6.5). This is where the
   stateless-frame decision has to be revisited: deltas need the previous frame, so
   the client would have to ask for a keyframe after a drop. Design it as
   "keyframe every N, delta in between, request a keyframe on loss" -- the same
   shape as a video codec, and for the same reason.
3. **Camera-aware culling.** The client knows the camera; send it up and let the
   server drop what cannot be seen. A dense 81σ box only ever shows its shell.
   Use a coarse depth test, *not* a radius test -- lamellae are visible through
   gaps, and a naive shell cull deletes real structure.
4. **Directors at a lower rate than positions.** They change slowly under
   smoothing, and they are 4 of the 10 bytes.

### 4. Confirm the GPU side is actually on the GPU

Three checks, all in `snellius/README.md`, all still open:

- Grep the server's log for whether `fix nve/sphere update dipole` resolved to
  `nve/sphere/kk`. If it did not, the integrator is copying the whole state
  host↔device every step and the pair kernel's win is gone. The probe reports
  whether the style exists; the log says whether it was *used*.
- Read `Loop time` and `Performance`, never the section timers, unless you add
  `timer sync`. Kokkos kernel launches are asynchronous, so a `Pair` line can stamp
  the launch and let the work land in `Comm` -- which is how a "526×" speedup that
  is above the hardware roofline gets reported.
- Once the pair kernel is no longer dominant, find out what is: `Comm`, `Neigh`
  (rebuilt every step, and a full list is bigger), or `Modify`.

### 5. Then the interesting question: how big?

Once the analysis is off the frame path and the panels come from GPU-side computes,
the remaining costs scale gently:

| | 10k | 100k | 200k |
|---|---|---|---|
| GL render (measured, plan §4) | 2.6 ms | 6.4 ms | 10.3 ms |
| decode + smoothing | 0.8 ms | ~8 ms | ~16 ms |
| sim on one A100 (estimated) | 0.7 ms | 7 ms | 14 ms |
| wire at 60 fps, 6.5 B/bead | 31 Mbit/s | 312 | 624 |

So **100k at 60 fps is reachable** with items 1–3 done, and 200k is the point where
the laptop's own per-frame numpy (decode, smoothing, the CPU-side scene assembly
the plan flags as unmeasured) starts to matter more than anything remote. One A100
is not saturated below ~200k, so multi-GPU stays off the list.

### 6. Smaller things worth doing

- **Reuse the client's frame buffers.** At 100k every frame allocates a few MB of
  fresh numpy arrays. Decoding into pre-allocated buffers is easy and removes a
  chunk of GC churn from the frame path.
- **Reconnect automatically.** The server already survives a client vanishing (it
  keeps the simulation and waits, paused), and the panel already notices a dead
  link. One retry loop away from a tunnel drop being invisible.
- **A "connect" preset for a bigger run.** `RemoteTarget` and the playground's
  bead count are the only two things that change between the 10k test and a 100k
  demo -- `LAMMPS_LIVE_REMOTE_*` environment overrides already cover the target
  half.
- **Interpolate on the client** even at 60 fps: it would make a dropped frame
  invisible rather than a small jump.
- **Try `--free-run`.** The server can integrate flat out between sends instead of
  one chunk per frame -- the same physics, fast-forwarded, which on an A100 that can
  do several hundred chunks a second turns the coarsening run into something you
  watch in seconds rather than minutes. It is off by default because the sliders
  then act on a simulation that has already moved on.

---

## Appendix: the first real run

**Run 1 (2026-08-18)** got as far as the deploy step and failed on the argv
flattening described in [§2.1](#the-trap-in-that-which-cost-the-first-real-run) --
the login and the one-time code bridge both worked first time.

**Run 2** reached the probe and found two things: the interpreter that `env.sh`
leaves on PATH had no numpy (a real problem, on the far side), and the probe itself
was asking a GPU question on a login node (a real problem, in here). Both are fixed;
the probe now names the missing dependency instead of blaming LAMMPS, and checks the
build on the node. The rest of the list below is still unexercised, in likely order
of trouble:

1. **`python3` after `source _build/hpc/env.sh` may not be the python with the
   `lammps` module.** Run the probe on the login node first; if it needs a
   different interpreter, `LAMMPS_LIVE_REMOTE_PYTHON=/path/to/python` overrides it
   (every `RemoteTarget` field has an environment override).
2. **The login may ask something the panel has not seen.** It relays whatever SSH
   asks, so this should just work -- but if SURF's prompt sequence surprises it, the
   panel's log shows the exact question.
3. **`srun --jobid=` into a `--no-shell` allocation** is the documented way to use
   one, but if your Slurm objects, `--overlap` is the flag it wants.
4. **`ssh` to the compute node** (the two-hop tunnel) is verified on Snellius and
   asks for nothing, but it is a site policy. If a future node refuses, set
   `LAMMPS_LIVE_REMOTE_TUNNEL=forward` to fall back to the one-hop form -- the
   server's bind address follows automatically.
4. **The queue.** `gpu_a100` may not be instant; the panel shows Slurm's own reason
   while it waits, and Cancel is safe at any point.
5. If a step fails, run the same flow on the terminal, where the output is not
   wrapped in a GUI:

   ```
   python -m lammps_live.remote.session --playground mesomem_remote --play
   ```
