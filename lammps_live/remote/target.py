"""Where a remote playground runs: one machine, one queue, one tunnel.

Written as a declaration on the playground (`remote=RemoteTarget(...)`) rather
than as flags on the command line, because it is a property of the demo -- "this
one runs on the A100" -- and because the connect flow has to be driveable from a
button, with nothing to type but the login prompt's answer.

Every field can be overridden twice over -- from the CONFIG FILE (a
`[remote.systems.NAME]` block; see userconfig.py) and then from the ENVIRONMENT
(LAMMPS_LIVE_REMOTE_USER, _HOST, _TIME, _PARTITION, _PORT, ...). That is what
makes the same playground file work for a second person with a different account,
a different cluster and a different scratch path without editing it, and it is
the whole answer to "so I am not the only one who can run this":

    RemoteTarget(...) in the playground   what the DEMO needs
    [remote] in the config file           what YOUR MACHINE is    <- persistent
    LAMMPS_LIVE_REMOTE_*                  this run only           <- --gpu-hours

The order is deliberate. The playground declares the shape of the run (a GPU, an
hour, this many ranks); the config file says whose account and which cluster,
because that is a property of the person and not of the demo; the environment is
the one-off on top, which is why the CLI flags write variables rather than
threading values down -- both places a target is resolved go through `resolved()`.
"""
import os
from dataclasses import dataclass, fields, replace

from . import protocol
from .. import userconfig


@dataclass(frozen=True)
class RemoteTarget:
    """An SSH login, a Slurm allocation, and where the code lives on the far side."""

    # --- the login ------------------------------------------------------------
    host: str = "snellius.surf.nl"
    user: str = ""                     # "" -> whatever ~/.ssh/config resolves
    label: str = "remote GPU"
    # --- the allocation -------------------------------------------------------
    partition: str = "gpu_a100"
    gpus: int = 1
    # HOW THIS SITE ASKS FOR A GPU. `--gpus=N` is Slurm 20.11 and later; plenty of
    # clusters are older or simply configure GPUs as a generic resource, where the
    # only spelling that works is `--gres=gpu:1` (or `gpu:a100:1`). Setting this to
    # anything non-empty replaces `--gpus=N` with `--gres=<this>` -- which is the
    # kind of thing that must be a config key, because it is a property of the site
    # and there is no way to guess it from here.
    gres: str = ""
    ntasks: int = 1
    cpus_per_task: int = 18
    # Wall clock for the allocation. This is the backstop that releases the GPU if
    # everything else fails to -- a crashed app, a lost network, a closed lid --
    # so it is deliberately not generous.
    time: str = "01:00:00"
    # How long a queued request may sit before the session gives up. A GPU
    # partition that is full is the normal case, not the exceptional one, and a
    # request behind other people's jobs waits however long they take -- so the
    # cutoff is an hour, not the couple of minutes an idle queue needs. The wait
    # is polled and reported (state and Slurm's own reason), and Cancel ends it,
    # so a long one is legible rather than a spinner. `--time` above is what
    # bounds the allocation once it starts; this bounds getting one at all.
    queue_wait: float = 3600.0
    account: str = ""
    job_name: str = "mesomem-live"
    extra_salloc: tuple = ()
    # --- the far side ---------------------------------------------------------
    # Where the cluster's own LAMMPS build lives, and the script that puts it on
    # PATH. Sourced before the server starts, in a login shell.
    remote_dir: str = "~/Projects/MesoMemLive/mesomem_gpu"
    env_script: str = "_build/hpc/env.sh"
    # Where the connect flow unpacks this package. Not on PATH, not installed:
    # PYTHONPATH points at it, so nothing on the cluster is modified.
    deploy_dir: str = "~/.lammps_live_remote"
    python: str = "python3"
    profile: str = "cluster-gpu"       # a hosts.HostProfile name
    # --- the link -------------------------------------------------------------
    port: int = 5723                   # the server's listening port, on the node
    local_port: int = 5723             # this end of the tunnel
    # HOW THE LOCAL PORT REACHES THE SERVER. Two ways, and the difference is where
    # the SSH session ENDS:
    #
    #   "jump"     (default) a second SSH whose session terminates ON THE COMPUTE
    #              NODE, carried through the login node as an opaque stream. The
    #              login node relays bytes it cannot read, and the forward's far end
    #              is loopback on the node -- so the server binds 127.0.0.1 and the
    #              port is unreachable from the rest of the cluster.
    #   "forward"  one hop: `ssh -O forward -L <local>:<node>:<port>` added to the
    #              login-node connection. The login node terminates the session,
    #              decrypts, and opens a separate PLAIN TCP connection onward to the
    #              node -- so the server has to bind 0.0.0.0 and the port is
    #              reachable (token-protected) from the cluster's internal network.
    #
    # "jump" is better on both counts and is the default. "forward" is kept because
    # ssh-to-a-compute-node is a site policy: it needs sshd running on the node and
    # a PAM stack that lets the owner of a running job in (`pam_slurm_adopt`). Where
    # that is not allowed, one hop is the only way through.
    tunnel: str = "jump"               # "jump" | "forward"
    codec: str = protocol.DEFAULT_CODEC
    # HOW OFTEN THE FAR SIDE SENDS, not how often the window draws. Those were the
    # same number until the client learned to fill in between frames
    # (playground/jitter.py), and separating them is what makes a 50k demo
    # affordable: at 6.5 B/bead a 60 fps wire is 19.5 MB/s, and 20 fps is 6.5 --
    # over an SSH tunnel from Amsterdam that is the difference between a link that
    # keeps up and one that does not.
    #
    # It costs nothing in simulation speed: the server takes a proportionally
    # LONGER stride per frame (see FrameServer._stride_for), so the demo advances
    # at exactly the pace its scenario was tuned at, and only the sampling of the
    # trajectory gets coarser -- which is what the client-side rattle exists to
    # hide. What it does not do is keep the GPU busy; the A100 idles ~96% either
    # way, and that is fine (it is here to make 50k beads possible, not to
    # maximise steps per second).
    fps: float = 20.0
    # Integrate flat out between sends instead of taking one honest stride. Looks
    # like the way to use the idle GPU and is a trap: at 20 fps on an A100 it
    # advances 1,420 steps a frame instead of 60, running the demo 23x fast and
    # leaving consecutive frames so far apart that they are nearly uncorrelated --
    # which is exactly the input the client's smoothing and rattle fill cannot do
    # anything with. Off, and it should stay off unless the pace is the point.
    free_run: bool = False
    # Idle timeout handed to the server, so an abandoned allocation ends itself
    # even if the teardown never runs. Slurm's --time is the outer backstop; this
    # is the one that gives the GPU back in minutes rather than an hour.
    exit_when_idle: float = 900.0

    def resolved(self):
        """A copy with the config file, then LAMMPS_LIVE_REMOTE_*, applied.

        Both layers are optional and the common case is neither. A value that
        cannot be read (a partition given as a list, `gpus = "one"`) is skipped
        with a message rather than raised: the config file is edited by hand and
        one bad line should cost that line, not the demo.
        """
        overrides = dict(userconfig.remote_overrides(self))
        for f in fields(self):
            raw = os.environ.get(f"LAMMPS_LIVE_REMOTE_{f.name.upper()}")
            if raw is None:
                continue
            current = overrides.get(f.name, getattr(self, f.name))
            if isinstance(current, bool):
                overrides[f.name] = raw.strip().lower() not in ("0", "false", "no", "")
            elif isinstance(current, int):
                overrides[f.name] = int(raw)
            elif isinstance(current, float):
                overrides[f.name] = float(raw)
            elif isinstance(current, tuple):
                overrides[f.name] = tuple(raw.split())
            else:
                overrides[f.name] = raw
        return replace(self, **overrides) if overrides else self

    @property
    def destination(self):
        return f"{self.user}@{self.host}" if self.user else self.host

    def node_destination(self, node):
        """The compute node as an SSH destination, for the two-hop tunnel."""
        return f"{self.user}@{node}" if self.user else node

    @property
    def server_bind(self):
        """Which interface the server listens on -- a consequence of the tunnel
        mode, never a separate choice. Two hops end on the node, so loopback is
        reachable and nothing else needs to be; one hop arrives from the login node,
        so loopback would be unreachable."""
        return "127.0.0.1" if self.tunnel == "jump" else "0.0.0.0"

    def gpu_request(self):
        """How to ask for the GPU, as argv -- empty on a CPU-only target.

        One place, because the same request is made three times (salloc, the
        probe's srun, the server's srun) and three copies of a site-specific
        spelling is three chances to fix two of them.
        """
        if self.gres:
            return [f"--gres={self.gres}"]
        if self.gpus:
            return [f"--gpus={self.gpus}"]
        return []

    def salloc_args(self):
        """The allocation request, as argv. `--no-shell` is the whole trick: the
        allocation is created and the command returns, so it does not have to be
        held open by an interactive shell on a pipe -- which is what made the
        obvious `ssh host salloc ... bash` approach so fragile."""
        args = ["salloc", "--no-shell", f"--partition={self.partition}"]
        # A CPU-only target asks for no GPU AT ALL rather than for zero of them:
        # `--gpus=0` is not "never mind", it is a request some Slurm versions
        # reject outright and others grant while binding nothing -- and a CPU
        # cluster (profile "cluster-cpu") is a supported target, not a mistake.
        args += self.gpu_request()
        args += [f"--ntasks={self.ntasks}",
                 f"--cpus-per-task={self.cpus_per_task}",
                 f"--time={self.time}",
                 f"--job-name={self.job_name}"]
        if self.account:
            args.append(f"--account={self.account}")
        args.extend(self.extra_salloc)
        return args
