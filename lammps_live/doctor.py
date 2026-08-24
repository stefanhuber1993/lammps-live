"""`lammps-live --doctor`: everything this machine decided, on one screen.

There are now three places a run can be configured -- the playground file, the
config file, the environment -- and two places it can fail before any physics
happens: the C++ toolchain that compiles the pair style here, and the LAMMPS that
has to already exist on the cluster. When someone says "it does not work on my
machine", this is the output to ask for. It answers, in order:

    what am I running on          platform, python, lammps module
    what did the config say       which file, which machine block, which keys
    how would the pair style compile   compiler, arch flags, mpi headers
    is it compiled                the artifact, and whether it is stale
    where would the remote demos go    the RESOLVED target, per playground
    and how do I check that far side   the exact probe command, with your paths

Nothing here starts a simulation, allocates anything, or touches the network. The
one thing it does do is a couple of one-line test compiles, to find out which
architecture flags this compiler actually accepts rather than guess.
"""
import os
import platform
import sys

from . import userconfig

OK = "ok"
WARN = "warn"
BAD = "FAIL"


def _lammps_line():
    """Whether the local LAMMPS module is importable, and which one it is."""
    try:
        import lammps
    except Exception as exc:
        return BAD, [f"lammps module  NOT importable -- {type(exc).__name__}: {exc}",
                     "               `pip install lammps` (the wheel links MPICH), "
                     "or point PYTHONPATH at your own build's python/ directory"]
    lines = [f"lammps module  {lammps.__file__}"]
    version = getattr(lammps, "__version__", "")
    if version:
        lines.append(f"  version      {version}")
    inc = os.path.join(os.path.dirname(lammps.__file__), "include", "lammps")
    if os.path.isdir(inc):
        lines.append(f"  headers      {inc}")
        return OK, lines
    lines.append(f"  headers      MISSING at {inc} -- the pair style cannot be "
                 "compiled against this install")
    return BAD, lines


def _toolchain_lines():
    """How the pair style would be compiled here, or why it could not be."""
    from .playground import toolchain
    try:
        chain = toolchain.detect()
    except toolchain.ToolchainError as exc:
        lines = ["compiler      cannot be resolved"]
        lines += [f"  {line}" for line in str(exc).splitlines()]
        return BAD, lines, None
    return (WARN if chain.notes else OK), chain.describe(), chain


def _plugin_lines(chain):
    """Whether the compiled artifact exists here and matches these settings."""
    from .forcefields.mesomem import MESOMEM_PLUGIN as spec
    from .playground import plugin
    lines = [f"pair style    {os.path.basename(spec.lib_path)}"]
    lines.append(f"  sources     {spec.directory}")
    if not os.path.isfile(spec.lib_path):
        lines.append("  compiled    no -- built on demand the first time a 3D "
                     "scene opens (~10 s), or now with `lammps-live --build-plugin`")
        return WARN, lines
    reason = plugin._needs_build(spec, chain)
    lines.append(f"  compiled    yes, {spec.lib_path}")
    if reason:
        lines.append(f"  stale       yes ({reason}) -- it will be rebuilt on next use")
        return WARN, lines
    lines.append("  stale       no")
    return OK, lines


def _remote_lines():
    """The resolved target of every remote playground, and the probe for it."""
    from .playground import registry
    lines, status = [], OK
    entries = []
    for key, pg in registry.all_playgrounds():
        target = getattr(pg, "remote", None)
        if target is not None:
            entries.append((key, target.resolved()))
    if not entries:
        return OK, ["remote        (no playground declares a remote target)"]

    if not userconfig.load().path:
        # Worth saying once, here rather than in the middle of a Connect: with no
        # config file the login below is whatever the playground FILE declares,
        # which is the author's account and almost certainly not yours.
        lines.append("remote        no config file -- the logins below are the "
                     "playgrounds' declared defaults")
        lines.append("              (`lammps-live --write-config` to set your own; "
                     "docs/cluster-setup.md)")

    seen = set()
    for key, target in entries:
        lines.append(f"remote        {key}  ->  {target.label}")
        lines.append(f"  login       {target.destination}"
                     + ("" if target.user else
                        "   (no user set -- whatever ~/.ssh/config resolves)"))
        lines.append(f"  allocation  {target.partition}, {target.gpus} gpu, "
                     f"{target.ntasks}x{target.cpus_per_task} cpu, {target.time}"
                     + (f", account {target.account}" if target.account else ""))
        lines.append(f"  far side    {target.remote_dir}"
                     f"  (env {target.env_script or '<none>'}, {target.python}, "
                     f"profile {target.profile}, tunnel {target.tunnel})")
        if not target.user:
            status = WARN
        key_id = (target.destination, target.remote_dir)
        if key_id in seen:
            continue
        seen.add(key_id)
        env = (f"cd {target.remote_dir} && source {target.env_script} && "
               if target.env_script else f"cd {target.remote_dir} && ")
        lines.append("  check it    ssh " + target.destination + " bash -lc '"
                     + env + f"{target.python} -c \"import lammps\"'")
        lines.append("              (and, in an allocation, the full probe -- see "
                     "docs/cluster-setup.md)")
    return status, lines


def report(stream=None):
    """Print the whole thing. Returns 0 if nothing is broken, 1 if something is.

    A warning is not a failure: a pair style that has not been compiled yet, or a
    cluster login left to ~/.ssh/config, are both perfectly normal states.
    """
    out = stream or sys.stdout
    worst = OK

    def emit(lines):
        for line in lines:
            print(line, file=out)

    def worsen(status):
        nonlocal worst
        order = {OK: 0, WARN: 1, BAD: 2}
        if order[status] > order[worst]:
            worst = status

    print("lammps-live doctor\n" + "=" * 60, file=out)
    emit([
        f"platform      {platform.platform()}  ({platform.machine()})",
        f"python        {sys.version.split()[0]}  {sys.executable}",
    ])
    print("", file=out)

    status, lines = _lammps_line()
    worsen(status)
    emit(lines)
    print("", file=out)

    cfg = userconfig.load()
    emit(cfg.describe())
    for problem in cfg.problems:
        worsen(WARN)
        emit([f"  PROBLEM     {problem}"])
    print("", file=out)

    status, lines, chain = _toolchain_lines()
    worsen(status)
    emit(lines)
    print("", file=out)

    if chain is not None:
        try:
            status, lines = _plugin_lines(chain)
            worsen(status)
            emit(lines)
        except Exception as exc:               # a broken checkout, not a broken machine
            worsen(WARN)
            emit([f"pair style    could not be inspected: {exc}"])
        print("", file=out)

    try:
        status, lines = _remote_lines()
        worsen(status)
        emit(lines)
    except Exception as exc:
        worsen(WARN)
        emit([f"remote        could not be resolved: {exc}"])
    print("", file=out)

    if worst == OK:
        print("Everything this machine can check by itself looks right.", file=out)
    elif worst == WARN:
        print("Usable, with the notes above. Nothing here stops a local run.", file=out)
    else:
        print("Something above has to be fixed before a scene will open. "
              "docs/install.md is the long version.", file=out)
    print("Remote demos additionally need a LAMMPS that already exists on the "
          "cluster, with the Python module and (for GPU targets) Kokkos+CUDA -- "
          "this machine cannot check that. See docs/cluster-setup.md.", file=out)
    return 0 if worst != BAD else 1
