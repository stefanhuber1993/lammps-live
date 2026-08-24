"""The file that makes this repo somebody else's too.

Everything in here exists because of one sentence from a collaborator: "I want a
config file where the username and HPC system can actually be set, so I am not
the only one running this." Until now the answer to "how do I run the remote
demo" was "edit `lammps_live/playgrounds/mesomem_remote.py` and put your own
username where mine is" -- which is a source edit to change an account name, and
it does not survive a `git pull`.

So there is now one TOML file, found in the usual places, that carries the two
kinds of local truth this app cannot know:

  * WHERE YOU RUN IT REMOTELY -- login, account, partition, paths. One block per
    machine you have, and a name that selects between them, because a person with
    both Snellius and a group cluster should not have to keep two checkouts.
  * HOW IT COMPILES HERE -- compiler, architecture flags, which MPI's headers.
    A pair style compiled with `-O3 -march=native` on the machine it will run on
    is worth real frames, and the flags that do that are different on macOS, on a
    Linux box, and under MSVC (see playground/toolchain.py).

Search order, first hit wins:

    $LAMMPS_LIVE_CONFIG                      an explicit file (or --config)
    ./lammps-live.toml                       next to the checkout you are in
    $XDG_CONFIG_HOME/lammps-live/config.toml
    ~/.config/lammps-live/config.toml        the normal home for it
    ~/.lammps-live.toml                      dotfile, for people who prefer one

There is deliberately NO fourth way to say these things. The precedence stack for
a remote setting is, lowest first:

    the playground's own RemoteTarget(...)   "this demo runs on an A100"
    the config file                          "...and my A100 is reached like this"
    LAMMPS_LIVE_REMOTE_*                     one-off, this run only

`--gpu-hours` and `--hpc` are the CLI's way into the top layer -- they set the
environment variable rather than opening a parallel plumbing route, which is the
same trick cli.py already uses so that both places a target gets resolved (the
RemoteSystem and the RemoteSession) see the same answer.

Unknown keys are reported, never ignored: `usr = "pietro"` silently doing nothing
is a worse failure than a warning, since the symptom is an SSH prompt for the
wrong account twenty seconds later.
"""
import difflib
import os
import sys
from dataclasses import dataclass, field, fields

# Where a build setting can be spelled per-platform: `[build.linux]` overlays
# `[build]` on Linux, and so on. One config file checked into a group's repo then
# covers the laptop and the workstation without a second file.
_PLATFORM_TABLES = {"darwin": "darwin", "win32": "windows", "cygwin": "windows"}

# Every table this file understands. Anything else at the top level is a typo,
# and typos are reported.
_KNOWN_TABLES = ("remote", "build", "app")

_REMOTE_KEYS_NOT_TARGET_FIELDS = ("system", "systems")

_BUILD_KEYS = (
    "compiler", "std", "optimize", "arch", "extra_flags", "link_flags",
    "mpi", "mpi_include", "define", "verbose",
)

_APP_KEYS = ("input", "playground", "ui_scale", "fullscreen")


def _toml_loader():
    """`tomllib` on 3.11+, `tomli` before it, and a plain sentence if neither.

    The dependency is declared in pyproject.toml for the older interpreters, so
    the error below is what a `pip install -e .` was skipped, not a normal path.
    """
    try:
        import tomllib
        return tomllib.load
    except ModuleNotFoundError:
        pass
    try:
        import tomli
        return tomli.load
    except ModuleNotFoundError:
        raise RuntimeError(
            "Reading a config file on Python < 3.11 needs the `tomli` package "
            "(`pip install tomli`, or just `pip install -e .` again -- it is "
            "declared as a dependency for these interpreters)."
        )


def search_paths():
    """The candidate config locations, in the order they are tried."""
    paths = []
    explicit = os.environ.get("LAMMPS_LIVE_CONFIG")
    if explicit:
        paths.append(os.path.expanduser(explicit))
    paths.append(os.path.abspath("lammps-live.toml"))
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        paths.append(os.path.join(os.path.expanduser(xdg), "lammps-live", "config.toml"))
    paths.append(os.path.expanduser("~/.config/lammps-live/config.toml"))
    paths.append(os.path.expanduser("~/.lammps-live.toml"))
    # Keep order, drop duplicates (XDG_CONFIG_HOME is often literally ~/.config).
    seen, unique = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def default_path():
    """Where `--write-config` puts a new file when not told otherwise."""
    xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(os.path.expanduser(xdg), "lammps-live", "config.toml")


def find_path():
    """The config file in force, or None. First hit in `search_paths()`."""
    for path in search_paths():
        if os.path.isfile(path):
            return path
    return None


@dataclass(frozen=True)
class Config:
    """A parsed config file -- or the empty one, which is the common case.

    `problems` is part of the value, not something raised: a typo in one key
    should not stop the app, but it must reach a human. cli.py prints them once
    at startup and `--doctor` prints them in context.
    """

    path: str = ""
    data: dict = field(default_factory=dict)
    problems: tuple = ()

    # --- remote ---------------------------------------------------------------
    @property
    def _remote(self):
        table = self.data.get("remote")
        return table if isinstance(table, dict) else {}

    def system_names(self):
        """Every named machine in the file, in declaration order."""
        systems = self._remote.get("systems")
        return tuple(systems) if isinstance(systems, dict) else ()

    def selected_system(self):
        """Which machine block applies, and where that choice came from.

        Returns (name, source). The environment wins over the file so that
        `--hpc` (which writes LAMMPS_LIVE_SYSTEM) can pick a machine for one run
        without editing anything -- the same one-door trick as --gpu-hours.
        """
        env = os.environ.get("LAMMPS_LIVE_SYSTEM")
        names = self.system_names()
        if env:
            return env, "LAMMPS_LIVE_SYSTEM"
        chosen = self._remote.get("system")
        if isinstance(chosen, str) and chosen:
            return chosen, "config [remote] system"
        if len(names) == 1:
            # One machine and no choice to make. Requiring `system = "..."` here
            # would be ceremony for the single-cluster case, which is most people.
            return names[0], "the only one defined"
        return "", ""

    def remote_settings(self):
        """The remote keys in force: `[remote]`'s own scalars, then the selected
        `[remote.systems.NAME]` on top.

        Keys directly under `[remote]` are the shorthand for "I have one cluster";
        a `systems` block is for "I have two". Both work, and a user key at the
        top level is a sensible default for every machine (it is usually the same
        account name), which is why the merge is in that order.
        """
        merged = {k: v for k, v in self._remote.items()
                  if k not in _REMOTE_KEYS_NOT_TARGET_FIELDS}
        name, _source = self.selected_system()
        if not name:
            return merged
        systems = self._remote.get("systems")
        block = systems.get(name) if isinstance(systems, dict) else None
        if isinstance(block, dict):
            merged.update(block)
        return merged

    def remote_overrides(self, target):
        """The subset of `remote_settings()` that `target` can actually take,
        coerced to the types its fields already hold.

        Takes the instance rather than importing RemoteTarget, both to keep this
        module free of any dependency on remote/ (target.py imports THIS one) and
        because the field types are read off the live values anyway.
        """
        allowed = {f.name for f in fields(target)}
        out = {}
        for key, raw in self.remote_settings().items():
            if key not in allowed:
                continue
            try:
                out[key] = _coerce_like(getattr(target, key), raw)
            except (TypeError, ValueError) as exc:
                self._note(f"[remote] {key} = {raw!r}: {exc}")
        return out

    # --- build ----------------------------------------------------------------
    def build_settings(self):
        """`[build]` with the platform overlay for THIS machine applied on top."""
        table = self.data.get("build")
        if not isinstance(table, dict):
            return {}
        overlay_name = _PLATFORM_TABLES.get(sys.platform, "linux")
        merged = {k: v for k, v in table.items() if not isinstance(v, dict)}
        overlay = table.get(overlay_name)
        if isinstance(overlay, dict):
            merged.update(overlay)
        return merged

    # --- app ------------------------------------------------------------------
    def app_settings(self):
        table = self.data.get("app")
        return dict(table) if isinstance(table, dict) else {}

    def _note(self, message):
        # `problems` is a tuple on a frozen dataclass; append through object.__
        # setattr because a config that notices a bad value while being read is
        # still the same config.
        object.__setattr__(self, "problems", self.problems + (message,))

    def describe(self):
        """A few lines for --doctor. Never prints a password-shaped thing --
        nothing here is a secret, but say so rather than assume it."""
        if not self.path:
            return ["config file   (none found -- `lammps-live --write-config` makes one)"]
        lines = [f"config file   {self.path}"]
        name, source = self.selected_system()
        known = ", ".join(self.system_names()) or "none defined"
        if name:
            lines.append(f"  system      {name}   (from {source}; defined: {known})")
        else:
            lines.append(f"  system      <none selected>   (defined: {known})")
        for key, value in sorted(self.remote_settings().items()):
            lines.append(f"    remote.{key} = {value!r}")
        for key, value in sorted(self.build_settings().items()):
            lines.append(f"    build.{key} = {value!r}")
        return lines


def _coerce_like(current, raw):
    """TOML value -> the type this field already holds.

    TOML has real types, so this is mostly a check rather than a parse; it exists
    for the two cases where being strict would only annoy (an int where a float
    lives, a list where a tuple lives) and to turn everything else into a message
    that names the key.
    """
    if isinstance(current, bool):
        if isinstance(raw, bool):
            return raw
        raise TypeError(f"wants true or false, got {type(raw).__name__}")
    if isinstance(current, int):
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise TypeError(f"wants a whole number, got {type(raw).__name__}")
        return raw
    if isinstance(current, float):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"wants a number, got {type(raw).__name__}")
        return float(raw)
    if isinstance(current, tuple):
        if isinstance(raw, str):
            return tuple(raw.split())
        if not isinstance(raw, (list, tuple)):
            raise TypeError(f"wants a list of strings, got {type(raw).__name__}")
        return tuple(str(v) for v in raw)
    if isinstance(raw, (dict, list)):
        raise TypeError(f"wants a string, got {type(raw).__name__}")
    return str(raw)


def _remote_field_names():
    """The keys a `[remote]` block may set: exactly RemoteTarget's fields.

    Imported lazily and forgivingly. This module is imported BY remote/target.py,
    so a module-level import would be a cycle; and a checkout where that import
    fails for an unrelated reason should lose the typo check, not the config file.
    """
    try:
        from .remote.target import RemoteTarget
        return {f.name for f in fields(RemoteTarget)}
    except Exception:
        return set()


def _check_remote_keys(block, where, known, problems):
    if not known:
        return
    for key in block:
        if key in _REMOTE_KEYS_NOT_TARGET_FIELDS or key in known:
            continue
        near = difflib.get_close_matches(key, sorted(known), n=2, cutoff=0.6)
        hint = f" -- did you mean {', '.join(near)}?" if near else ""
        problems.append(f"[{where}] unknown key {key!r}{hint}")


def _validate(data):
    """Everything that is a typo rather than a value, as a list of sentences."""
    problems = []
    for table in data:
        if table not in _KNOWN_TABLES:
            problems.append(f"unknown section [{table}] -- expected one of "
                            + ", ".join(f"[{t}]" for t in _KNOWN_TABLES))
    remote = data.get("remote")
    if remote is not None and not isinstance(remote, dict):
        problems.append("[remote] must be a section, not a value")
        remote = None
    if isinstance(remote, dict):
        known = _remote_field_names()
        _check_remote_keys(remote, "remote", known, problems)
        systems = remote.get("systems")
        if systems is not None and not isinstance(systems, dict):
            problems.append("[remote.systems] must be a section of named machines")
        elif isinstance(systems, dict):
            for name, block in systems.items():
                if not isinstance(block, dict):
                    problems.append(f"[remote.systems.{name}] must be a section")
                else:
                    _check_remote_keys(block, f"remote.systems.{name}", known, problems)
        chosen = remote.get("system")
        if isinstance(chosen, str) and chosen and isinstance(systems, dict) \
                and chosen not in systems:
            problems.append(
                f'[remote] system = "{chosen}" names no machine in this file '
                f'(defined: {", ".join(systems) or "none"})')
    build = data.get("build")
    if isinstance(build, dict):
        for key, value in build.items():
            if isinstance(value, dict):
                if key not in ("darwin", "linux", "windows"):
                    problems.append(f"[build.{key}] is not a platform "
                                    "-- expected [build.darwin], [build.linux] "
                                    "or [build.windows]")
                continue
            if key not in _BUILD_KEYS:
                problems.append(f"[build] unknown key {key!r} -- expected one of "
                                + ", ".join(_BUILD_KEYS))
    app = data.get("app")
    if isinstance(app, dict):
        for key in app:
            if key not in _APP_KEYS:
                problems.append(f"[app] unknown key {key!r} -- expected one of "
                                + ", ".join(_APP_KEYS))
    return problems


# The parsed file, cached on (path, mtime, size). `resolved()` is called from the
# panel and from the session, potentially per frame, and a config file is a stat
# call rather than a parse each time. The mtime in the key is what makes an edit
# take effect on the next Connect rather than on the next launch.
_CACHE = {}


def load(path=None, use_cache=True):
    """Read the config file in force (or the one named), never raising.

    A file that does not parse is a Config with `problems` and no data: the app
    keeps its declared defaults and says why, which is better than refusing to
    start a demo because of a stray bracket in a file about SSH usernames.
    """
    path = path or find_path()
    if not path:
        return Config()
    path = os.path.expanduser(path)
    try:
        stat = os.stat(path)
        key = (path, stat.st_mtime_ns, stat.st_size)
    except OSError as exc:
        return Config(problems=(f"cannot read {path}: {exc}",))
    if use_cache and key in _CACHE:
        return _CACHE[key]
    try:
        loader = _toml_loader()
        with open(path, "rb") as handle:
            data = loader(handle)
    except Exception as exc:
        cfg = Config(path=path, problems=(f"{path}: {exc}",))
    else:
        cfg = Config(path=path, data=data, problems=tuple(_validate(data)))
    _CACHE.clear()          # one file is in force at a time; never grow this
    _CACHE[key] = cfg
    return cfg


def remote_overrides(target):
    """What the config file says about this target. The door target.py knocks on."""
    return load().remote_overrides(target)


def build_settings():
    """What the config file says about compiling. The door toolchain.py knocks on."""
    return load().build_settings()


def report_problems(stream=None):
    """Print anything wrong with the config file, once. Returns how many.

    Called from cli.main so that a typo is visible in the terminal the run
    started in -- the GUI is the wrong place to notice you spelled `partiton`.
    """
    cfg = load()
    if not cfg.problems:
        return 0
    stream = stream or sys.stderr
    where = cfg.path or "config"
    for problem in cfg.problems:
        print(f"[config] {where}: {problem}", file=stream)
    return len(cfg.problems)


TEMPLATE = '''\
# lammps-live configuration
#
# Two things this file is for: WHERE the remote demos run (your login, your
# allocation, your paths) and HOW the MesoMem pair style is compiled on this
# machine. Everything is optional -- an absent key keeps the built-in default.
#
# Put it at ~/.config/lammps-live/config.toml (where `--write-config` just put
# it), or next to a checkout as ./lammps-live.toml, or anywhere and point
# LAMMPS_LIVE_CONFIG at it.
#
# Check it with:   lammps-live --doctor


# ---------------------------------------------------------------------------
# Where the remote playgrounds run.
#
# The prerequisite is a LAMMPS THAT ALREADY EXISTS on that machine, built with
# the Python module and (for the GPU targets) Kokkos + CUDA, with the MesoMem
# pair style compiled IN rather than loaded as a plugin. Nothing here builds it
# for you -- see docs/cluster-setup.md, and check a candidate build with
#   python -m lammps_live.remote.probe
# ---------------------------------------------------------------------------
[remote]
# Which machine below to use. Omit it if you define exactly one.
system = "snellius"

# Keys here apply to every machine unless its own block overrides them, which is
# handy for the one thing that is usually the same everywhere:
# user = "your-login"

[remote.systems.snellius]
host = "snellius.surf.nl"
user = "your-login"              # "" uses whatever ~/.ssh/config resolves
label = "Snellius gpu_a100"      # what the Connect panel calls it
partition = "gpu_a100"
account = ""                     # Slurm account, if your site needs one
gpus = 1
# gres = "gpu:a100:1"            # Slurm older than 20.11, or GPUs as a generic
                                 # resource: replaces --gpus=N entirely
ntasks = 1
cpus_per_task = 18
time = "01:00:00"                # wall clock; --gpu-hours overrides it
# Where the cluster's own LAMMPS build lives, and the script that puts it on
# PATH (sourced in a login shell before the server starts).
remote_dir = "~/Projects/MesoMemLive/mesomem_gpu"
env_script = "_build/hpc/env.sh"
python = "python3"               # the interpreter that can `import lammps`
profile = "cluster-gpu"          # cluster-gpu | cluster-cpu | local
tunnel = "jump"                  # "jump" (ends on the node) | "forward"
fps = 20.0                       # how often the far side SENDS, not the frame rate

# A second machine, selected with `system = "..."` above or `--hpc groupbox`:
# [remote.systems.groupbox]
# host = "cluster.example.org"
# user = "your-login"
# label = "group cluster (CPU)"
# partition = "compute"
# gpus = 0                       # no GPU asked for at all, not "zero of them"
# cpus_per_task = 32
# profile = "cluster-cpu"
# remote_dir = "~/lammps-mesomem"
# env_script = "env.sh"


# ---------------------------------------------------------------------------
# How the MesoMem pair style is compiled here.
#
# It is built on demand into a runtime LAMMPS plugin the first time a 3D scene
# opens (~10 s, once). These are the knobs; `lammps-live --doctor` prints what
# was detected and `lammps-live --build-plugin` compiles it now.
# ---------------------------------------------------------------------------
[build]
# compiler = "g++"               # default: $CXX, else clang++/g++ (cl.exe on Windows)
# std = "c++17"
# optimize = "-O3"               # "-O2", "-Ofast", "none", ...
arch = "native"                  # "native" | "none" | any flag, e.g. "-march=x86-64-v3"
# extra_flags = ["-fopenmp"]
# link_flags = []
# mpi = "auto"                   # "auto" | "stubs" | "none"
# mpi_include = "/usr/include/x86_64-linux-gnu/mpich"
# verbose = false                # echo the compile command

# Per-platform overlays, so one file can cover a laptop and a workstation:
# [build.linux]
# compiler = "g++"
# arch = "native"
# [build.darwin]
# compiler = "clang++"
# [build.windows]
# compiler = "cl"
# arch = "none"
'''


def write_template(path=None, overwrite=False):
    """Write the starter config. Returns the path; refuses to clobber."""
    path = os.path.expanduser(path or default_path())
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as handle:
        handle.write(TEMPLATE)
    return path
