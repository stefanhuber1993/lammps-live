"""Compile-on-demand LAMMPS pair-style plugins.

Generalized from systems/mesomem_ff/__init__.py, which did exactly this for one
hardcoded pair style. A custom force field is the normal case for someone
inventing one, so this is the path: drop your `pair_*.cpp` / `.h` plus a small
`*plugin.cpp` registration stub in a directory, point a PluginSpec at it, and the
style is compiled into a runtime-loadable shared library and pulled into the
stock pip-installed LAMMPS with `plugin load` -- no full LAMMPS rebuild.

WHAT COMMAND does the compiling is not decided here any more: toolchain.py picks
the compiler, the architecture flags and the MPI headers for whatever machine
this is, from the config file's `[build]` table and the environment. This file is
only the caching -- when is the artifact stale, and what does it get called.

Both of those are per-machine too:

  * THE NAME carries a platform tag (`mesomem-linux-x86_64.so`). A checkout on a
    shared home directory is normal on a cluster, and two machines writing one
    filename is a `plugin load` of a library for the wrong ISA.
  * STALENESS is not only "is a source newer". A build also has to be redone when
    the way it is built changes -- otherwise adding `arch = "native"` to the
    config gives back the same old unoptimised library and the knob looks broken.
    So the command line is written next to the artifact and compared.
"""
import json
import os
import subprocess
from dataclasses import dataclass

from . import toolchain
from .toolchain import ToolchainError  # re-exported: callers catch one type


@dataclass(frozen=True)
class PluginSpec:
    """A custom pair style to compile and load.

    `directory` holds the sources; `sources` are the .cpp files to compile (the
    pair style plus its plugin registration stub); `headers` are watched for
    staleness but not compiled; `lib_stem` names the cached artifact.
    """
    directory: str
    sources: tuple
    headers: tuple = ()
    lib_stem: str = "plugin"

    @property
    def lib_path(self):
        return os.path.join(
            self.directory,
            f"{self.lib_stem}-{toolchain.platform_tag()}{toolchain.LIB_SUFFIX}")

    @property
    def info_path(self):
        """Where the command line that produced `lib_path` is recorded."""
        return self.lib_path + ".build.json"

    def source_paths(self):
        return [os.path.join(self.directory, s) for s in self.sources]

    def watched_paths(self):
        return self.source_paths() + [os.path.join(self.directory, h)
                                      for h in self.headers]


def _recorded_signature(spec):
    try:
        with open(spec.info_path) as handle:
            return json.load(handle).get("signature", "")
    except (OSError, ValueError):
        return ""


def _needs_build(spec, chain):
    """Why this has to be compiled again, or "" if it does not."""
    lib = spec.lib_path
    if not os.path.isfile(lib):
        return "no compiled library yet"
    lib_mtime = os.path.getmtime(lib)
    for path in spec.watched_paths():
        if os.path.isfile(path) and os.path.getmtime(path) > lib_mtime:
            return f"{os.path.basename(path)} is newer than the library"
    if chain is not None and _recorded_signature(spec) != chain.signature():
        return "the build settings changed"
    return ""


def build(spec, chain=None, force=False, log=None):
    """Compile the plugin if it is stale. Returns (path, what_was_done).

    `log` is any `print`-alike; --build-plugin passes one so the command is
    visible, the app passes none so a first launch does not spray a compile line
    across a demo.
    """
    chain = chain or toolchain.detect(stub_dir_for=spec.directory)
    reason = "asked to" if force else _needs_build(spec, chain)
    if not reason:
        return spec.lib_path, "up to date"

    cmd = chain.compile_command(
        spec.source_paths(), spec.lib_path,
        include_dirs=[toolchain.lammps_include_dir(), spec.directory],
        lammps_lib=toolchain.lammps_import_library())
    if log and (chain.verbose or force):
        log(" ".join(cmd))
    if chain.flavour == "msvc":
        # cl writes its .obj files where /Fo points and will not create the
        # directory itself.
        os.makedirs(os.path.join(os.path.dirname(spec.lib_path) or ".", "_obj"),
                    exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ToolchainError(
            f"Failed to compile the {spec.lib_stem} pair-style plugin ({reason}).\n"
            + " ".join(cmd) + "\n" + (proc.stderr or proc.stdout)
            + "\nRun `lammps-live --doctor` to see how the compiler, the "
              "architecture flags and the MPI headers were chosen, and set "
              "[build] in your config file to correct any of them."
        )
    with open(spec.info_path, "w") as handle:
        json.dump({"signature": chain.signature(), "command": cmd}, handle, indent=1)
    return spec.lib_path, reason


def ensure_loaded(spec, lmp):
    """Compile (if stale) and `plugin load` the style into `lmp`. Idempotent per
    LAMMPS instance -- loading twice is harmless (LAMMPS warns and keeps the
    existing style)."""
    build(spec)
    lmp.command(f"plugin load {spec.lib_path}")
    return spec.lib_path
