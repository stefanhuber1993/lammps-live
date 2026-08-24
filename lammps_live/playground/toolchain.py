"""Which compiler, which flags, which MPI -- decided once, for this machine.

The pair style is compiled here rather than downloaded (plugin.py), and until now
the command line that did it was four hardcoded flags that happened to be right on
one laptop. They are not right everywhere, and the differences are not cosmetic:

  * THE COMPILER IS NOT ALWAYS `clang++`. Linux boxes have g++ and often a much
    newer one behind a module; Windows has MSVC's `cl`, whose command line shares
    no syntax at all with the Unix one -- `/LD` not `-shared`, `/O2` not `-O3`,
    `/I` not `-I`, and an import library at link time instead of undefined
    symbols resolved at dlopen.
  * ARCHITECTURE FLAGS ARE WORTH REAL FRAMES and are spelled differently per
    target. `-march=native` is an x86 spelling; Apple silicon wants `-mcpu`; MSVC
    has no "native" at all, only `/arch:AVX2` and friends. So the flag is PROBED
    (compile an empty file with it) rather than assumed, and a compiler that
    refuses it silently gets a build without it instead of a build failure.
  * MPI IS THE ONE THAT ACTUALLY BREAKS. A pair style includes <mpi.h>
    transitively through LAMMPS' pointers.h AND calls MPI_Bcast, so it has to be
    compiled against the headers of the SAME MPI the loaded liblammps was linked
    against. The pip wheel links MPICH; a distro LAMMPS is usually Open MPI; a
    Windows or serial build has no MPI at all, only LAMMPS' STUBS. Guessing wrong
    gives either a compile error (best case) or a load-time symbol clash.

Everything here is overridable from the config file's `[build]` table
(userconfig.py) and from the environment, so a machine that needs something
unusual needs no source edit:

    CXX / LAMMPS_LIVE_CXX            the compiler
    LAMMPS_LIVE_CXXFLAGS             extra flags, split on whitespace
    LAMMPS_LIVE_BUILD_ARCH           "native", "none", or a literal flag
    LAMMPS_LIVE_MPI_INCLUDE          the directory holding mpi.h
    LAMMPS_LIVE_BUILD_VERBOSE=1      echo the command

`lammps-live --doctor` prints the whole resolution, which is the thing to paste
into an issue when a build fails on a machine nobody here has.
"""
import glob
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

# --- what a platform calls a shared library ---------------------------------
if sys.platform == "darwin":
    LIB_SUFFIX = ".dylib"
elif os.name == "nt":
    LIB_SUFFIX = ".dll"
else:
    LIB_SUFFIX = ".so"


def platform_tag():
    """`darwin-arm64`, `linux-x86_64`, `windows-amd64`.

    Part of the compiled library's filename, because a home directory is often
    shared between machines -- an NFS $HOME mounted on both an x86 login node and
    an aarch64 workstation would otherwise have the two builds fight over one
    path, and the loser is a `plugin load` of a library for the wrong ISA.
    """
    system = {"darwin": "darwin", "win32": "windows"}.get(sys.platform, "linux")
    machine = (platform.machine() or "unknown").lower().replace(" ", "")
    return f"{system}-{machine}"


class ToolchainError(RuntimeError):
    """Something about this machine's build environment is missing, with the fix."""


@dataclass(frozen=True)
class Toolchain:
    """A resolved answer to "how do I compile a LAMMPS pair style here?"."""

    cxx: str
    flavour: str                      # "unix" (gcc/clang syntax) | "msvc"
    std: str = "c++17"
    optimize: str = "-O3"
    arch_flags: tuple = ()
    extra_flags: tuple = ()
    link_flags: tuple = ()
    mpi_include: str = ""
    mpi_mode: str = "auto"            # how mpi_include was arrived at
    notes: tuple = ()                 # things a human should know, not errors
    verbose: bool = False
    source: str = "detected"          # where the settings came from

    # --- the command line -----------------------------------------------------
    def compile_command(self, sources, output, include_dirs=(), lammps_lib=""):
        """The full argv that turns `sources` into the shared library `output`."""
        includes = [d for d in include_dirs if d]
        if self.mpi_include:
            includes.append(self.mpi_include)
        if self.flavour == "msvc":
            objdir = os.path.join(os.path.dirname(output) or ".", "_obj")
            cmd = [self.cxx, "/nologo", "/EHsc", "/MD", "/LD",
                   f"/std:{self.std}", self.optimize,
                   *self.arch_flags, *self.extra_flags,
                   *(f"/I{d}" for d in includes),
                   f"/Fo{objdir}{os.sep}", *sources, f"/Fe:{output}"]
            link = list(self.link_flags)
            if lammps_lib:
                link.append(lammps_lib)
            if link:
                cmd.append("/link")
                cmd.extend(link)
            return cmd
        cmd = [self.cxx, f"-std={self.std}"]
        if self.optimize and self.optimize != "none":
            cmd.append(self.optimize)
        cmd += ["-shared", "-fPIC", *self.arch_flags, *self.extra_flags,
                *self.link_flags, *(f"-I{d}" for d in includes),
                *sources, "-o", output]
        if lammps_lib:
            cmd.append(lammps_lib)
        return cmd

    def signature(self):
        """What has to change for a cached library to be stale.

        The mtime check in plugin.py answers "did the C++ change"; this answers
        "did the way I compile it change" -- adding `arch = "native"` to the config
        and getting the old unoptimised library back is exactly the kind of quiet
        no-op that makes a knob untrustworthy.
        """
        return " ".join([self.cxx, self.flavour, self.std, self.optimize,
                         *self.arch_flags, *self.extra_flags, *self.link_flags,
                         self.mpi_include])

    def describe(self):
        lines = [
            f"compiler      {self.cxx}   ({self.flavour} syntax, from {self.source})",
            f"  std         {self.std}",
            f"  optimize    {self.optimize}",
            f"  arch        {' '.join(self.arch_flags) or '(none)'}",
        ]
        if self.extra_flags:
            lines.append(f"  extra       {' '.join(self.extra_flags)}")
        if self.link_flags:
            lines.append(f"  link        {' '.join(self.link_flags)}")
        lines.append(f"  mpi headers {self.mpi_include or '(none)'}   [{self.mpi_mode}]")
        for note in self.notes:
            lines.append(f"  note        {note}")
        return lines


# --- compiler discovery ------------------------------------------------------
def _msvc_like(exe):
    """Whether this compiler speaks MSVC's command line rather than Unix's.

    `clang-cl` is clang wearing MSVC's flags, which is why the test is on the
    NAME and not on the family: the syntax is the thing that matters here.
    """
    stem = os.path.basename(exe).lower()
    stem = stem[:-4] if stem.endswith(".exe") else stem
    return stem in ("cl", "clang-cl")


def _candidate_compilers():
    """Compilers to try, best first, for this platform."""
    if os.name == "nt":
        # MSVC first: the pip LAMMPS wheel for Windows is an MSVC build, and a
        # plugin has to match its C++ runtime and name mangling.
        return ("cl", "clang-cl", "g++", "clang++")
    if sys.platform == "darwin":
        return ("clang++", "g++")
    return ("g++", "clang++")


def _find_compiler(requested):
    if requested:
        exe = shutil.which(requested) or (requested if os.path.isfile(requested) else None)
        if exe is None:
            raise ToolchainError(
                f"The configured C++ compiler {requested!r} is not on PATH. Set "
                "[build] compiler in your config file, or CXX, to one that is "
                "(`lammps-live --doctor` shows what was searched)."
            )
        return exe, "configured"
    for name in _candidate_compilers():
        exe = shutil.which(name)
        if exe:
            return exe, "found on PATH"
    raise ToolchainError(
        "No C++ compiler found to build the MesoMem pair style. macOS: "
        "`xcode-select --install`. Debian/Ubuntu: `sudo apt install "
        "build-essential`. Fedora: `sudo dnf install gcc-c++`. Windows: install "
        "the Visual Studio Build Tools and run from a 'Developer Command Prompt' "
        "(or set [build] compiler in the config file)."
    )


# Probing costs one tiny compile per flag; the answer cannot change within a run.
_FLAG_CACHE = {}


def flag_supported(cxx, flag):
    """Does this compiler accept this flag? Asked, not assumed.

    `-march=native` is an x86 spelling that some arm64 compilers accept, some
    ignore and some reject; `-mcpu=native` is the reverse. Rather than encode a
    matrix of compiler versions that will be wrong next year, compile an empty
    file and see.
    """
    key = (cxx, flag)
    if key in _FLAG_CACHE:
        return _FLAG_CACHE[key]
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "probe.cpp")
        with open(src, "w") as handle:
            handle.write("int main(){return 0;}\n")
        if _msvc_like(cxx):
            cmd = [cxx, "/nologo", "/c", flag, src, f"/Fo{tmp}{os.sep}"]
        else:
            cmd = [cxx, "-Werror", flag, "-c", src, "-o", os.path.join(tmp, "probe.o")]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            ok = proc.returncode == 0
        except (OSError, subprocess.SubprocessError):
            ok = False
    _FLAG_CACHE[key] = ok
    return ok


def _arch_flags(cxx, requested):
    """Turn `arch = "native"` into whatever this compiler calls that.

    Returns (flags, notes). "none" is honoured exactly (a reproducible build, or a
    binary that has to run on a whole cluster from a shared filesystem); anything
    starting with `-` or `/` is passed through as written, because the point of the
    knob is to let someone say `-march=x86-64-v3` when the machine that compiles is
    not the machine that runs.
    """
    if requested in (None, "", "none", "off", False):
        return (), ()
    if isinstance(requested, (list, tuple)):
        return tuple(str(f) for f in requested), ()
    requested = str(requested)
    if requested not in ("native", "auto", "on", "true"):
        return tuple(requested.split()), ()
    if _msvc_like(cxx):
        # MSVC has no "compile for this exact CPU" -- only ISA levels, and picking
        # one here would silently produce a binary that will not start on an older
        # machine. Say so and leave it alone; `arch = "/arch:AVX2"` is explicit.
        return (), ('MSVC has no -march=native; set [build] arch = "/arch:AVX2" '
                    "explicitly if every machine that loads this build has it",)
    for flag in ("-march=native", "-mcpu=native"):
        if flag_supported(cxx, flag):
            if flag_supported(cxx, "-mtune=native"):
                return (flag, "-mtune=native"), ()
            return (flag,), ()
    return (), (f"{os.path.basename(cxx)} accepted neither -march=native nor "
                "-mcpu=native; building without architecture tuning",)


# --- MPI ---------------------------------------------------------------------
def _mpi_from_wrapper():
    """Ask an MPI compiler wrapper where its headers are."""
    for wrapper in ("mpicxx", "mpic++", "mpicc"):
        exe = shutil.which(wrapper)
        if not exe:
            continue
        for flag in ("-showme:incdirs", "-show", "-compile_info"):
            try:
                out = subprocess.check_output([exe, flag], text=True,
                                              stderr=subprocess.DEVNULL, timeout=30)
            except Exception:
                continue
            for tok in out.replace("-I", " -I").split():
                cand = tok[2:] if tok.startswith("-I") else tok
                if os.path.isfile(os.path.join(cand, "mpi.h")):
                    return cand, f"{wrapper} {flag}"
    return "", ""


def _mpi_from_usual_places():
    """The per-OS locations, globbed rather than hardcoded.

    Debian/Ubuntu put mpi.h under a multiarch subdir (/usr/include/<arch>/mpich);
    Fedora uses /usr/include/mpich-<arch> or /usr/lib64/mpich/include; Homebrew
    and MacPorts have their own prefixes; conda environments carry their own.
    Open MPI is searched too, after MPICH -- the pip wheel wants MPICH, but a
    distro or module LAMMPS is as often Open MPI, and the right answer is
    whichever one THAT build used.
    """
    candidates = []
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        candidates.append(os.path.join(prefix, "include"))
    candidates += ["/opt/homebrew/include", "/usr/local/include",
                   "/opt/local/include", "/usr/include/mpich",
                   "/usr/include/openmpi-x86_64", "/usr/include/openmpi"]
    candidates += glob.glob("/usr/include/*/mpich")        # Debian/Ubuntu multiarch
    candidates += glob.glob("/usr/include/*/openmpi")
    candidates += glob.glob("/usr/include/mpich-*")        # Fedora
    candidates += glob.glob("/usr/include/openmpi-*")
    candidates += glob.glob("/usr/lib*/mpich/include")
    candidates += glob.glob("/usr/lib*/openmpi/include")
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "mpi.h")):
            return cand, "found on this system"
    return "", ""


def _mpi_from_lammps_package():
    """A serial LAMMPS ships its own STUBS mpi.h with the headers. Use it.

    This is the Windows and conda-serial case: there is no MPI installed because
    the build does not use one, and the header that matches it is the one that
    was built against -- LAMMPS' own stub, where MPI_Comm is an int and the
    functions are no-ops living inside liblammps itself.
    """
    try:
        import lammps
    except Exception:
        return "", ""
    root = os.path.dirname(lammps.__file__)
    for cand in (os.path.join(root, "include", "lammps"),
                 os.path.join(root, "include", "lammps", "STUBS"),
                 os.path.join(root, "include")):
        if os.path.isfile(os.path.join(cand, "mpi.h")):
            return cand, "LAMMPS' own MPI stubs"
    return "", ""


_STUB_HEADER = """\
/* Generated by lammps_live/playground/toolchain.py -- see [build] mpi = "stubs".
 *
 * A minimal stand-in for LAMMPS' src/STUBS/mpi.h, for a SERIAL LAMMPS build that
 * shipped no mpi.h with its headers. The declarations below have to match the
 * ABI of the stub library compiled into liblammps: MPI_Comm and friends are ints
 * there, and the functions are single-rank no-ops that liblammps exports. If the
 * build being loaded is a REAL MPI build, this header is wrong and the right fix
 * is to install that MPI's development headers and set [build] mpi_include.
 */
#ifndef LAMMPS_LIVE_MPI_STUB_H
#define LAMMPS_LIVE_MPI_STUB_H
#include <cstdlib>
typedef int MPI_Comm;
typedef int MPI_Fint;
typedef int MPI_Datatype;
typedef int MPI_Status;
typedef int MPI_Op;
typedef int MPI_Request;
typedef int MPI_Info;
#define MPI_COMM_WORLD 0
#define MPI_SUCCESS 0
#define MPI_INT 1
#define MPI_FLOAT 2
#define MPI_DOUBLE 3
#define MPI_CHAR 4
#define MPI_BYTE 5
#define MPI_LONG 6
#define MPI_LONG_LONG 7
#define MPI_DOUBLE_INT 8
#define MPI_MAX 1
#define MPI_MIN 2
#define MPI_SUM 3
#define MPI_STATUS_IGNORE ((MPI_Status *) 0)
extern "C" {
int MPI_Comm_rank(MPI_Comm comm, int *rank);
int MPI_Comm_size(MPI_Comm comm, int *size);
int MPI_Bcast(void *buf, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
int MPI_Barrier(MPI_Comm comm);
}
#endif
"""


def _write_stub_header(directory):
    """Drop the generated stub next to the sources and return its directory."""
    stub_dir = os.path.join(directory, "_mpi_stub")
    os.makedirs(stub_dir, exist_ok=True)
    path = os.path.join(stub_dir, "mpi.h")
    existing = ""
    if os.path.isfile(path):
        with open(path) as handle:
            existing = handle.read()
    if existing != _STUB_HEADER:
        with open(path, "w") as handle:
            handle.write(_STUB_HEADER)
    return stub_dir


def resolve_mpi_include(settings, stub_dir_for=None):
    """Where mpi.h is, and how that was decided. Returns (path, how, notes).

    Order: an explicit setting, then the wrapper of an MPI that is installed, then
    the usual per-OS locations, then LAMMPS' own stubs -- explicit before clever,
    installed before assumed, and the serial fallback last so it never shadows a
    real MPI that a real build was linked against.
    """
    notes = ()
    mode = str(settings.get("mpi", "auto") or "auto").lower()
    explicit = (settings.get("mpi_include")
                or os.environ.get("LAMMPS_LIVE_MPI_INCLUDE")
                or os.environ.get("MESOMEM_MPI_INCLUDE"))
    if explicit:
        explicit = os.path.expanduser(str(explicit))
        if not os.path.isfile(os.path.join(explicit, "mpi.h")):
            raise ToolchainError(
                f"No mpi.h in {explicit} (from [build] mpi_include or "
                "LAMMPS_LIVE_MPI_INCLUDE). Point it at the directory that "
                "CONTAINS mpi.h, not at the file."
            )
        return explicit, "configured", notes
    if mode == "none":
        return "", "disabled", notes
    if mode == "stubs":
        found, how = _mpi_from_lammps_package()
        if found:
            return found, how, notes
        if stub_dir_for:
            return (_write_stub_header(stub_dir_for), "generated stub",
                    notes + ("compiled against a generated serial MPI stub -- "
                             "correct only for a LAMMPS built with STUBS",))
        return "", "stubs unavailable", notes

    for finder in (_mpi_from_wrapper, _mpi_from_usual_places, _mpi_from_lammps_package):
        found, how = finder()
        if found:
            return found, how, notes
    raise ToolchainError(
        "Could not locate mpi.h. The pair style includes it through LAMMPS' own "
        "headers, so it has to be the MPI your liblammps was built against "
        "(MPICH for the pip wheel).\n"
        "  macOS:         brew install mpich\n"
        "  Debian/Ubuntu: sudo apt install mpich libmpich-dev\n"
        "  Fedora:        sudo dnf install mpich mpich-devel\n"
        "  conda:         conda install -c conda-forge mpich mpi\n"
        "Then re-run, or set [build] mpi_include (or LAMMPS_LIVE_MPI_INCLUDE) to "
        "the directory containing mpi.h. For a serial LAMMPS with no MPI at all, "
        'set [build] mpi = "stubs".'
    )


# --- the whole resolution ----------------------------------------------------
def detect(settings=None, stub_dir_for=None):
    """Resolve compiler, flags and MPI for this machine.

    `settings` is the config file's `[build]` table; passing one explicitly is
    what the tests do, and what --doctor does when asked about a hypothetical.
    """
    if settings is None:
        from ..userconfig import build_settings
        settings = build_settings()
    settings = dict(settings or {})
    notes = ()

    requested = (settings.get("compiler") or os.environ.get("LAMMPS_LIVE_CXX")
                 or os.environ.get("CXX") or "")
    cxx, source = _find_compiler(requested)
    if not requested:
        source = f"{source} ({', '.join(_candidate_compilers())})"
    flavour = "msvc" if _msvc_like(cxx) else "unix"

    std = str(settings.get("std") or "c++17")
    optimize = str(settings.get("optimize")
                   or ("/O2" if flavour == "msvc" else "-O3"))

    # Tuning for the machine that will run it, by default: this library is
    # compiled ON the box that loads it, so the usual reason not to (a binary
    # that has to start on an older CPU) does not apply -- and `arch = "none"`
    # is there for the case where it does, such as a shared home directory.
    arch_setting = settings.get("arch", os.environ.get("LAMMPS_LIVE_BUILD_ARCH", "native"))
    arch, arch_notes = _arch_flags(cxx, arch_setting)
    notes += arch_notes

    extra = settings.get("extra_flags") or []
    if isinstance(extra, str):
        extra = extra.split()
    extra = [str(f) for f in extra]
    extra += os.environ.get("LAMMPS_LIVE_CXXFLAGS", "").split()

    link = settings.get("link_flags") or []
    if isinstance(link, str):
        link = link.split()
    link = [str(f) for f in link]
    if flavour == "unix":
        if sys.platform == "darwin":
            # Mach-O wants to be told that undefined symbols are fine: they are
            # resolved against the already-loaded liblammps at dlopen time.
            link += ["-undefined", "dynamic_lookup"]
        elif os.name != "nt":
            link += ["-Wl,--allow-shlib-undefined"]

    mpi_include, mpi_mode, mpi_notes = resolve_mpi_include(settings, stub_dir_for)
    notes += mpi_notes

    verbose = bool(settings.get("verbose")) or os.environ.get("LAMMPS_LIVE_BUILD_VERBOSE") == "1"
    return Toolchain(cxx=cxx, flavour=flavour, std=std, optimize=optimize,
                     arch_flags=tuple(arch), extra_flags=tuple(extra),
                     link_flags=tuple(link), mpi_include=mpi_include,
                     mpi_mode=mpi_mode, notes=notes, verbose=verbose,
                     source=source)


def lammps_include_dir():
    """The LAMMPS headers bundled inside the `lammps` Python package."""
    try:
        import lammps
    except Exception as exc:
        raise ToolchainError(
            "The `lammps` Python module is not importable, so there are no LAMMPS "
            f"headers to compile the pair style against ({exc}). `pip install "
            "lammps`, or point PYTHONPATH at your own build's python/ directory."
        )
    inc = os.path.join(os.path.dirname(lammps.__file__), "include", "lammps")
    if not os.path.isdir(inc):
        raise ToolchainError(
            f"LAMMPS headers not found at {inc}. This happens with a LAMMPS built "
            "without `make install-python`, or a distro package that ships the "
            "module without the development headers -- install the -dev package, "
            "or build LAMMPS with -DBUILD_SHARED_LIBS=yes -DPKG_PYTHON=yes."
        )
    return inc


def lammps_import_library():
    """The link-time library a Windows plugin needs, or "" where none is needed.

    ELF and Mach-O let a plugin leave liblammps' symbols undefined and resolve
    them when the host process dlopens it. PE does not: a DLL must name what it
    imports, so on Windows the plugin has to be linked against liblammps' import
    library (MSVC) or against the DLL itself (MinGW binutils accepts that).
    """
    if os.name != "nt":
        return ""
    try:
        import lammps
    except Exception:
        return ""
    root = os.path.dirname(lammps.__file__)
    for name in ("liblammps.lib", "lammps.lib", "liblammps.dll.a", "liblammps.dll"):
        cand = os.path.join(root, name)
        if os.path.isfile(cand):
            return cand
    raise ToolchainError(
        "No LAMMPS import library (liblammps.lib) found next to the lammps Python "
        f"module in {root}. A Windows plugin DLL cannot leave its LAMMPS symbols "
        "undefined the way a .so or .dylib can, so one is required. Use a LAMMPS "
        "built with -DBUILD_SHARED_LIBS=yes (which produces it), or run the demos "
        "under WSL2, where the Linux path applies unchanged."
    )
