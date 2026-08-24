"""Compiling the pair style on a machine that is not this one.

The demo compiles its own force field on first launch, which is fine until the
machine is a Linux box with g++ and Open MPI, or Windows with MSVC, or a cluster
login node with a module-provided compiler and a shared home directory. Each of
those breaks a different assumption in the four hardcoded flags this used to
have, so each of them is a test -- and since none of those machines is here, the
tests drive `Toolchain` directly rather than running a compiler.

The two that DO run a compiler are marked: the flag probe (which is the whole
reason `-march=native` is safe to default to) and one real build.
"""
import json
import os
import shutil

import pytest

from lammps_live.playground import plugin, toolchain
from lammps_live.playground.toolchain import Toolchain, ToolchainError


UNIX = Toolchain(cxx="/usr/bin/g++", flavour="unix", optimize="-O3",
                 arch_flags=("-march=native",), mpi_include="/usr/include/mpich",
                 link_flags=("-Wl,--allow-shlib-undefined",))
MSVC = Toolchain(cxx="cl.exe", flavour="msvc", optimize="/O2",
                 arch_flags=("/arch:AVX2",), mpi_include=r"C:\mpi\include")


def test_the_unix_command_is_a_shared_library_with_every_include():
    cmd = UNIX.compile_command(["a.cpp", "b.cpp"], "out.so",
                               include_dirs=["/inc/lammps", "/src"])
    assert cmd[0] == "/usr/bin/g++"
    assert "-shared" in cmd and "-fPIC" in cmd
    assert "-O3" in cmd and "-march=native" in cmd
    assert cmd[-2:] == ["-o", "out.so"]
    for flag in ("-I/inc/lammps", "-I/src", "-I/usr/include/mpich"):
        assert flag in cmd
    assert "-Wl,--allow-shlib-undefined" in cmd, \
        "ELF resolves LAMMPS' symbols at dlopen; the link must allow that"


def test_the_msvc_command_shares_no_syntax_with_it():
    """Not a cosmetic difference: every flag is spelled differently, the output is
    named differently, and -- the one that actually matters -- a DLL cannot leave
    LAMMPS' symbols undefined, so an import library goes after /link."""
    cmd = MSVC.compile_command(["a.cpp"], r"C:\out\mesomem.dll",
                               include_dirs=[r"C:\lammps\include"],
                               lammps_lib=r"C:\lammps\liblammps.lib")
    assert "/LD" in cmd and "-shared" not in cmd
    assert "/O2" in cmd and "/std:c++17" in cmd
    assert r"/IC:\lammps\include" in cmd
    assert r"/Fe:C:\out\mesomem.dll" in cmd
    assert cmd[cmd.index("/link") + 1:] == [r"C:\lammps\liblammps.lib"]


def test_optimization_can_be_turned_off_entirely():
    """`optimize = "none"` is for bisecting a physics bug against a debug build --
    it has to actually drop the flag rather than pass the word to the compiler."""
    cmd = Toolchain(cxx="g++", flavour="unix", optimize="none").compile_command(
        ["a.cpp"], "out.so")
    assert "none" not in cmd and not any(c.startswith("-O") for c in cmd)


def test_the_signature_changes_when_the_way_it_is_built_changes():
    """This is what makes a config change actually rebuild. Without it, adding
    `arch = "native"` returns the cached unoptimised library and the knob looks
    like it does nothing."""
    import dataclasses
    plain = dataclasses.replace(UNIX, arch_flags=())
    assert plain.signature() != UNIX.signature()
    assert dataclasses.replace(UNIX, optimize="-O0").signature() != UNIX.signature()
    assert dataclasses.replace(UNIX, mpi_include="/other").signature() != UNIX.signature()
    assert dataclasses.replace(UNIX).signature() == UNIX.signature()


def test_the_library_is_named_per_platform(tmp_path):
    """A cluster home directory is shared between the login node and a laptop more
    often than not, and one filename for two ISAs is a `plugin load` of a library
    that cannot possibly load."""
    spec = plugin.PluginSpec(directory=str(tmp_path), sources=("a.cpp",),
                             lib_stem="mesomem")
    name = os.path.basename(spec.lib_path)
    assert name.startswith("mesomem-")
    assert name.endswith(toolchain.LIB_SUFFIX)
    assert toolchain.platform_tag() in name
    assert spec.info_path.endswith(".build.json")


def test_a_changed_setting_makes_the_cached_library_stale(tmp_path):
    import dataclasses
    spec = plugin.PluginSpec(directory=str(tmp_path), sources=("a.cpp",),
                             lib_stem="mesomem")
    (tmp_path / "a.cpp").write_text("int main(){}\n")
    open(spec.lib_path, "w").close()
    os.utime(spec.lib_path, None)
    with open(spec.info_path, "w") as handle:
        json.dump({"signature": UNIX.signature()}, handle)

    assert plugin._needs_build(spec, UNIX) == ""
    assert plugin._needs_build(spec, dataclasses.replace(UNIX, arch_flags=())) \
        == "the build settings changed"

    # And the older question still counts: a newer source beats a matching
    # signature.
    os.utime(tmp_path / "a.cpp", (os.path.getmtime(spec.lib_path) + 10,) * 2)
    assert "newer" in plugin._needs_build(spec, UNIX)


def test_a_missing_library_is_the_first_reason_of_all(tmp_path):
    spec = plugin.PluginSpec(directory=str(tmp_path), sources=("a.cpp",))
    assert plugin._needs_build(spec, UNIX) == "no compiled library yet"


# --- what `detect()` does with settings --------------------------------------
@pytest.fixture
def fake_mpi(tmp_path):
    (tmp_path / "mpi.h").write_text("/* enough to satisfy the check */\n")
    return str(tmp_path)


def test_an_explicit_arch_flag_is_passed_through_verbatim(fake_mpi, monkeypatch):
    """The point of the knob: someone compiling on a login node for a whole
    cluster says `-march=x86-64-v3`, and nothing here is clever about it."""
    monkeypatch.delenv("CXX", raising=False)
    chain = toolchain.detect({"arch": "-march=x86-64-v3", "mpi_include": fake_mpi})
    assert chain.arch_flags == ("-march=x86-64-v3",)


def test_arch_none_means_none(fake_mpi):
    chain = toolchain.detect({"arch": "none", "mpi_include": fake_mpi})
    assert chain.arch_flags == ()


def test_extra_flags_take_a_list_or_a_string(fake_mpi):
    a = toolchain.detect({"extra_flags": ["-fopenmp", "-g"], "mpi_include": fake_mpi})
    b = toolchain.detect({"extra_flags": "-fopenmp -g", "mpi_include": fake_mpi})
    assert a.extra_flags == b.extra_flags == ("-fopenmp", "-g")


def test_the_environment_can_add_flags_without_a_config_file(fake_mpi, monkeypatch):
    monkeypatch.setenv("LAMMPS_LIVE_CXXFLAGS", "-DDEBUG_MESOMEM -g")
    chain = toolchain.detect({"mpi_include": fake_mpi})
    assert chain.extra_flags[-2:] == ("-DDEBUG_MESOMEM", "-g")


def test_a_compiler_that_is_not_there_says_so_by_name(fake_mpi):
    with pytest.raises(ToolchainError, match="not-a-compiler"):
        toolchain.detect({"compiler": "not-a-compiler-42", "mpi_include": fake_mpi})


def test_an_mpi_include_without_mpi_h_in_it_is_caught_here(tmp_path):
    """Rather than four hundred lines into a compile log. The usual mistake is
    naming the file instead of the directory, so the message says which."""
    with pytest.raises(ToolchainError, match="CONTAINS mpi.h"):
        toolchain.detect({"mpi_include": str(tmp_path)})


def test_mpi_none_compiles_without_any_mpi_include(monkeypatch):
    """For a LAMMPS whose headers already carry everything -- not the normal case,
    but the escape hatch that means a strange build is not a dead end."""
    chain = toolchain.detect({"mpi": "none"})
    assert chain.mpi_include == "" and chain.mpi_mode == "disabled"


def test_stubs_mode_writes_a_header_next_to_the_sources(tmp_path, monkeypatch):
    """A serial LAMMPS (the Windows wheel, some conda builds) has no MPI to point
    at, and the pair style still includes mpi.h and calls MPI_Bcast. The generated
    stub is documented as matching LAMMPS' own STUBS -- and it must SAY so, since
    getting that wrong is a symbol clash at load time rather than a compile error."""
    monkeypatch.setattr(toolchain, "_mpi_from_lammps_package", lambda: ("", ""))
    chain = toolchain.detect({"mpi": "stubs"}, stub_dir_for=str(tmp_path))
    header = os.path.join(chain.mpi_include, "mpi.h")
    assert os.path.isfile(header)
    text = open(header).read()
    assert "typedef int MPI_Comm;" in text
    assert "MPI_Bcast" in text and "MPI_Allreduce" in text
    assert any("stub" in note for note in chain.notes), \
        "compiling against a guessed ABI has to be announced"


def test_an_explicit_setting_beats_a_detected_one(fake_mpi, monkeypatch):
    """Explicit before clever, everywhere: someone with two MPIs installed has to
    be able to say which, and be believed."""
    monkeypatch.setattr(toolchain, "_mpi_from_wrapper",
                        lambda: ("/should/not/be/used", "wrapper"))
    chain = toolchain.detect({"mpi_include": fake_mpi})
    assert chain.mpi_include == fake_mpi
    assert chain.mpi_mode == "configured"


# --- the two that run a real compiler ----------------------------------------
@pytest.mark.skipif(not (shutil.which("c++") or shutil.which("g++") or
                         shutil.which("clang++")),
                    reason="no C++ compiler on this machine")
def test_a_flag_is_probed_rather_than_assumed(fake_mpi):
    """`-march=native` is an x86 spelling that arm64 compilers variously accept,
    ignore or reject, and `-mcpu=native` is the reverse. Asking costs one empty
    compile and removes a matrix of compiler versions from this file."""
    cxx = shutil.which("clang++") or shutil.which("g++") or shutil.which("c++")
    assert toolchain.flag_supported(cxx, "-O2")
    assert not toolchain.flag_supported(cxx, "-fthis-flag-does-not-exist")
    # Whatever this machine is, "native" resolves to something it accepts, or to
    # nothing at all -- never to a flag that fails the build.
    for flag in toolchain.detect({"arch": "native", "mpi_include": fake_mpi}).arch_flags:
        assert toolchain.flag_supported(cxx, flag), flag


def test_the_real_pair_style_compiles_and_records_how(tmp_path):
    """End to end on this machine: the MesoMem sources, the detected toolchain, an
    artifact with a platform tag, and a sidecar saying what produced it."""
    pytest.importorskip("lammps")
    from lammps_live.forcefields.mesomem import MESOMEM_PLUGIN as spec

    path, what = plugin.build(spec)
    assert os.path.isfile(path)
    assert what in ("up to date", "no compiled library yet",
                    "the build settings changed") or "newer" in what
    with open(spec.info_path) as handle:
        recorded = json.load(handle)
    assert recorded["signature"] == toolchain.detect(
        stub_dir_for=spec.directory).signature()
    assert any("mesomem" in part for part in recorded["command"])
