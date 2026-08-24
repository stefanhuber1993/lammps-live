# Installing it, on whichever machine you have

The app needs three things on the machine you sit at: a LAMMPS with a **Python
module**, a **C++ compiler** (the MesoMem pair style is compiled here, on first
launch), and the MPI **headers** that LAMMPS was built against. Everything else is
pip's problem.

The one command that tells you where you stand:

```bash
lammps-live --doctor
```

It prints the platform, the LAMMPS module, the compiler and architecture flags it
resolved, where it found `mpi.h`, which config file is in force, and the resolved
remote target of every remote scene. When something does not work, that output is
the thing to paste into a message.

---

## macOS

```bash
brew install mpich git            # hidapi too, for the joystick
python3 -m venv venv && source venv/bin/activate
pip install -e .
lammps-live --doctor              # then: lammps-live
```

MPICH and **not** Open MPI: the `lammps` wheel links against MPICH, and the pair
style is compiled against whichever MPI's headers it finds. Two MPIs installed at
once is the classic way to get a plugin that compiles and then will not load.

Xcode's command line tools supply `clang++` (`xcode-select --install` if
`--doctor` says no compiler). Apple silicon and Intel both work; the compiled
library is named per platform (`mesomem-darwin-arm64.dylib`), so a shared
directory between two Macs is fine.

## Linux

```bash
sudo apt install build-essential mpich libmpich-dev git   # Debian / Ubuntu
sudo dnf install gcc-c++ mpich mpich-devel git            # Fedora / RHEL
python3 -m venv venv && source venv/bin/activate
pip install -e .
lammps-live --doctor
```

Three things are different from macOS and all three are covered by `--doctor`:

- **Which MPI.** The pip wheel wants MPICH's headers. A distribution's own LAMMPS
  package is usually built against Open MPI instead — if you use *that* LAMMPS
  (`pip install -e . --no-deps`, plus the distro `python3-lammps`), then the
  matching headers are Open MPI's, and `--doctor` will show which it picked.
  Force the choice when both are installed:

  ```toml
  [build]
  mpi_include = "/usr/lib/x86_64-linux-gnu/openmpi/include"
  ```

  On Fedora, `mpich` and `openmpi` live behind `module load mpi/mpich-x86_64`,
  which is what puts `mpicxx` on PATH; the detection asks that wrapper first.
- **The GPU renderer** needs an OpenGL 3.3 core context. It falls back to the CPU
  renderer by itself if it cannot get one (a headless box, a VM without GL), and
  says so in the terminal.
- **The joystick** needs a udev rule so `/dev/hidraw*` is reachable without root:

  ```bash
  echo 'SUBSYSTEM=="hidraw", ATTRS{idVendor}=="045e", ATTRS{idProduct}=="001b", TAG+="uaccess"' \
    | sudo tee /etc/udev/rules.d/99-sidewinder-ff2.rules
  sudo udevadm control --reload-rules && sudo udevadm trigger
  ```

## Windows

**WSL2 is the path that works with no surprises**: install Ubuntu, then follow
the Linux instructions inside it unchanged. `wslg` gives the window, and recent
WSL2 forwards the GPU for OpenGL. A force-feedback joystick over USB does not
pass through, so use `--input mouse` there.

Native Windows is supported by the build machinery — the compiler flags, the
output name and the link step all switch to MSVC's spelling — with two caveats
that are properties of the platform, not of this app:

- **A plugin DLL cannot leave LAMMPS' symbols undefined.** ELF and Mach-O resolve
  them when the host process loads the library; PE has to import them by name at
  link time. So the build links against `liblammps.lib` next to the `lammps`
  Python module, and a LAMMPS built without `-DBUILD_SHARED_LIBS=yes` does not
  have one. `--doctor` says so rather than failing in the linker.
- **A Windows LAMMPS is usually serial**, built with LAMMPS' own MPI stubs and
  shipping no `mpi.h`. Point the build at the stubs:

  ```toml
  [build]
  mpi = "stubs"
  arch = "none"       # MSVC has no -march=native; /arch:AVX2 if you want one
  ```

Run from a *Developer Command Prompt* (or *x64 Native Tools* prompt) so `cl.exe`
is on PATH, then `pip install -e .` and `lammps-live --doctor`.

## conda / mamba

```bash
conda create -n lammps-live -c conda-forge python=3.12 lammps mpich numpy
conda activate lammps-live
pip install -e . --no-deps
pip install pygame moderngl "sidewinder @ git+https://github.com/stefanhuber1993/sidewinder.git"
```

`$CONDA_PREFIX/include` is searched for `mpi.h` before the system locations, so a
conda LAMMPS and a conda MPI find each other. Mixing a conda LAMMPS with a system
MPI is the one combination to avoid.

## Your own LAMMPS build

Nothing requires the wheel. Any LAMMPS with a shared library and the Python module
works — point `PYTHONPATH` at it (or `make install-python`) and the app will use
it, headers and all:

```bash
cmake -B build -S cmake -DBUILD_SHARED_LIBS=yes -DCMAKE_BUILD_TYPE=Release \
      -DPKG_MOLECULE=yes -DPKG_PLUGIN=yes
cmake --build build -j && make install-python
```

`-DPKG_PLUGIN=yes` is the one that matters here: it is what makes `plugin load`
available, which is how the MesoMem style gets in without rebuilding LAMMPS.
(`-DPKG_PYTHON` is a *different* feature — Python *inside* LAMMPS — and is not
needed.)

---

## Compiling the pair style

The MesoMem force field is the authors' C++ ([arXiv:2602.24123]), compiled here
into a runtime LAMMPS plugin the first time a 3D scene opens. It takes about ten
seconds, once, and there is nothing to download.

[arXiv:2602.24123]: https://arxiv.org/abs/2602.24123

```bash
lammps-live --build-plugin      # compile it now, and print the command
lammps-live --rebuild-plugin    # compile it again even if it looks current
```

The artifact lands next to the sources in
`lammps_live/forcefields/mesomem_ff/`, named for the platform it was built on
(`mesomem-linux-x86_64.so`, `mesomem-darwin-arm64.dylib`), with a small
`.build.json` beside it recording the exact command. Two things make it stale: a
source file newer than the library, and a change to any of the settings below —
so turning a flag on really does rebuild, rather than quietly returning the old
library.

### The `[build]` table

All of it is optional. See [the config file](#the-config-file) for where this
lives.

| key | default | what it does |
|---|---|---|
| `compiler` | `$CXX`, else `clang++`/`g++` (`cl` on Windows) | the C++ compiler, by name or full path |
| `std` | `c++17` | language standard |
| `optimize` | `-O3` (`/O2` on MSVC) | optimization flag; `"none"` drops it entirely |
| `arch` | `"native"` | `"native"`, `"none"`, or a literal flag such as `-march=x86-64-v3` |
| `extra_flags` | — | anything else, as a list or a string |
| `link_flags` | platform default | added to the link |
| `mpi` | `"auto"` | `"auto"`, `"stubs"` (serial LAMMPS), `"none"` |
| `mpi_include` | detected | the directory **containing** `mpi.h` |
| `verbose` | `false` | echo the compile command |

`arch = "native"` is the default because this library is compiled *on* the machine
that loads it, so the usual reason not to tune for the local CPU does not apply.
It is **probed, not assumed**: an empty file is compiled with `-march=native`,
then `-mcpu=native`, and whichever the compiler accepts is used — on a compiler
that takes neither, the build simply happens without it. Set `arch = "none"` when
the machine that compiles is not the machine that runs (a shared home directory
across a heterogeneous cluster), or a specific ISA level when you know it.

Platform overlays let one file cover several machines:

```toml
[build]
arch = "native"

[build.linux]
compiler = "g++"
extra_flags = ["-fno-math-errno"]

[build.windows]
compiler = "cl"
mpi = "stubs"
arch = "none"
```

### The same knobs from the environment

For a one-off, without touching a file:

```bash
CXX=g++-14 lammps-live --rebuild-plugin
LAMMPS_LIVE_BUILD_ARCH=none lammps-live --build-plugin
LAMMPS_LIVE_CXXFLAGS="-g -DDEBUG_MESOMEM" lammps-live --rebuild-plugin
LAMMPS_LIVE_MPI_INCLUDE=/usr/include/mpich lammps-live --doctor
LAMMPS_LIVE_BUILD_VERBOSE=1 lammps-live
```

---

## The config file

One TOML file carries the two things this repo cannot know about your machine:
where you run the remote scenes, and how to compile here.

```bash
lammps-live --write-config       # ~/.config/lammps-live/config.toml, commented
$EDITOR ~/.config/lammps-live/config.toml
lammps-live --doctor             # check what it made of it
```

Searched in order, first hit wins:

1. `$LAMMPS_LIVE_CONFIG`, or `--config PATH`
2. `./lammps-live.toml` — next to a checkout, for per-project settings
3. `$XDG_CONFIG_HOME/lammps-live/config.toml`
4. `~/.config/lammps-live/config.toml`
5. `~/.lammps-live.toml`

A typo is reported rather than ignored (`[remote] unknown key 'usr' -- did you
mean user?`), because a key that silently does nothing shows up twenty seconds
later as an SSH prompt for the wrong account.

The `[remote]` half is [docs/cluster-setup.md](cluster-setup.md). There is also a
small `[app]` table for personal defaults, which any flag overrides:

```toml
[app]
input = "joystick"
playground = "mesomem_sheet"
ui_scale = 1.5
fullscreen = false
```

---

## When it does not work

| what you see | what it is | fix |
|---|---|---|
| `Could not locate mpi.h` | no MPI development headers | `brew install mpich` / `apt install libmpich-dev` / `dnf install mpich-devel`, or set `[build] mpi_include` |
| plugin compiles, `plugin load` fails with undefined symbols | compiled against a *different* MPI than liblammps links | make `--doctor`'s "mpi headers" line match your LAMMPS' MPI |
| `LAMMPS headers not found at .../include/lammps` | a LAMMPS without its development headers | use the pip wheel, or build with `-DBUILD_SHARED_LIBS=yes` and `make install-python` |
| `No C++ compiler found` | no toolchain | `xcode-select --install`, `apt install build-essential`, `dnf install gcc-c++`, or MSVC Build Tools |
| `Unrecognized pair style 'mesomem'` | LAMMPS without the PLUGIN package | rebuild with `-DPKG_PLUGIN=yes`, or use the pip wheel |
| the scene opens on the CPU renderer | no OpenGL 3.3 core context | expected in VMs and over plain X11 forwarding; it still runs |
| joystick not found on Linux | `/dev/hidraw*` permissions | the udev rule above; `lammps-live --calibrate` to check |
| a flag in the config does nothing | it went in the wrong table | `lammps-live --doctor` prints every key it actually read |

For the remote scenes, everything above is about *this* machine. The cluster has
its own prerequisites, and they are not installed for you:
[docs/cluster-setup.md](cluster-setup.md).
