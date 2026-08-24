# Running the remote scenes on *your* cluster

`mesomem_remote` and `mesomem_polymer` put the simulation on a cluster GPU and
keep the picture on your laptop at 60 fps. Two things have to be true before that
can work, and only one of them is this repo's job.

> **THE LAMMPS ON THE CLUSTER HAS TO EXIST ALREADY.**
> Nothing in this app builds, installs or updates LAMMPS on the far side. The
> connect flow ships *itself* over (a tarball of `lammps_live/`, into
> `~/.lammps_live_remote`), and then imports the LAMMPS that is already there. If
> there is no such build, the connect flow refuses **before** allocating anything,
> and says which of the requirements below is missing.

The rest of this file is: what that build has to be, how to check a candidate,
how to build one if you have to, and how to point the app at your account and your
machine without editing a source file.

---

## 1. What the cluster build must have

| requirement | why | how it is checked |
|---|---|---|
| **A LAMMPS Python module** (`import lammps` works, and `liblammps.so` opens) | a 60 fps demo is driven frame by frame through the library; an `lmp` binary cannot be driven at all | `probe --level light` on the login node, then `--level full` on the node |
| **numpy** in that same interpreter | every frame's positions and directors are gathered as arrays | the same probe (scipy is *not* needed — the analysis runs on your laptop) |
| **The MesoMem pair style compiled in** (`pair_style mesomem`) | on the cluster it cannot be a runtime plugin, because the Kokkos variant cannot be one | `has_style("pair", "mesomem")` |
| **An atom style with per-atom mass, dipole and radius** — `dipole_sphere_angle`, or `hybrid sphere dipole` | each bead carries a director, a radius and (for the polymer scene) bonds | `has_style("atom", ...)`; the profile adapts the deck to whichever it finds |
| **A GPU build** — `mesomem/kk`, Kokkos + CUDA — *for the `cluster-gpu` profile* | this is what makes 10k–50k beads real-time | `has_style("pair", "mesomem/kk")` |
| **The MOLECULE package** — `bond_style fene`, `angle_style cosine` — *for `mesomem_polymer` only* | the ring polymers sealed inside the vesicle | `has_style("bond", "fene")` |
| **Slurm**, and SSH access to the login node | the allocation, and the tunnel | the connect flow's own steps |

Not required: the app installed on the cluster, a virtualenv there, a scheduler
plugin, or any input deck. Both ends build the same `Playground` file, so there is
one definition of the experiment and nothing to keep in sync.

**A GPU is not required either.** `profile = "cluster-cpu"` runs the same scenes on
the host styles of the same in-tree build, which is the honest fallback when the
GPU partition is full — and a CPU-only target asks Slurm for no GPU at all rather
than for zero of them.

## 2. Check a candidate build in two commands

The check is deliberately in two halves, because the interesting failure is
invisible on one of the two machines:

```bash
# On the LOGIN node: the interpreter, numpy, and the module files.
PYTHONPATH=~/.lammps_live_remote python3 -m lammps_live.remote.probe --level light

# Inside an allocation, on the NODE: opens the library for real.
srun --jobid=<id> --ntasks=1 --gpus=1 bash -lc '
  cd ~/your/lammps/build && source env.sh &&
  PYTHONPATH=~/.lammps_live_remote python3 -m lammps_live.remote.probe'
```

**Why two.** A Kokkos/CUDA `liblammps.so` links against `libcuda.so.1`, the NVIDIA
driver, which exists on GPU nodes and nowhere else. So `import lammps` succeeds on
a login node while *opening* the library there fails with `OSError:
libcuda.so.1: cannot open shared object file` — which says nothing at all about
the build being wrong. The connect flow runs both automatically, in that order, and
`--doctor` prints the exact command for your own paths.

The probe reports the module, numpy, the styles above, the installed packages, and
whether `pair_coeff` takes 8 or 9 values (this repo's patched style has a 9th
coefficient, `splay_symmetry`; the authors' original stops at 8 — the client is
told which, and hides the slider it cannot send).

Get the tarball there first, if you have not connected once yet:

```bash
rsync -a --exclude venv --exclude .git lammps_live \
      you@cluster:~/.lammps_live_remote/
```

## 3. If you have to build LAMMPS there

This is ordinary LAMMPS with two extra source files (the MesoMem pair style and
its atom style) dropped into `src/`. What matters for this app is the **shared
library plus Python module** and, for a GPU target, **Kokkos with the right
architecture**:

```bash
# CPU build -- enough for profile = "cluster-cpu"
cmake -B build -S cmake \
      -DBUILD_SHARED_LIBS=yes -DCMAKE_BUILD_TYPE=Release \
      -DPKG_MOLECULE=yes
cmake --build build -j 16
make install-python          # or: export PYTHONPATH=$PWD/build:$PYTHONPATH

# GPU build -- profile = "cluster-gpu"
cmake -B build-gpu -S cmake \
      -DBUILD_SHARED_LIBS=yes -DCMAKE_BUILD_TYPE=Release \
      -DPKG_KOKKOS=yes -DKokkos_ENABLE_CUDA=yes \
      -DKokkos_ARCH_AMPERE80=yes \        # A100. HOPPER90 = H100, VOLTA70 = V100
      -DPKG_MOLECULE=yes
cmake --build build-gpu -j 16 && make install-python
```

- `-DBUILD_SHARED_LIBS=yes` is the one that produces the Python module. (`-DPKG_PYTHON`
  is a *different* feature — Python *inside* LAMMPS — and is not needed.)
- Get `Kokkos_ARCH_*` wrong and the build works but runs on the host: check the
  neighbour-list line for `kokkos_device`, and see
  [snellius/README.md](snellius/README.md) for how to read the timings honestly.
- The plugin path used locally (`-DPKG_PLUGIN=yes`, compile the style on demand)
  is deliberately *not* how the cluster does it — a Kokkos style cannot be a
  runtime plugin.

Then write the `env.sh` that the app sources before starting the server. It has one
job: put that build's Python on `PATH` and its library on `LD_LIBRARY_PATH`, with
whatever module stack it needs:

```bash
# ~/lammps-mesomem/env.sh
module load 2023
module load foss/2023a CUDA/12.1.1
module load SciPy-bundle/2023.07-gfbf-2023a      # this is where numpy comes from
export PYTHONPATH=$HOME/lammps-mesomem/build-gpu:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME/lammps-mesomem/build-gpu:$LD_LIBRARY_PATH
```

The most common failure by far is that `python3` after sourcing it is **not** the
python with numpy and LAMMPS. Set `python` in the config to the exact interpreter
if the module stack does not put the right one first.

## 4. Point the app at your account

Nothing above needs a source edit. Write a config file:

```bash
lammps-live --write-config
$EDITOR ~/.config/lammps-live/config.toml
```

```toml
[remote]
system = "mycluster"

[remote.systems.mycluster]
host = "cluster.example.org"     # or a Host alias from ~/.ssh/config
user = "your-login"              # "" to let ~/.ssh/config decide
label = "MyCluster A100"         # what the Connect card calls it
partition = "gpu"
account = "prj1234"              # omit if your site does not use accounts
gpus = 1
# gres = "gpu:a100:1"            # older Slurm: replaces --gpus=N entirely
ntasks = 1
cpus_per_task = 18
time = "01:00:00"
remote_dir = "~/lammps-mesomem"  # where the cluster's LAMMPS build lives
env_script = "env.sh"            # sourced there, in a login shell
python = "python3"
profile = "cluster-gpu"          # or "cluster-cpu"
tunnel = "jump"
fps = 20.0
```

Then:

```bash
lammps-live --doctor                             # shows the resolved target
lammps-live --playground mesomem_remote          # N, then Connect
lammps-live --hpc othercluster --playground mesomem_remote   # a different one
```

### Every key, and when you need it

| key | default | when to change it |
|---|---|---|
| `host`, `user` | Snellius, unset | always. An `~/.ssh/config` alias works and is the easy way to carry a jump host, a key and a port |
| `label` | "remote GPU" | shown on the Connect card, so name it something you will recognise in a talk |
| `partition` | `gpu_a100` | always — partition names are site-specific |
| `account` | none | sites that bill against a project |
| `gpus` / `gres` | 1 / unset | `gres = "gpu:a100:1"` on Slurm older than 20.11, or where GPUs are a generic resource. `gpus = 0` with `profile = "cluster-cpu"` for a CPU-only run |
| `ntasks`, `cpus_per_task` | 1, 18 | match the node's cores per GPU |
| `time` | `01:00:00` | the wall clock, and the backstop that gives the GPU back when everything else has failed to. `--gpu-hours` overrides it per run |
| `queue_wait` | 3600 s | how long a queued request may sit before the session gives up |
| `extra_salloc` | none | anything else your site needs: `["--constraint=a100", "--exclusive"]` |
| `remote_dir`, `env_script` | Snellius paths | always |
| `python` | `python3` | when the module stack does not put the right interpreter first |
| `profile` | `cluster-gpu` | `cluster-cpu` for a host-only build; `local` only for the loopback |
| `tunnel` | `jump` | `forward` where SSH to a compute node is not allowed (see below) |
| `port`, `local_port` | 5723 | a clash with something else you run |
| `fps` | 20 | how often the far side **sends** (not the frame rate). 6.5 B/bead: 50k beads at 20 fps is 6.5 MB/s, at 60 fps it is 19.5 |
| `exit_when_idle` | 900 s | the server ends its own allocation after this much silence |

Any of them can also be set for one run through the environment
(`LAMMPS_LIVE_REMOTE_PARTITION=gpu_h100 lammps-live ...`), which wins over the
file. The full precedence is: the playground's declaration, then the config file,
then the environment.

## 5. Two things about the network

**The tunnel ends on the compute node**, not on the login node: a second SSH whose
session terminates on the node, carried through the login node as an opaque
stream. The login node relays bytes it cannot read, the frames are encrypted the
whole way, and the server listens on `127.0.0.1` — the port does not exist for the
rest of the cluster.

That needs `sshd` on the compute node and a PAM stack that admits the owner of a
running job (`pam_slurm_adopt`), which is a site policy. Where it is not allowed,
`tunnel = "forward"` falls back to a one-hop `ssh -L` through the login node; the
server then binds `0.0.0.0` and is reachable (token-protected) from the cluster's
internal network. The bind address follows the choice automatically — it is never a
separate decision. The long version is
[remote-gpu.md §2](remote-gpu.md#2-the-networking-from-the-bottom).

**One login, one code.** Every step after the first rides the same authenticated
`ControlMaster` connection, including the tunnel's jump hop, so a one-time code is
typed once per session rather than once per step.

## 6. Getting the allocation back

A forgotten A100 is the failure with a bill attached, so it is released three ways:
the app cancels the job when the window closes, the server runs
`scancel $SLURM_JOB_ID` when it exits (including after `exit_when_idle`, which
covers a hard-killed app), and Slurm's `--time` ends it regardless.

```bash
lammps-live --playground mesomem_remote --gpu-hours 3
```

`--gpu-hours` is the outermost of those three — the one that still works when the
app has been killed and the network has gone — so it is not a number to pad, and a
longer request can sit in the queue longer.

## 7. Debugging, without the GUI

```bash
# The whole SSH + Slurm + tunnel flow, prompts on the terminal, one line per frame:
python -m lammps_live.remote.session --playground mesomem_remote --play --seconds 30

# The far side's build, from here:
lammps-live --doctor        # prints the probe command with your own paths in it

# No cluster at all -- the server on this laptop, over the loopback:
python -m lammps_live.remote.server --playground mesomem_remote --profile local \
       --token dev --port 5723
lammps-live --playground mesomem_remote --remote 127.0.0.1:5723 --token dev
```

That last pair is how the client, the codec and the whole control channel are
tested, and it needs no cluster, no GPU and no account — worth running once before
blaming the network.

| symptom | what it means | fix |
|---|---|---|
| `no lammps Python module on this PYTHONPATH` | the build has no shared library / Python module | rebuild with `-DBUILD_SHARED_LIBS=yes`, `make install-python` |
| `this python has no numpy` | `env.sh` selects an interpreter without it | load a SciPy bundle in `env.sh`, or set `python` to one that has it |
| `libcuda.so.1: cannot open shared object file` **on the login node** | expected, not a fault | it is checked again on the GPU node; only a failure *there* matters |
| `Unrecognized pair style 'mesomem'` | the style is not in that build | the pair style has to be compiled in on the cluster; a plugin will not do for `mesomem/kk` |
| `Cannot set per-type atom mass for atom style dipole_sphere_angle/kk` | an old client against a per-atom-mass build | fixed by the host profile — make sure both ends are the same checkout |
| the run works and is slow | `-sf kk` fell back to host styles | check the log for `kokkos_device` on the pair line and for `not supported by Kokkos` |
| queued forever | the partition is full | `queue_wait` bounds it, Cancel ends it, and the panel reports Slurm's own reason |

---

Snellius specifics — the build that exists there, the benchmark decks and the
port's open questions — are in [snellius/README.md](snellius/README.md). The design
of the whole remote path, and every decision in it, is
[remote-gpu.md](remote-gpu.md).
