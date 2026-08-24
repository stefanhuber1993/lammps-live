# Running the MesoMem demo on Snellius

**This file is Snellius-specific.** For running the live demo on *any* cluster --
what the far-side LAMMPS has to be, how to check a candidate build, and the config
file that carries your login, account and paths -- see
[../cluster-setup.md](../cluster-setup.md). Everything below assumes the build that
already exists here.

Decks for the scale-up in [../a100-plan.md](../a100-plan.md). Same physics as
`lammps_live/playgrounds/mesomem_assembly.py`, written out as plain LAMMPS input
and sized for 100k beads instead of 1500.

| file | what it is |
|---|---|
| `in.mesomem_100k` | the demo system: 100k beads, phi = 0.1, L = 80.6 sigma, Langevin. Bulk throughput **and** the 20-step frame cadence. |
| `in.mesomem_validate` | correctness gate. NVE, no thermostat, prints energies that must match digit-for-digit across CPU rank counts, and to ~1e-12 against the GPU. |
| `run_snellius.sh` | runs the same deck on 1 core, all cores, and one MIG slice, then greps out the numbers that matter. |

For the **live** demo -- the app drawing a cluster GPU's simulation in real time --
skip to [the live demo](#the-live-demo-mesomem_remote) below; it uses no deck from
this directory. How it works, and every decision in it, is
[../remote-gpu.md](../remote-gpu.md).

Both decks take `-var nbeads N` (and `-var phi`, `-var ktilt`, ...), so the whole
size sweep is one loop and there is nothing to edit.

## The live demo: `mesomem_remote`

The decks above are for benchmarking. The demo itself is a playground in the app --
`mesomem_remote`, 10,000 beads -- and it needs no deck at all: the server on the
node builds the same `Playground` file the client draws, so there is exactly one
definition of the experiment. Select it in the app (key 4, or Tab), press **N**, and
press Connect.

The login and the paths come from your config file rather than from the playground
source (`lammps-live --write-config`, then `--doctor` to check it); the values in
`mesomem_remote.py` are only what this machine happened to need first.

`mesomem_polymer` is the second playground on the same target and reaches the node
by exactly the same route -- a closed vesicle with a melt of ring polymers sealed
inside it, ~50,000 particles, which is the collaborator's `polymer/` system built
from the same `Playground` file rather than from a data file. It asks two things of
the node's LAMMPS that `mesomem_remote` does not, both of which the in-tree build
already has: the MOLECULE package (`bond_style fene`, `angle_style cosine`), and an
atom style carrying the molecule and bond fields -- which is what
`dipole_sphere_angle` is, and why it is named that.

What the panel then does, and roughly how long each step takes:

| step | what runs | typical |
|---|---|---|
| login | one SSH `ControlMaster` -- prompts appear in the panel | you type it |
| deploy | `tar` of `lammps_live/` over that connection into `~/.lammps_live_remote` | ~2 s |
| probe | `python -m lammps_live.remote.probe --level light` on the LOGIN node: interpreter, numpy, the module files | ~20 s |
| allocate | `salloc --no-shell`, then `squeue` until the node appears -- the job id comes off Slurm's `Pending` line, so the wait is polled and reported rather than blocked on | queue, up to an hour |
| check | `srun --jobid=N ... probe --level full` on the NODE: opens liblammps, reads the styles and the `pair_coeff` arity | ~15 s |
| launch | `srun --jobid=N python -m lammps_live.remote.server` | ~10 s |
| tunnel | a second `ssh` that ENDS on the node, jumping through the login node over the connection that is already authenticated | instant |

**One login, one code.** Every step after the first rides the same authenticated
connection -- including the tunnel's jump hop, via
`ProxyCommand ssh -S <ctl> -W %h:%p` -- so the one-time code is typed once per
session and not once per step. (The second hop, to the node itself, does not prompt:
Slurm's PAM stack admits the owner of a running job.)

**The tunnel ends on the compute node, not on the login node.** So the login node
relays a stream it cannot read, the frames are encrypted the whole way, and the
server listens on `127.0.0.1` -- the port does not exist for the rest of the
cluster. `RemoteTarget(tunnel="forward")`, or
`LAMMPS_LIVE_REMOTE_TUNNEL=forward`, falls back to a one-hop `-L` forward for a site
that does not allow ssh to a compute node; the server's bind address follows the
choice automatically. The full comparison is in
[../remote-gpu.md](../remote-gpu.md#2-the-networking-from-the-bottom).

**The allocation is released three ways**, because a forgotten A100 is the failure
with a bill attached: the app cancels the job when the window closes, the server
runs `scancel $SLURM_JOB_ID` when it exits (including after `--exit-when-idle`,
default 15 minutes, which covers a hard-killed app), and Slurm's `--time` ends it
regardless.

**How long to ask for.** One hour by default, and `--gpu-hours` changes it:

```
lammps-live --playground mesomem_remote --gpu-hours 3
```

It becomes the `--time=HH:MM:SS` on `salloc`, applies to every remote playground in
the session, and overrides both the playground's own declaration and
`LAMMPS_LIVE_REMOTE_TIME`. Two reasons not to pad it: it is the outermost of the
three releases above -- the one that still works when the app has been killed and
the network has gone -- and a longer request can sit in the queue longer.

### Check the build first, in one command

The one hard requirement is that this build has a **Python module**, not just an
`lmp` binary -- a frame-by-frame demo cannot be driven any other way. On the login
node, where numpy and the module files can be checked but the library cannot be
opened (see below):

```
PYTHONPATH=~/.lammps_live_remote python3 -m lammps_live.remote.probe --level light
```

and inside an allocation, for the real thing:

```
srun --jobid=<id> --ntasks=1 --gpus=1 bash -lc '
  cd ~/Projects/MesoMemLive/mesomem_gpu && source _build/hpc/env.sh &&
  PYTHONPATH=~/.lammps_live_remote python3 -m lammps_live.remote.probe'
```

**Why two.** A Kokkos/CUDA `liblammps.so` links against `libcuda.so.1`, the NVIDIA
driver, which exists on GPU nodes and nowhere else. `import lammps` therefore works
on a login node while opening the library there fails with
`OSError: libcuda.so.1: cannot open shared object file` -- which says nothing about
the build being wrong. The connect flow does both automatically, in that order.

It reports whether `import lammps` works, whether `mesomem`, `mesomem/kk`,
`dipole_sphere_angle` and `nve/sphere/kk` are registered, and whether `pair_coeff`
takes 8 or 9 values (the client is told, so the coefficients it sends match what
this build accepts). If the module is missing, rebuild with
`-DBUILD_SHARED_LIBS=yes` and `make install-python` (`-DPKG_PYTHON` is a different
feature -- Python *inside* LAMMPS -- and is not what this needs). The connect
flow runs this itself and refuses to allocate a GPU for a build that cannot serve.

### Debugging the SSH and Slurm half without the GUI

```
python -m lammps_live.remote.session --playground mesomem_remote --play --seconds 30
```

Same flow, prompts on the terminal, one line per received frame (size, rate, RTT),
then it tears the allocation down. This is the thing to run when a step fails.

### Without a cluster at all

The server runs on this laptop, on the CPU, over the loopback -- which is how the
client, the codec and the whole control channel are tested:

```
python -m lammps_live.remote.server --playground mesomem_remote --profile local \
       --token dev --port 5723
lammps-live --playground mesomem_remote --remote 127.0.0.1:5723 --token dev
```

### What the wire carries

Positions as 3 x 12 bits over the cell (packed two codes to three bytes), directors
octahedral-8, per-bead energies as uint8 and only while the energy colouring is
switched on: 6.5 B/bead, so 65 kB and 3.9 MB/s at 10k and 60 fps. Control goes the other way as JSON -- one message per
slider change, which is the same LAMMPS command the local app issues on itself.
Details and the measured precision in `lammps_live/remote/protocol.py`; the
bandwidth table it implements is §5 of ../a100-plan.md.

### Why 10,000 and not 100,000

Because 10k is what the *drawing* machine can keep up with, not what the GPU can
run. The Python analysis measures 6.8 ms/frame average and 31.7 ms peak at 10k
(§3 of the plan, re-measured), against a 16.7 ms frame; the A100 at that size is
idling. 10k is therefore the right size to prove the pipeline before making the
analysis cheap enough for the size the hardware could actually do.

## The §1 gate has passed

**The pair style is ported and running on the device.** `PairMesomemKokkos` in
`mesomem_gpu/cpp_files/pair_mesomem_kokkos.cpp` registers as `mesomem/kk`, and
the neighbour-list line confirms it is genuinely on the GPU rather than
suffix-resolved back to the host:

```
pair mesomem/kk, perpetual / attributes: full, newton off, kokkos_device
```

`atom_vec_dipole_sphere_angle_kokkos.cpp` is there too, which is the bigger deal:
with the atom vector itself on device, the whole class of per-step host↔device
copies that the plan warned about does not arise. `fix_langevin_kokkos.cpp`
covers the bath.

So the feasibility question is answered, and the remaining work is not "does this
run on a GPU". These four are what is left:

1. **Do not quote the 526× — the `Pair` line is a timer artifact.** Measured at
   40k atoms on a MIG slice: `Pair 10.123s` CPU vs `0.019226s` GPU. That ratio is
   above the hardware's ceiling, so it is not measuring the kernel. Kokkos
   launches are asynchronous and LAMMPS defaults to `timer nosync`, so a section
   timer can stamp the *launch* and let the actual work land in whichever later
   section next forces a fence — usually `Comm` or `Other`.

   The arithmetic, with `S` = steps in the timed section and ~24 full-list
   neighbours per bead: `40000 × 24 × S / 0.019226` pair interactions/s, times
   ~150 FP64 flops each. At any S in the 500–2000 range that lands between 3 and
   14 TFLOP/s — versus ~9.7 TFLOP/s FP64 for a *whole* A100 and ~1.4 for a `1g`
   slice. Over peak by 20–50×, and FP32 (2.8 TFLOP/s on a slice) does not rescue
   it either.

   Fix: add `timer sync` to the deck, or just stop reading section timers and
   read **`Loop time` and `Performance`**. Total wall clock cannot be misattributed.
   The speedup is real and probably large; 526× is not its size.

2. **Is `fix nve/sphere update dipole` on device?** This is now the prime suspect,
   because it is the one style in the deck you did not list a Kokkos file for.
   Upstream `FixNVESphereKokkos` exists but its coverage of the `update dipole`
   extra term is the open question — and if `-sf kk` quietly leaves it as the host
   fix, the integrator pulls positions, dipoles and torques back every single
   step, which puts the sync cost straight back after the pair kernel removed it.
   Grep the log for whether the fix resolved to `nve/sphere/kk`, and for any
   `not supported by Kokkos` line.

3. **Leave the neighbour list alone: `full, newton off` is correct.** The
   `-pk kokkos newton on neigh half` flag in earlier versions of
   `run_snellius.sh` was written for a host-side pair style and is now actively
   wrong; it has been removed. Two consequences worth knowing:
   - A full list means every pair is evaluated twice. On the GPU that is the
     right trade (no force write conflicts, no atomics), so do not try to
     "optimise" it back to half.
   - It also changes the force summation order, so **CPU and GPU results are not
     bit-identical** and should not be expected to be. Rank-to-rank on the CPU
     still is. See the note on `in.mesomem_validate` below.

4. **Find out where the bottleneck went.** With pair no longer dominant, re-read
   the breakdown for what now is: `Comm` (MPI + any residual device staging),
   `Neigh` (rebuilt every step under `check yes`, and the full list is bigger),
   `Modify` (the fixes), `Other`. Then look past LAMMPS entirely — §3 of the plan
   measures the Python `Analysis.update` at 1.5 µs/bead/chunk, which caps N near
   10k on its own no matter how fast the kernel is. That, and the wire format in
   §5, are now the whole demo.

`run_snellius.sh` greps for all of this.

### MIG caveat

A MIG slice is a fraction of the A100 — a `1g` profile is roughly a seventh of
the SMs — and MIG instances cannot be ganged together for one job. That cuts both
ways now:

- A slice **understates** the full card by roughly 7×, so a slice that already
  clears the frame budget is a comfortable result, not a marginal one.
- A slice also has a much lower roofline, which is what makes the arithmetic in
  point 1 above conclusive. Do not use a slice number to argue the kernel is
  faster than the hardware allows.

Use MIG for correctness, for the `update dipole` question, and for shaking out
the launch path. Request a full A100 for any number you intend to quote.

## The CPU path is now genuinely the fallback

This section used to argue for measuring CPU first, on the reasoning that a
55× speedup was needed and a full node could plausibly supply it without any
kernel work. With `mesomem/kk` on the device that argument is moot — the GPU
path exists, works, and by any honest reading of the timings is far ahead. Run
the CPU columns anyway, for three reasons that are not about speed:

1. **The 1-core column is the denominator.** Every speedup claim needs it, and
   it is the only way to sanity-check point 1 above. Reference: 0.45 µs/bead/step
   on one M1 P-core with this deck, 76% of it pair.
2. **The CPU is the correctness reference.** `in.mesomem_validate` gates the port
   against it, and the pair style is decomposition-exact on the CPU (bit-identical
   across 1/2/4/8 ranks under NVE), which is what makes it a trustworthy
   reference at all.
3. **It is the answer when there is no GPU to allocate.** A demo that needs a
   free A100 has a scheduling dependency; a 128- or 192-core node is easier to
   get. At plausible parallel efficiency a full node lands near the 1200 steps/s
   that 100k at 60 fps needs — tight, but a working degraded mode rather than
   nothing. Per-core throughput on Genoa is below an M1 P-core, so budget for it.

The one thing neither path gets for free is the app's data model: the live app
gathers positions and directors by atom id every frame, which is what
`atom_modify map array` in the decks is for. Under Kokkos that gather is also the
one place a host↔device copy is genuinely unavoidable — it is the frame payload,
so it has to reach the host. Budget it as part of §5's transfer cost, not as a
port defect, and keep the map on when benchmarking because the real run pays for it.

## Sizing

`L` is derived, not hard-coded: `L = (N·(π/6)/φ)^(1/3)`.

| N | L (σ) at φ = 0.1 | number density |
|---|---|---|
| 1,500 (paper) | 19.878 | 0.191 |
| 25,000 | 50.775 | 0.191 |
| 100,000 | 80.600 | 0.191 |

The app hard-codes 20.0 for the 1500-bead run, which is the same cell to three
digits (φ = 0.0982).

φ = 0.1 is the paper's condition, so the coarsening timeline carries over
unchanged: patches by t ≈ 500 τ, large lamellae by t ≈ 2000 τ. At timestep 0.01
and 20 steps per frame that is 0.2 τ/frame — 12 τ/s at 60 fps, so ~2.8 minutes of
wall clock to the large-lamellae regime. Scaling up buys **more independent
membranes in frame**, not faster coarsening.

## What the decks leave out

- **`k_upright` and `center_accel`.** The two watchability nudges in
  `RandomFill` are per-frame numpy momentum kicks applied by the client, not
  LAMMPS fixes, so there is no line for them here. Neither changes the assembly
  (measured) and neither affects timing.
- **`splay_symmetry`.** This repo's pair style takes an optional 10th
  `pair_coeff` argument the authors' original does not. The decks pass 8 values
  after `* *`, which both versions accept, and the patched one defaults it to 0 —
  the paper's signed splay term.
- **The Python analysis.** §3 of the plan is the real ceiling and none of it is
  in these decks. A green light here says the *sim* fits; it says nothing about
  `Analysis.update` at 1.5 µs/bead/chunk, which caps N near 10k on its own. In the
  live demo that analysis runs on the CLIENT, not here -- the server does none of
  it, which is why its frame budget is only the integration and the send.
