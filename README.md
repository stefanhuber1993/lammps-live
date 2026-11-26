# Feel the force

Experience molecular dynamics with a force feedback joystick.

**Stefan Huber**<sup>1</sup>, **Pietro Sillano**<sup>2</sup>, **Timon Idema**<sup>2</sup>

<sup>1</sup> Visualisation & XR Team, SURF, Amsterdam<br>
<sup>2</sup> Kavli Institute of Nanoscience Delft, Department of Bionanoscience, TU Delft

In molecular dynamics simulations, forces and torques result in particle
motion. The resulting dynamics are usually visualized as movies or plots,
something to watch rather than to play with. In this interactive demo you can
not only control the simulation, but also experience what happens in it in an
immersive manner, feeling the forces at work: a live LAMMPS simulation runs under
a 60 fps game loop, you push one bead around with a 1998 joystick, and the
joystick pushes back with the force the membrane puts on that bead.

![Seven MesoMem beads](docs/images/mesomem_patch.png)

MesoMem ([Sillano, Marrink & Idema, Phys. Rev. E 2026](https://journals.aps.org/pre/abstract/10.1103/4dhv-8xd7),
[arXiv:2602.24123](https://arxiv.org/abs/2602.24123)) is a solvent-free coarse-grained model of a biological
lipid-bilayer membrane. A patch of many lipids becomes one anisotropic bead, so
the simulation can reach the length and time scales on which membranes
self-assemble and remodel. That economy also makes it fast enough to run live.
Each bead carries a director (the yellow/blue banding), and there are three
energy terms: van der Waals, tilt and splay. The app uses the authors' own C++
pair style, compiled into LAMMPS as a plugin.

Presented as a poster and live demo at NWO Biophysics 2026, next to the MesoMem
poster by Sillano et al.

[![The poster](docs/images/poster.jpg)](docs/poster/feel-the-force-poster2_print.pdf)

## Install

On your laptop:

```bash
brew install hidapi git           # Linux: apt install build-essential mpich libmpich-dev libhidapi-hidraw0
python3 -m venv venv && source venv/bin/activate
pip install -e .
lammps-live --doctor              # check what this machine resolved to
```

Any Python 3.9 or newer works, including the one that comes with macOS. MPI
comes from pip, not Homebrew: the `lammps` wheel links MPICH and looks for it
only inside the venv, so `pip install -e .` installs `lammps[mpi]`, which puts
MPICH there. Don't install Open MPI next to it. The MesoMem plugin compiles itself
the first time you open a 3D scene (about 10 seconds), against the same MPICH.
`brew install hidapi` is only for the joystick.
On Linux the joystick needs a udev rule, see [docs/install.md](docs/install.md).
Scenes 1 to 6 need nothing else.

## Install for the Snellius scenes

Only needed for scenes 7 and 8, which run on a GPU on Snellius. Skip this if you
only use the laptop scenes.

**1. What you need on Snellius.** A login that can SSH to `snellius.surf.nl`, and
a budget on the GPU partition (`accinfo` on the login node shows your account
names). You also need a LAMMPS build with the Python module, numpy, the MesoMem
pair style and Kokkos+CUDA. The app does not build LAMMPS. It copies itself to
`~/.lammps_live_remote` on every connect, so you don't install the app on the
cluster. The build is the only thing you set up there.

**2. Get a LAMMPS build.** You can use one of two:

- *An existing shared build (easiest).* If someone has put a build where your
  account can read it, point `remote_dir` at it. Nothing goes in your home
  directory. The owner can grant read access to their own home with an ACL
  ([SURF: ACLs](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660238/Access+Control+Lists+ACLs)):
  `setfacl -m u:<login>:--x` on each parent directory, then
  `setfacl -R -m u:<login>:rX,d:u:<login>:rX` on the build. A project space under
  `/project/<name>` also works.
- *Your own build.* Build LAMMPS with the MesoMem files and write an `env.sh` that
  loads the modules and sets `PYTHONPATH` and `LD_LIBRARY_PATH`. A Kokkos+CUDA
  compile takes a while. See [docs/cluster-setup.md §3](docs/cluster-setup.md#3-if-you-have-to-build-lammps-there).

**3. Write the config file on your laptop:**

```bash
lammps-live --write-config       # writes ~/.config/lammps-live/config.toml
```

Then edit these lines in it:

```toml
[remote.systems.snellius]
host = "snellius.surf.nl"
user = "your-login"
partition = "gpu_a100"
account = "your-budget"          # from `accinfo`
remote_dir = "/project/<name>/lammps-mesomem"   # the build from step 2
env_script = "env.sh"            # relative to remote_dir, sourced before the server starts
profile = "cluster-gpu"
```

**4. Check it:** `lammps-live --doctor` shows the target it resolved and the
probe command. More detail: [docs/cluster-setup.md](docs/cluster-setup.md) and
[docs/snellius/README.md](docs/snellius/README.md).

## Run it

```bash
lammps-live --input joystick      # or --input mouse; starts at scene 1
lammps-live --list                # every scene, by name
lammps-live --playground mesomem_sheet   # start at one scene
```

Trigger (or Space) starts and stops the simulation, buttons 3/4 go to the
previous/next scene. For the Snellius scenes, go to scene 7, press `N`, then
Connect. The first Connect asks for your password and one-time code in the
panel. It checks the build on the login node before it asks Slurm for a GPU, so a
wrong path fails within seconds and costs no budget.

At a demo:

```bash
# once beforehand: the whole remote path, from the terminal
python -m lammps_live.remote.session --playground mesomem_remote --play --seconds 30

lammps-live --input joystick --fullscreen --lock --idle-return
```

`--lock` is kiosk mode (closing or leaving fullscreen needs a password),
`--idle-return` goes back to scene 1 after a minute without input.

For a chemistry audience, add `--with-copper`. It puts a ninth scene after the
eight: a 2D copper atom deposited on a copper crystal, with the real EAM
potential (key `9`).

## The game loop

LAMMPS runs inside Python as a library and shares its memory with it. Every
frame the joystick sets one extra force on the controlled bead, and the
simulation advances a short burst of steps.

```python
from lammps import lammps

lmp = lammps()                                       # LAMMPS as a library
lmp.file("in.membrane")                              # an ordinary input deck
lmp.command("variable fx internal 0.0")              # slots Python can write
lmp.command("variable fy internal 0.0")
lmp.command("fix drive probe addforce v_fx v_fy 0")  # push the probe bead

while running:
    fx, fy = hand_force()                            # from the stick
    lmp.set_internal_variable("fx", fx)              # not a new fix, so no re-setup
    lmp.set_internal_variable("fy", fy)
    lmp.command("run 20 pre no post no")             # advance 20 steps
    x = lmp.numpy.extract_atom("x")                  # positions, no copy
    draw(x)
```

The real version is `lammps_live/playground/modes.py` (the drive) and
`lammps_live/playground/system.py` (the stepping). The run happens in a
background thread, so the next 20 steps compute while the frame draws.

## The joystick

A Microsoft Sidewinder Force Feedback 2, run by our own Python driver,
[sidewinder](https://github.com/stefanhuber1993/sidewinder). Python sets a spring
about 60 times a second. The spring itself runs on the stick's own chip, which
pushes back much faster than that. The force from the membrane moves the
spring's centre and sets how stiff it is, so you feel a pull towards wherever the
membrane wants the bead to go.

```python
from sidewinder.ff2 import FF2Device

ff = FF2Device()
spring = ff.spring(stiffness=1)                  # limp until something pushes

while running:
    s = ff.read_input()                          # s.x, s.y, s.twist in -1..1
    set_hand_force(gain * s.x, -gain * s.y)      # into fix addforce above
    fx, fy = force_on_probe()                    # what the membrane pushes with
    k = int(1 + 126 * contact(fx, fy))           # firm on contact
    spring.set_condition(axis=0, cp_offset=int(fx), pos_coeff=k, neg_coeff=k)
    spring.set_condition(axis=1, cp_offset=int(-fy), pos_coeff=k, neg_coeff=k)
```

On top of that there is a damper and a small vibration that grows with
temperature (`lammps_live/input/joystick.py`). The buttons: trigger is
start/stop, 2 resets, 3/4 go to the previous/next scene, the hat moves around the
GUI, 5 and up are each scene's "action" buttons, twist turns the bead's
director, and the thrust lever cuts a slice through the 3D view. Mouse and
keyboard work too (`--input mouse`).

## Eight scenes

| | | |
|---|---|---|
| 1 | `mesomem_bead` | interaction of two beads |
| 2 | `mesomem_patch` | 6+1 patch, feel the force |
| 3 | `mesomem_patch_torque` | 6+1 patch, feel the torque |
| 4 | `mesomem_sheet` | 900 beads, an infinite (periodic) membrane |
| 5 | `mesomem_assembly` | 1500 beads in a box, self-assembling |
| 6 | `mesomem_rod` | "endocytosis" of a rod, with an added rod-bead force term |
| 7 | `mesomem_remote` | 50,000 beads on Snellius, large scale self-assembly |
| 8 | `mesomem_vesicle_chain` | 56,000 beads, a membrane vesicle filled with a polymer chain |

With `--with-copper` there is a ninth, `cu_deposition`: a 2D copper atom pulled
onto a copper crystal with the EAM potential, real atoms after the coarse-grained
beads.

Temperature, `k_tilt`, `k_splay` and the other coefficients are sliders you can
move while it runs.

## How scenes 7 and 8 run on Snellius

These two run on a GPU on Snellius, the Dutch national supercomputer at SURF,
and the laptop only draws. Press `N` on either scene and the app asks Slurm for a
GPU, starts a server there, and connects to it over a TCP connection through an
SSH tunnel. Each bead is sent as 3 × 12 bit position and 2 × 8 bit orientation,
which is about 7 MB/s for 56k beads at 20 fps, little enough for conference wifi.
Both remote scenes share one allocation, so switching between them doesn't queue
again.
Setting it up is under [Install for the Snellius scenes](#install-for-the-snellius-scenes).

## Rendering

The 3D scenes draw every bead as a sphere impostor: a flat square, on which a
shader computes a perfect sphere pixel by pixel. All beads go to the GPU in one
call, so 50,000 beads cost the CPU the same as ten. Shadows and ambient occlusion
are added afterwards, at a cost per pixel, not per bead.

The same renderer runs in a browser, also on a phone:
[the impostor viewer](https://stefanhuber1993.github.io/lammps-live/). The long
explanation is in [docs/impostor-book](docs/impostor-book/).

## More

- [docs/details.md](docs/details.md): what every scene shows, the full joystick
  mapping, the remote setup in depth, kiosk mode (`--lock`), adding a scene
- [docs/install.md](docs/install.md): other platforms and compilers, Linux udev rule
- [docs/remote-gpu.md](docs/remote-gpu.md): how the remote connection works
- [docs/FAQ.md](docs/FAQ.md): short answers to questions people ask at the poster

Contact: visualization@surf.nl
