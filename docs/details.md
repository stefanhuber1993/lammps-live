# LAMMPS live: the long version

Everything the [README](../README.md) leaves out: what each scene shows on screen,
the remote GPU setup, the whole joystick mapping, installing on other machines,
kiosk mode, and how to add a scene of your own.

- [Scenes](#scenes)
- [It runs on a supercomputer](#it-runs-on-a-supercomputer)
- [The joystick](#the-joystick)
- [Run it](#run-it)
- [Adding a scene](#adding-a-scene)
- [Under the hood](#under-the-hood)

## Scenes

The eight of them are a sequence rather than a menu, in three acts, and the app
says where you are: **Rules** (what one interaction is, 1 to 3), **Material**
(what a lot of them make, and that nobody had to arrange it, 4 and 5), and
**Life** (what such a material is for, at the size the science is done at, 6 to
8). Each scene opens with its number and a descriptive title in large type across
the top, and one line saying what to do with your hands. While a scene builds,
that title is shown over a progress bar. `Tab` and buttons 3/4 stop at the ends:
there is nothing after 8 and nothing before 1.

The interface itself arrives as it becomes worth looking at (`Lesson.ui`, see
`lammps_live/ui/disclosure.py`). The two-bead scene has no side panel and no
status line at all -- two beads and the readout between them; the seven-bead
scenes add the panel with two dials and the pulled bead's energy; the sheet adds
the colour toggle and the plots; the assembly box the clock; the two cluster
scenes a SURF / Snellius badge. Whatever is new on a scene gets an amber arrow
and one line of explanation for its first few seconds. Everything hidden is one
click away under "Advanced".

A minute with nobody touching anything takes the demo back to scene 1, counting
down the last ten seconds on screen; any input cancels it. The bead colouring is
remembered per scene, and Reset puts it back to that scene's default.

Under each scene are its **hero knobs**: the one or two things worth doing to it,
on the input device's buttons `5` upward (`F1`-`F4` on the keyboard) with the
number printed on the button (`F1`-`F4` are buttons 5-8). `van der Waals
only`, on the seven-bead patch and the assembly box, sets `k_tilt` and `k_splay`
to zero and leaves the plain attraction: the same beads, and no membrane (it is
on the twist patch too, and on the assembly box it sits on button 8).
`Make the membrane curve`, on button 5 of both assembly boxes, sets the
spontaneous curvature c0 to 0.2 and the sheets close into vesicles; c0 is also
a visible slider there. `Stiff chain`, on the vesicle, takes the chain's bending
stiffness from 0 to 10. `Heat`, on the sheet, takes the temperature to 0.2 against a melting point of 0.3, which is
where the membrane stops sitting still and behaves like the liquid it is. Both
toggle, both put back the settings they found rather than the defaults, and both
say in numbers what they changed.

There is no printed key list on screen. The bindings are discoverable (the hat
moves a visible cyan frame, the trigger runs the simulation, and every button
worth pressing is drawn under the scene with its number on it), and a key list
belongs on a card next to the display rather than in a fifth of it.

| | |
|---|---|
| `mesomem_bead` | two beads: one in your hand, one nailed down at the edge of the net. Drive them together and feel the pair potential switch on, term by term |
| `mesomem_patch` | seven beads. Pull the middle one out and feel tilt and splay resist |
| `mesomem_patch_torque` | the same seven, twisted instead of pulled: the stick turns the middle bead's director and the ring splays after it |
| `mesomem_sheet` | ~900 beads, periodic, so a piece of an endless membrane. Watch a deformation spread |
| `mesomem_assembly` | 1500 beads from a random start, assembling. Play / Pause / Reset |
| `mesomem_rod` | 3600 beads at constant tension, opening warm (T = 0.2) at eps_rod = 1. Steer a rod-shaped "bacterium" in and watch the membrane engulf it -- sideways first, then a neck, then the rod standing up inside the pit. Cut it open with the thrust lever to read the profile |
| `mesomem_remote` | same thing at 50,000 beads, running on a cluster A100 |
| `mesomem_vesicle_chain` | a closed vesicle round ONE 32,768-bead polymer laid along a 3D Moore curve, coloured as a rainbow along its length, on the same A100. Slice it open with the thrust lever to see in |

(`mesomem_polymer`, the collaborator's ring-polymer melt that used to be scene 8,
is shelved but still runs by name.)

`1`-`9` or `Tab` switches between them, `lammps-live --list` prints them.

Three atomistic classics -- `cu_deposition` (copper, EAM), `lj_argon` (argon
melting) and `nacl` (a salt lattice whose ionic charge you can switch off and
watch it fall apart) -- also ship, in real units rather than reduced ones. They
are off the list above and out of the `Tab` cycle, since the talk is about the
membrane; `lammps-live --playground lj_argon` still runs one.

## It runs on a supercomputer

`mesomem_remote` and `mesomem_polymer` put the simulation on a cluster GPU -- an
A100 at [Snellius](https://www.surf.nl), or whichever machine you point them at --
and keep the picture here at 60 fps. You press
`N` and the app does the rest: asks Slurm for the GPU, ships itself over, starts
the server there, tunnels a port back, and gives the allocation up again when
you close the window. Both ends build the same scene file, so there's one
definition of the demo and no input deck to keep in sync by hand.

It behaves exactly like the local one, every slider and Play/Pause/Reset
included, because what goes over the wire is the same LAMMPS commands the local
app runs on itself. And the GPU stays yours while you wander off to show
another scene.

**One GPU, both of them.** The two remote scenes share the allocation: you ask
for a GPU once, at the start, and after that `Tab` between them costs nothing --
the run you leave keeps running, and going back to it is one socket. Pressing
Connect on the other one *moves* the GPU: the far side sets aside the simulation
it was holding and builds the other one on the same node, through the same
tunnel, with no queue and no second one-time code. So an hour's allocation is an
hour of switching between demos, not one demo.

**And it comes back where you left it.** Moving the GPU used to throw the run
away, which meant four minutes of coarsening gone every time you switched. The
far side now parks the state and puts it back, so going between two demos costs
about a second and neither of them starts over. Reset is about as quick, for the
same reason: placing 50,000 beads with a minimum separation used to be 47 seconds
of LAMMPS rejecting candidates one at a time, and is now half a second of numpy.

`--gpu-hours` says how long to ask Slurm for (the default is one hour). It is the
backstop that gives the GPU back when everything else has failed to, so it is not
a number to pad -- and a longer request may sit in the queue longer:

```bash
lammps-live --playground mesomem_remote --gpu-hours 3
```

**On your cluster, not just mine.** The login, the account, the partition and the
paths on the far side live in a config file, not in the scene file, so running
this is not a source edit:

```bash
lammps-live --write-config       # ~/.config/lammps-live/config.toml
```

```toml
[remote.systems.mycluster]
host = "cluster.example.org"
user = "your-login"
partition = "gpu"
account = "prj1234"
remote_dir = "~/lammps-mesomem"  # where the cluster's LAMMPS build lives
env_script = "env.sh"            # sourced there before the server starts
profile = "cluster-gpu"          # "cluster-cpu" if there is no GPU to have
```

Define several and `--hpc mycluster` picks one for a run. The caveat is the one
thing this app can't do for you: **the LAMMPS on the far side has to exist
already**, with a Python module, numpy, the MesoMem pair style compiled in, and
Kokkos+CUDA if you want the GPU profile. The connect flow probes for exactly that
and refuses to allocate anything for a build that can't serve, and
`lammps-live --doctor` prints the probe command with your own paths already in
it.

How it works: [docs/remote-gpu.md](remote-gpu.md).
How to run it on your own cluster: [docs/cluster-setup.md](cluster-setup.md).
Snellius specifically: [docs/snellius/README.md](snellius/README.md).

## The joystick

An old Microsoft Sidewinder Force Feedback 2, talked to over raw HID with
[this driver](https://github.com/stefanhuber1993/sidewinder). You can reach the
whole demo from it, which is the point. Once you're standing in front of people
with a stick in your hand you don't want to go hunting for the keyboard.

The stick has two axes and there's more than two things worth steering, so only
one control is live at a time and the hat switch moves between them, laid out
like the screen: left and right cross between the scene and the control panel,
up and down walk the panel's rows (the bead colouring, then each slider). A cyan
frame shows which one you're on. Then you just push the stick to drive it. Most
of the travel is a slow band for placing a value carefully, and it accelerates
near the end when you want to cross the whole range.

Once the panel has the focus the stick's own up/down axis walks the rows too, so
the hand never has to leave the stick: push forward or back to pick the control,
left or right to move its value. A flick is exactly one row, and holding it walks
about two rows a second -- slow enough to let go on the one you wanted.

The trigger starts and stops the simulation, on every scene, and 2 resets it --
back to the beginning, which means the sliders too: the reason to push a dial
somewhere absurd is to see what happens, and what you want next is one button
that undoes all of it. 3 and 4 are scene back and forward. Every scene shows
Play, Pause and Reset buttons for the same three actions, and Space, R and Tab
are the keyboard twins.

The remote scenes' Connect card takes the stick while it is up, because it is
modal and there is no simulation behind it to steer: left and right pick a
button, and the trigger presses the one with the cyan ring on it. Standing in
front of a room with a stick in one hand, reaching back for a trackpad to press
Connect is exactly the kind of thing that goes wrong in public.

The thrust lever cuts the scene open, and it's a position, not a button: shove it
to either stop and there's no cut at all, and anywhere in between the view
narrows to a slab 15% of the box thick, square-on to whichever direction you're
looking from, with the lever's position sweeping that slab through the box.
Nothing times out -- cut in, let go, and it stays cut. It works on any of the 3D
scenes; on `mesomem_polymer` it's the only way to see anything at all, since a
closed membrane is opaque and the whole point of that one is what's inside it. A
lever you haven't touched since you started the app never cuts anything, wherever
it happens to be sitting.

The force feedback runs on the device itself instead of being streamed frame by
frame: a spring whose centre and stiffness follow the contact force, a damper
that stiffens when you're in contact, and a vibration standing in for thermal
jitter. None of it is required, `--input mouse` and `--input keyboard` work
fine.

**Push or twist.** A scene says which of the two the stick's axes are for.
Normally they push the bead around, inside the drawn net, and the twist axis
turns its director as a second control. `mesomem_patch_torque` swaps that round:
the two axes *turn* the director instead, about two axes, and nothing pushes the
bead at all -- where it goes is the membrane's answer. Everything downstream
follows into the rotational domain, force feedback included, so what you feel is
the tilt term twisting back. Green and red mean the same as always: what you're
doing, and what the membrane is doing about it.

## Run it

```bash
brew install mpich git          # Linux: apt install build-essential mpich libmpich-dev git
python3 -m venv venv && source venv/bin/activate
pip install -e .
lammps-live --doctor            # what this machine resolved to
lammps-live --input mouse
```

MPICH and not Open MPI, because that's what the `lammps` wheel links against.
The MesoMem force field compiles itself into a LAMMPS plugin the first time you
open a 3D scene, takes about 10 seconds once, and there's nothing to download
for it.

```bash
lammps-live --input joystick               # wants hidapi: brew install hidapi
lammps-live --playground mesomem_assembly  # start on a specific scene
lammps-live --ui-scale 1.5                 # bigger UI on a 4K screen
lammps-live --list                         # everything runnable
```

**Other machines, other compilers.** The pair style is compiled here, so the
compiler, the architecture flags and which MPI's headers to use are all
configurable rather than hardcoded — `-march=native` by default (probed, not
assumed), `g++` or MSVC where that's what there is, LAMMPS' own MPI stubs for a
serial build. Windows works natively and works with no surprises at all under
WSL2. `lammps-live --doctor` prints every one of those decisions and
`--build-plugin` compiles on the spot:

```toml
# ~/.config/lammps-live/config.toml   (lammps-live --write-config makes one)
[build]
compiler = "g++"
arch = "native"                  # or "none", or "-march=x86-64-v3"
mpi_include = "/usr/lib/x86_64-linux-gnu/openmpi/include"
```

Per-platform install notes, the whole `[build]` table and a symptom-to-fix table
are in [docs/install.md](install.md). The Linux joystick also needs a udev
rule so you can get at `/dev/hidraw*` without root:

```bash
echo 'SUBSYSTEM=="hidraw", ATTRS{idVendor}=="045e", ATTRS{idProduct}=="001b", TAG+="uaccess"' \
  | sudo tee /etc/udev/rules.d/99-sidewinder-ff2.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### Unattended: `--lock`

```bash
LAMMPS_LIVE_LOCK_PASSWORD=... lammps-live --input joystick --fullscreen --lock
```

Closing the window, Cmd-Q, Esc, minimising, hiding and leaving fullscreen all
ask for the password (without the variable it is asked for on the terminal at
startup). On macOS it also hides the Dock and the menu bar and disables Cmd-Tab,
Force Quit and logout for as long as the app runs; a watchdog exits the app if
its main loop ever hangs for 90 s, so a frozen demo cannot take the machine with
it. The right password does what was asked and leaves the app unlocked;
`Ctrl-L`, or a minute with nobody touching it, locks it again. Five wrong answers
lock the prompt for 30 s.

It is a lock for visitors, not a security boundary: see the notes on running the
machine unattended below the scene list in `remote-gpu.md`, and use a
separate macOS account and screen lock as well.

## Adding a scene

One file of about 50 lines that names a force field, a scenario and a mode.
Nothing to subclass:

```python
PLAYGROUND = Playground(
    name="MesoMem membrane patch",
    force_field="mesomem",
    scenario=hex_patch(n_rings=1),
    mode="game",
)
```

Every live parameter the force field declares turns into a slider on its own.
Put the file in `lammps_live/playgrounds/`, or keep it wherever and run
`lammps-live --playground ./my_idea.py`.

## Under the hood

Some things worth knowing, the details are elsewhere:

- The 3D scenes are GPU sphere impostors going through a deferred shading chain
  with ambient occlusion, contact shadows and depth of field. [The impostor
  book](impostor-book/) is the long version of that story, and the
  [Impostor Viewer](https://stefanhuber1993.github.io/lammps-live/) is the same
  renderer in a browser tab, for your own dump files. It lives on the
  `gh-pages` branch, which is its only copy: edit it there.
- MesoMem runs in the paper's reduced LJ units and the atomistic scenes run in
  real metal units, and the readouts follow whichever model you're in rather
  than one house style.
- Dragging a slider until the simulation dies is fair game. It recovers on its
  own and tells you what happened, on the cluster too, where the old failure
  mode was losing the GPU with it.
- `a100-plan.md` is the plan for making the remote one bigger, with the
  measurements it's based on.
- Nothing about your machine is hardcoded any more: the compiler and its flags,
  the MPI headers, your cluster login and its paths are all one TOML file
  (`lammps-live --write-config`), layered under the environment for one-off
  overrides. [docs/install.md](install.md) is this end,
  [docs/cluster-setup.md](cluster-setup.md) is the other one, and
  `lammps-live --doctor` prints what both of them came out as.
