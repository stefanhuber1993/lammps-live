# LAMMPS live

An MD simulation you can grab with your hands. It's real LAMMPS running under a
60 fps game loop, and you drive it with a force feedback joystick, so you
actually feel what the model pushes back with.

![Seven MesoMem beads](docs/images/mesomem_patch.png)

Those seven balls are MesoMem beads, the coarse grained membrane model from the
Idema group. It's their actual LAMMPS pair style
([arXiv:2602.24123](https://arxiv.org/abs/2602.24123)) compiled in, not my
approximation of it. Every bead carries a director, which is what the
yellow/blue banding shows, and basically everything the model does (membranes
staying flat, healing, assembling themselves out of a random soup) comes from
those directors wanting to line up with their neighbours.

## What it's for

In the paper membrane elasticity is a handful of numbers. A tilt modulus, a
splay modulus, a transition somewhere around `k_tilt ~ 10`. Here you can just
grab the thing and find out what those numbers feel like.

- **Feel it.** Pull a bead out of the sheet and the membrane resists in your
  hand. The stick is limp when nothing is holding you and gets firm on contact,
  and it buzzes more when you turn the temperature up. Twist hard enough and the
  director flips over to the other normal, because the tilt term has two minima.
  That's not an effect I added, it's the force field.
- **Turn the knobs while it runs.** Every coefficient is a live slider. Take
  `k_tilt` down through the transition and the same run stops making membranes
  and starts making blobs.
- **Watch it build itself.** 1500 beads dropped in at random, coarsening into
  flat lamellae in front of you, in about a minute.

Mostly it's a demo you give standing up, and a way to get an intuition for a
model whose parameters otherwise stay pretty abstract.

## Scenes

| | |
|---|---|
| `mesomem_bead` | two beads: one in your hand, one nailed down at the edge of the net. Drive them together and feel the pair potential switch on, term by term |
| `mesomem_patch` | seven beads. Pull the middle one out and feel tilt and splay resist |
| `mesomem_patch_torque` | the same seven, twisted instead of pulled: the stick turns the middle bead's director and the ring splays after it |
| `mesomem_sheet` | ~900 beads, periodic, so a piece of an endless membrane. Watch a deformation spread |
| `mesomem_assembly` | 1500 beads from a random start, assembling. Play / Pause / Reset |
| `mesomem_rod` | 3600 beads at constant tension. Steer a rod-shaped "bacterium" in and watch the membrane engulf it -- sideways first, then a neck, then the rod standing up inside the pit. Cut it open with the thrust lever to read the profile |
| `mesomem_remote` | same thing at 10,000 beads, running on a cluster A100 |
| `mesomem_polymer` | a closed vesicle with 32,000 beads of ring polymer sealed inside, on the same A100. Slice it open with the thrust lever to see in |
| `cu_deposition`, `lj_argon`, `nacl` | the atomistic classics: copper (EAM), argon melting, and a salt lattice where you can switch the ionic charge off and watch it fall apart |

`1`-`9` or `Tab` switches between them, `lammps-live --list` prints them.

## It runs on a supercomputer

`mesomem_remote` and `mesomem_polymer` put the simulation on an A100 at
[Snellius](https://www.surf.nl) and keeps the picture here at 60 fps. You press
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

How it works: [docs/remote-gpu.md](docs/remote-gpu.md).
How to run it: [docs/snellius/README.md](docs/snellius/README.md).

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

On Linux the joystick also needs a udev rule so you can get at `/dev/hidraw*`
without root:

```bash
echo 'SUBSYSTEM=="hidraw", ATTRS{idVendor}=="045e", ATTRS{idProduct}=="001b", TAG+="uaccess"' \
  | sudo tee /etc/udev/rules.d/99-sidewinder-ff2.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

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
  book](docs/impostor-book/) is the long version of that story.
- MesoMem runs in the paper's reduced LJ units and the atomistic scenes run in
  real metal units, and the readouts follow whichever model you're in rather
  than one house style.
- Dragging a slider until the simulation dies is fair game. It recovers on its
  own and tells you what happened, on the cluster too, where the old failure
  mode was losing the GPU with it.
- `docs/a100-plan.md` is the plan for making the remote one bigger, with the
  measurements it's based on.
