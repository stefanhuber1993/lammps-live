# FAQ: how the demo works

Short answers to the questions a poster visitor is likely to ask. Each section
says which file to open if you want the full story.

---

## 1. Is LAMMPS from Python slower than LAMMPS on the command line?

**Hardly, and for the physics itself not at all.** The `lammps` Python module is
not a reimplementation. It is a thin wrapper (ctypes) around the same compiled
`liblammps` library that the `lmp` executable uses. When Python says
`lmp.command("run 20")`, the whole run (neighbour lists, force loops, integration)
happens in the same C++ code at the same speed. Python only *sends the order*.

**Where the overhead is**

| What | Cost | Why it is small |
|---|---|---|
| Sending one command string | ~2 µs (measured) | LAMMPS parses one line; nothing gets copied |
| Starting each `run` | a full force evaluation plus a neighbour rebuild | skipped with `run N pre no post no` when nothing structural has changed (`playground/system.py`, `step()`) |
| Reading positions | one `numpy` copy per frame | `lmp.numpy.extract_atom("x")` is a *view* onto LAMMPS' own memory; we copy it once so the renderer has a stable snapshot |

Measured on this laptop (4000 LJ atoms, 2000 steps):

```
one `run 2000`                          3131 ms   (what the command line would do)
100 × `run 20`                          3693 ms   (+18%: every run re-does its setup)
100 × `run 20 pre no post no` + read x  3217 ms   (+3%: what the demo does)
```

**Why it is still fast enough to be interactive**

- The demo runs the simulation in small chunks (about 20 steps per frame) and
  only talks to LAMMPS between chunks. Python never sits inside the inner loop.
- While LAMMPS runs it releases Python's GIL, so the next chunk computes on one
  core while the previous frame draws on another (`lammps_live/stepper.py`). The
  simulation step largely disappears behind the drawing.
- The expensive physics (the MesoMem membrane) is the authors' C++ pair style,
  loaded into LAMMPS as a plugin, not Python code.

*One-liner for the poster:* "Python is the steering wheel and LAMMPS is the
engine. The engine runs at full speed; we only turn the wheel between strokes."

---

## 2. LAMMPS basics in a few snippets

A LAMMPS input ("deck") is a list of commands, executed top to bottom.

**A minimal Lennard-Jones liquid**

```lammps
units        lj                      # reduced units: sigma = epsilon = mass = 1
atom_style   atomic                  # particles are just points with a type
lattice      fcc 0.8442
region       box block 0 10 0 10 0 10
create_box   1 box                   # 1 atom type
create_atoms 1 box                   # fill the box with type-1 atoms
mass         1 1.0

pair_style   lj/cut 2.5              # which force law, and its global cutoff
pair_coeff   1 1 1.0 1.0 2.5         # epsilon, sigma, cutoff for type pair 1–1

velocity     all create 1.0 87287    # random velocities at T = 1
fix          1 all nve               # the integrator (constant N, V, E)
fix          2 all langevin 1.0 1.0 1.0 4242   # a thermostat (heat bath)
timestep     0.005
run          1000                    # take 1000 steps
```

The key ideas: **`pair_style`** picks the force law, **`pair_coeff`** sets its
numbers, and **`fix`** is anything applied every step (integrators,
thermostats, walls, pulling forces).

**The same thing from Python** (this is what the demo does)

```python
from lammps import lammps
lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
lmp.command("units lj")
# ... same commands as above ...
lmp.command("run 20")
x = lmp.numpy.extract_atom("x")      # positions as a numpy array, zero copy
```

**The MesoMem membrane in this repo** (`forcefields/mesomem.py`)

```lammps
atom_style hybrid sphere dipole       # each bead has a position AND an orientation (the "director")
plugin load mesomem.dylib             # the authors' C++ pair style, loaded at runtime
pair_style mesomem 2.5                # rc = 2.5
#          i j sigma eps k_tilt k_splay rc  wc  zeta c0  splay_symmetry
pair_coeff 1 1 1.0  1.0 12.0   1.0     2.5 2.0 5.0  0.0 0.0
```

**Membrane plus rod: two force laws at once** (`forcefields/mesomem_rod.py`)

```lammps
pair_style hybrid mesomem 2.5 rod_lj 3.24
pair_coeff 1 1 mesomem 1.0 1.0 12.0 1.0 2.5 2.0 5.0 0.0 0.0   # bead–bead
pair_coeff 1 2 rod_lj  0.5 3.0 3.24                           # bead–rod
set type 2 charge 5.0        # the rod's length is stored in its "charge" field
set type 2 diameter 3.0      # rod thickness
```

---

## 3. How do the sliders change the physics live?

LAMMPS lets you **re-issue a command between runs**, and it overwrites the old
setting in place. Particles, velocities and the deformed membrane all stay as
they are; only the force law changes. So a slider move becomes:

```
run 20 pre no post no
pair_coeff 1 1 1.0 1.0 15.0 1.0 2.5 2.0 5.0 0.0 0.0     ← k_tilt moved from 12 to 15
run 20                                                   ← next chunk uses the new k_tilt
```

Each parameter declares how much has to be redone when it changes
(`playground/params.py`):

| Tier | Example | What is re-issued |
|---|---|---|
| **HOT** | `k_tilt`, `k_splay`, `zeta`, `c0`, `eps_rod` | just the `pair_coeff` line |
| **HOT_RESTYLE** | `rc`, rod length or radius | `pair_style` + `pair_coeff`, because the global cutoff (and so the neighbour list) moves |
| **STRUCTURAL** | bead count, box size | cannot be a slider: needs a rebuild |

Temperature works the same way: the thermostat `fix` is re-issued with a new
target (`set_target_temp`). The rod's shape is per-particle data, so it is
changed with `set type 2 charge …` and then the pair style is re-declared.

After a structural change, the next `run` does its full setup again (no
`pre no`), so LAMMPS rebuilds everything it needs. If a setting blows the
simulation up, the error is caught and shown in the HUD, and Reset recovers.

---

## 4. The Sidewinder joystick: `ff.spring(...)` and `set_condition(...)`

### Is the joystick code part of this repo?

**Linked, not copied.** The driver lives in its own repo
(`github.com/stefanhuber1993/sidewinder`) and is a **pip git dependency** in
`pyproject.toml`:

```
"sidewinder @ git+https://github.com/stefanhuber1993/sidewinder.git"
```

`pip install -e .` downloads it into the venv's `site-packages`. It is not
vendored and not a git submodule. This repo only *uses* it
(`lammps_live/input/joystick.py`); protocol fixes go upstream.

### What the stick can do

The Sidewinder Force Feedback 2 speaks the standard **USB HID PID** (Physical
Interface Device) protocol. You don't stream motor currents to it. You upload an
*effect* ("be a spring centred here, this stiff") and the stick's own firmware
runs that effect against its own position sensor, so the force reacts
instantly, independent of our frame rate. We just update the effect's
parameters about 60 times per second.

### `ff.spring(stiffness, cp_offset_x, cp_offset_y, saturation)`

This is a convenience call: it allocates a *Spring* effect slot on the device,
sets both axes with `set_condition`, and starts playing it. It returns an
`Effect` handle. In the demo:

```python
self.spring = self.ff.spring(stiffness=1, saturation=255)    # starts almost limp
self.damper = self.ff.damper(coefficient=13, saturation=255) # a little viscous drag
```

### `effect.set_condition(axis, cp_offset, pos_coeff, neg_coeff, pos_sat, neg_sat, dead_band)`

This sends one HID "Set Condition" report (report ID 3) for **one axis** of a
playing effect. You can call it at any time while the effect plays.

| Input | Range | Meaning for a Spring |
|---|---|---|
| `axis` | 0 = X, 1 = Y | which axis this report sets |
| `cp_offset` | −128…127 | **centre point**: where the spring wants the stick to be |
| `pos_coeff` / `neg_coeff` | −128…127 | **stiffness** on each side of the centre (127 = maximum) |
| `pos_sat` / `neg_sat` | 0…255 | **saturation**: the maximum force on each side (a cap) |
| `dead_band` | 0…255 | zone around the centre with no force |

What the motors then do, computed on the stick itself:

```
F = −k · (stick_position − cp_offset),   clipped to ±saturation
```

For a *Damper* the same report means `F = −k · stick_velocity` (it opposes
motion, like moving through honey).

### How the simulation becomes a force in your hand (`forcefeedback.py`, `joystick.py`)

1. LAMMPS gives us the force on the bead you are pulling.
2. That force is **soft-saturated** with `tanh`, so small forces are exaggerated
   enough to feel and big spikes are capped smoothly instead of clipping.
3. It becomes the spring's **centre offset**: the stick is pulled in the
   direction the molecule pulls.
4. Its magnitude also sets the **stiffness** (limp when nothing touches the
   bead, firm in contact) and the **damper** coefficient.
5. A small **sine vibration** effect on top is scaled with √T, which gives the
   feel of thermal jitter.
6. The writes are made on a background thread, and are skipped if the value has
   not changed since the last write (each is a blocking USB transfer).

---

## 5. The rod: how its force term works, and the words to use

The rod (the "bacterium" being wrapped) is **one particle**, not a chain of
beads. It carries its geometry on itself: a centre, a long-axis direction (the
`mu` vector), a length `L` (stored in the `q`/charge field) and a radius.

The pair style `rod_lj` (Pietro Sillano, `mesomem_ff/pair_rod_lj.cpp`) does this
for every membrane bead near the rod:

1. **Closest-point projection.** Project the bead onto the rod's axis segment
   and clamp the result to the segment's two ends. That gives the contact point
   **C** on the axis.
2. **Surface-to-surface distance.** Take the distance `r` from the bead to C and
   use a 12-6 Lennard-Jones with an enlarged size
   `σ_eff = σ_bead + r_rod`. So the LJ "sees" the rod's surface, not its axis.
   Clamping to the ends makes the shape a **spherocylinder** (a capsule).

   ```
   U(r) = 4 ε_rod [ (σ_eff/r)^12 − (σ_eff/r)^6 ]
   ```

   The r⁻¹² part is the **steric / excluded-volume repulsion** (beads can't
   enter the rod). The −r⁻⁶ well is the **adhesion** (`eps_rod`, the slider that
   decides "wrap or just dent").
3. **Force and torque.** The force acts along the line bead→C, equal and
   opposite (Newton's third law). Because C is generally *not* the rod's centre,
   the force also has a **lever arm**, so the rod gets a **torque**
   `τ = (C − x_rod) × F`. That is why the membrane can turn the rod and roll
   around it, not only push it.

**Words for the poster:** *anisotropic* (direction-dependent) *point–segment
Lennard-Jones*; *rigid spherocylinder*; *closest-point projection*;
*surface-to-surface contact distance*; *excluded volume plus adhesion*;
*force and torque on a rigid body*. The membrane beads interact with each other
through MesoMem, and with the rod through `rod_lj`. `pair_style hybrid` routes
each type pair to the right law.

**The physics story:** wrapping is a competition between **adhesion energy**
gained (∝ contact area × ε_rod) and **bending energy** paid by the membrane
(set by k_tilt / k_splay). Strong enough adhesion wins, and the membrane
engulfs the rod.

---

## 6. Spontaneous curvature (c0)

**Idea:** a membrane doesn't have to prefer being flat. Real bilayers with
asymmetric lipids or bound proteins prefer to bend with a radius `R`. The
**spontaneous curvature** is `c0 = 1/R`. With c0 = 0 the membrane prefers
flat; with c0 > 0 a flat sheet is frustrated and wants to close into a sphere
of radius R.

**Geometry.** Put two neighbouring beads on a sphere of radius R, a distance
`r` apart. Their normals (the directors **n**) are not perpendicular to the
line joining them. Each is tipped by half the angle α the pair spans at the
sphere's centre:

```
sin(α/2) = r / (2R) = r·c0 / 2        ≡ s
n_i · r̂ = −s,    n_j · r̂ = +s,    n_i · n_j = cos α = 1 − 2s²
```

**How it enters the energy** (`forcefields/mesomem.py`, `energy_terms`, which
mirrors the C++):

```
U_tilt  = (k_tilt / 2) [ (n_i·r̂ + s)² + (n_j·r̂ − s)² ] · w(r)
U_splay = (k_splay / 2) ( n_i·n_j − 1 + 2s² )²          · w(r)
```

- With **c0 = 0** (s = 0): tilt wants n ⊥ r̂ (directors normal to the bonds)
  and splay wants n_i·n_j = 1 (parallel). Both are zero for a **flat** sheet.
- With **c0 > 0**: both terms are zero exactly at the sphere geometry above, so
  the lowest-energy shape is **curved with radius 1/c0**.
- `w(r)` is a smooth weight that fades the orientational terms to zero at the
  cutoff `wc`.

So c0 doesn't add a new force. It **moves the minimum** of the existing tilt
and splay terms from "flat" to "bent by a set amount". It is the
large-assembly box's hero knob: at c0 > 0, patches stop forming flat sheets
and close into vesicles.

*One-liner:* "c0 tells every pair of neighbours how much they should lean
towards each other. Add it up over the whole membrane and you get a sphere of
radius 1/c0."

---

## 7. The cluster colouring

**Purpose.** Give each aggregate (patch, membrane, vesicle) its own colour so
you can *see* self-assembly: many small colours merging into a few big ones.
It is pure visualisation and never feeds back into the simulation
(`playground/clustering.py`).

**Step 1: what is a cluster?** Two beads are "in contact" if their centres are
closer than **1.25 bead diameters**. A cluster is a **connected component** of
that contact graph: everything reachable through a chain of contacts. This is
the standard cluster analysis (`compute cluster/atom` in LAMMPS). We use it
instead of k-means or graph cuts because those would split one connected
membrane into pieces, while the question is "one piece or two?".

- Contacts are found with a **cell list** (bin the box into cubes of the cutoff
  size, only compare neighbouring cubes), which is O(N).
- Distances use the **minimum-image convention**, so a membrane across a
  periodic boundary is one cluster.
- Components are found with **Shiloach–Vishkin** label propagation, vectorised
  in numpy: each bead takes the smallest label among its neighbours, hooked onto
  the *root* of its tree. That converges in about 4 rounds even for a large
  membrane.
- **Size threshold:** tiny groups (< 4 beads) stay grey. In a dense gas random
  "chance clumps" get bigger, so the threshold rises with density (fitted as
  `1.3·exp(1.9·n̄)`, where n̄ is the expected number of neighbours within the
  cutoff).

**Step 2: stable colours over time.** Raw component labels are anonymous and
change every frame, so painting them directly would flicker. `ClusterTracker`
adds four things:

1. **Matching:** each new cluster keeps the colour most of its beads already
   wear, handled largest-first. When two clusters merge, the big one keeps its
   colour and only the small one changes.
2. **Map-colouring rule:** there are 10 colours, so they repeat, but never on
   one of a cluster's 4 nearest neighbouring clusters (like a political map).
   New clusters take the least-used allowed colour.
3. **Hysteresis:** a bead must disagree with its colour for 3 labellings in a
   row before it switches. A bead rattling across the cutoff never flickers.
4. **Rate:** clustering is recomputed only every 5+ frames (less often for big
   systems). The renderer crossfades colours in between. It also runs on the
   time-smoothed positions, not the raw jittery ones.

*One-liner:* "Beads that touch belong together. We find the connected groups,
then hand out colours like a map-maker: neighbours always differ, and a group
keeps its colour as it grows."

---

## 8. The six lines on the poster: steering LAMMPS from Python

```python
lmp = lammps()                                  # LAMMPS as a library
lmp.command("variable fx internal 0.0")         # a slot Python can write
lmp.command("fix drive probe addforce v_fx v_fy 0")  # push the probe bead
lmp.set_internal_variable("fx", joystick_fx)    # hand -> input force
lmp.command("run 20 pre no post no")            # advance 20 steps
x = lmp.numpy.extract_atom("x")                 # positions, no copy
```

This is the whole interactive loop, cut down to its bones. The first three lines
run **once**, when a scene is built. The last three run **every frame**, about 60
times a second. The real code is `playground/modes.py` (`control_commands`,
`set_input_force`) and `playground/system.py` (`step`).

**`lmp = lammps()`: LAMMPS as a library.** This loads `liblammps`, the same
compiled C++ the `lmp` executable runs, into the Python process (see section 1).
There is no input file and no separate program. The object `lmp` *is* a running
LAMMPS, and Python sends it commands one at a time. (The demo starts it with
`cmdargs=["-log", "none", "-screen", "none"]`, so it does not print to the
terminal or write a log.)

**`variable fx internal 0.0`: a slot Python can write.** LAMMPS variables are
usually formulas (`equal` style) or lists. The `internal` style is the one meant
for outside code: it holds a single number, and nothing inside LAMMPS changes it.
Think of it as a mailbox. Python puts a number in, and anything in the input
that says `v_fx` reads it out. In the real code there are three, `drive_x`,
`drive_y`, `drive_z`.

**`fix drive probe addforce v_fx v_fy 0`: push the probe bead.** A `fix` is
anything LAMMPS does on every timestep (section 2). `addforce` adds a constant
force vector to every atom in a group, here the group `probe`, which holds the
one bead you control (`group probe id 17`, say; in the code it is called
`controlled`). The force components are given as `v_fx` and `v_fy`, i.e. *read
from the variables*, and `addforce` re-reads them **every step**. The 0 is the z
component: on the 2D scenes the bead is held in a plane.

**`set_internal_variable("fx", joystick_fx)`: hand -> input force.** Each frame
the app reads the stick deflection, scales it to a force in simulation units,
and writes it into the mailbox. This is a direct memory write, not a command:
nothing is parsed, and **the fix itself is left untouched**. That is the point
of the whole construction (see the next line).

**`run 20 pre no post no`: advance 20 steps.** One frame of animation is 20 MD
steps. Normally every `run` starts with a full *setup*: rebuild the neighbour
lists, evaluate all forces once, re-initialise every fix. `pre no` skips that,
which is only allowed if nothing structural changed since the last run. `post
no` skips the timing summary printed at the end. Together they cut the overhead
of running in small chunks from about +18% to about +3% (the table in section 1).

*Why the variable and not just a number?* The obvious way to apply the stick's
force is to re-issue `fix drive probe addforce 1.3 -0.4 0` with the new numbers
every frame. That works, but **re-defining a fix counts as a structural
change**, so the next run would need its full setup again and `pre no` would be
lost on every frame. With the variable, the fix is defined once and never
touched again. Only the number in the mailbox changes, so every frame can skip
the setup. (A slider move that re-issues a `pair_coeff` is different: it really
does change the setup, so that one run takes the full setup. See section 3.)

**`x = lmp.numpy.extract_atom("x")`: positions, no copy.** This returns a numpy
array that points straight at LAMMPS' own position memory, an `(N, 3)` array of
x, y, z. Nothing is copied, so it costs the same for 10 beads or 100 000. It
comes with three catches, and the real code deals with all three:

- **It has more rows than atoms.** LAMMPS allocates spare room (in a quick test,
  1687 rows for 256 atoms), so the code slices it to the real count first:
  `extract_atom("x")[:natoms]`.
- **The rows are not in atom-id order.** LAMMPS re-sorts atoms in memory to keep
  neighbours close together, so row 0 is not necessarily atom 1. The code reads
  `extract_atom("id")` as well and orders by it.
- **It is a live view.** The next `run` overwrites it in place, and may even move
  it to a new block of memory. So the code takes one copy per frame
  (`np.array(...)`) and gives that stable snapshot to the renderer, while LAMMPS
  goes on to the next chunk.

**The frame, put together:**

```
read stick -> set_internal_variable -> run 20 pre no post no -> extract_atom -> draw
```

In the app the `run` happens on a worker thread (`lammps_live/stepper.py`).
LAMMPS releases Python's lock while it computes, so the next 20 steps are
calculated while the previous frame is being drawn.

*One-liner:* "Six lines: LAMMPS is a library, a variable is the mailbox between
the stick and a force, and we step 20 at a time and read the positions straight
out of LAMMPS' memory."

---

## 9. The game loop and the rendering: is it pygame? OpenGL?

**Both, each doing a different job.**

| Part | Library | Job |
|---|---|---|
| Window, events, keyboard and mouse, frame clock, all 2D drawing | **pygame** (SDL2) | the app shell: panel, sliders, plots, text, buttons, and the 2D scenes |
| The 3D bead scenes | **OpenGL 3.3** via **moderngl** | the beads, drawn on the GPU (`ui/gl3d.py`) |
| The physics | **LAMMPS** (C++, as a library) | runs on a worker thread (`stepper.py`) |
| The joystick | `sidewinder` driver over **hidapi** | its own background thread (`input/joystick.py`) |

pygame opens the window *as an OpenGL window*, and moderngl draws into it. If
there is no OpenGL 3.3 (an old machine, or a headless test), the app prints
"OpenGL unavailable" and falls back to a pure-pygame CPU renderer for the whole
session (`Renderer._draw_sim_3d_cpu`: numpy-shaded sphere sprites, sorted back
to front). That fallback is slower and less pretty, but it means the app always
starts.

### The game loop (`App.run` and `App._tick` in `app.py`)

A plain loop, once per frame, capped at 60 fps by pygame's clock:

```python
while running:
    running = self._handle_events(dt)   # pygame events: keys, clicks, resize, quit
    self._check_idle()                  # 60 s without input -> back to scene 1
    dt = self._tick(dt)                 # one frame, below
```

One `_tick` is five steps, in this order:

1. **Collect the simulation step launched last frame.** `stepper.wait()`. That
   step has been running *while the previous frame was being drawn*, so this
   usually waits for nothing.
2. **Push the inputs into LAMMPS.** Sliders become `pair_coeff` lines
   (section 3), the temperature dial re-issues the thermostat, and the stick
   becomes the drive force (section 8).
3. **Read everything this frame needs out of LAMMPS.** Positions, directors,
   energies, thermo, the force on the held bead. Every read is a **copy**, so
   the drawing has a stable snapshot.
4. **Start the next step on the worker thread.** `stepper.start(system, n)`,
   i.e. `run n pre no post no`. From here until step 1 of the next frame,
   nothing may touch LAMMPS.
5. **Force feedback and drawing, while that step runs.** Shape the measured
   force into what the stick should push back with and send it to the joystick.
   Then draw the frame and `clock.tick(60)`.

**Why steps 4 and 5 overlap.** The obvious loop (step, read, draw, repeat)
leaves one half idle while the other works. LAMMPS' `run` is C++ reached through
ctypes, and ctypes **releases Python's GIL** while it runs, so a Python thread
can draw on one core while LAMMPS computes on another. Measured on the
1500-bead scenes: step about 18-21 ms, draw about 5-6 ms. Overlapped, the step
hides the drawing completely. The cost is **one frame of latency** (16 ms at 60
fps): the picture is one step behind the one being computed. That is well below
the roughly 50 ms at which a haptic loop starts to feel disconnected.
`config.OVERLAP_SIM_AND_RENDER = False` turns the overlap off.

**How far each frame advances.** A fixed amount of *simulated time* per frame,
converted into a step count from the scene's timestep (`steps_per_frame`, up to
200). So a system with a smaller timestep runs more steps per frame rather than
looking like slow motion. On the MesoMem scenes that is 10 to 20 steps (20 on
the sheet and the assembly boxes).

**Run with `--debug`** to see where each frame goes: a header line splits it
into sim (only the part the drawing did *not* hide), analysis, stick read, force
feedback, gather, render, and other.

### How a 3D frame is drawn (`ui/gl3d.py`)

The beads are **sphere impostors**, the standard technique in molecular
visualisation (VMD and OVITO use it too). There is no triangle mesh per sphere.
Each bead is **one flat square facing the camera**, and the fragment shader
works out, per pixel, where a ray from the eye would hit a perfect sphere inside
that square. It then writes that point's true depth (`gl_FragDepth`). So:

- a bead is a mathematically perfect sphere at any zoom, for the cost of 4
  vertices;
- the GPU depth buffer resolves overlapping and intersecting beads exactly, so
  no sorting is needed;
- all beads go to the GPU in **one instanced draw call**. Per bead, the CPU
  uploads 15 floats (centre, radius, director, brightness, energy, tint, fade,
  material) into one buffer each frame.

After that comes a **deferred** post-processing chain, the look tuned in the
standalone showreel:

| Pass | What it does |
|---|---|
| 1 G-buffer | the impostors: colour, surface normal and position of the nearest bead per pixel |
| 2 Occlusion | at half resolution: ambient occlusion (dark crevices between packed beads) and soft sun shadows, both by sampling the depth buffer |
| 3 Blur | smooths the sampling noise of pass 2 |
| 4 Composite | lighting, the ink outline, fading into the background with depth, tonemapping |
| 5 Depth of field | blurs what is far from the focus plane |
| 6 Lines | bonds, the box and the control net, as real depth-tested lines, so beads hide them |
| 7 FXAA | screen-space antialiasing, since deferred rendering rules out MSAA |

Every pass after the first works **per pixel, not per bead**. So the 1500-bead
box post-processes for the same price as the 7-bead patch, and the bead count
mostly costs the upload and pass 1. What the picture looks like (the colours,
the outline, the strength of each pass) is not in the renderer: it is data in
`render_style.py`, which each playground can override.

### How the 2D UI gets on top

Once the window is an OpenGL window, pygame can no longer draw directly onto
it. So all the 2D drawing (panel, sliders, plots, titles, hero buttons) goes to
an **offscreen pygame surface with transparency**. At the end of each frame,
`ui/glcompositor.py` uploads that surface to a GPU texture and draws it as one
full-window quad over the 3D scene. It is transparent where the beads should
show through. Then `pygame.display.flip()` shows the finished frame. The older
2D scenes (argon, NaCl, copper deposition) are drawn entirely by pygame onto
that same surface. They are not in the demo sequence any more, but they still
load with `--playground lj_argon`.

### The browser version on the poster's QR code

The phone viewer (`impostor-viewer.html` on the `gh-pages` branch) is the same
renderer rewritten in WebGL2: the same impostors and a similar post-processing
chain, and the same `render_style` field names. It has no simulation behind it:
it plays a built-in demo or a LAMMPS dump file you drop on it. That is how a
phone can draw 100 000 beads: per bead it is still one square and one draw
call, and the rest is per pixel.

*One-liner:* "pygame is the window and the dashboard, OpenGL draws the beads as
perfect spheres from one square each, and LAMMPS computes the next step on
another core while the current one is being drawn."

---

## 10. From LAMMPS to your hand: the force loop in detail

Section 8 showed how the stick's force gets *into* LAMMPS. This section is the
whole round trip: stick → `fx, fy` → LAMMPS → reaction force → stick motors. The
files are `app.py` (`_tick`), `playground/modes.py` (`set_input_force`,
`interaction_force`, `constrain`), `forcefeedback.py` and `input/joystick.py`.

### Once, when the scene is built

```lammps
group     controlled id 17                 # the bead you hold
group     bath subtract all controlled     # everything else
fix       heat bath langevin T T 1.0 seed  # thermostat: NOT on the held bead
variable  drive_x internal 0.0             # the three mailboxes (section 8)
variable  drive_y internal 0.0
variable  drive_z internal 0.0
fix       drive controlled addforce v_drive_x v_drive_y v_drive_z
fix       damp  controlled viscous 4.0     # drag -gamma*v on the held bead
fix       plane controlled setforce NULL NULL 0.0   # no force along the pinned axis
```

The held bead moves in a **plane**, spanned by two world axes `u` and `v`; the
third axis is pinned. That is what lets a 2-axis stick fully steer a bead in 3D.
It is kept **out of the thermostat** on purpose. Otherwise the random kicks of
the heat bath would end up in your hand as noise, and the force measurement below
would no longer add up.

### Every frame

```
 ┌─ worker thread ─────────────┐   ┌─ main thread ─────────────────────────────────┐
 │ run 20 pre no post no       │   │ 1  wait for the run                           │
 │ then constrain(): plane,    │   │ 2  read stick -> fx, fy -> drive_u, drive_v   │
 │ leash, speed cap            │   │ 3  read f, v -> reaction force (F_u, F_v)     │
 └─────────────────────────────┘   │ 4  start the next run ───────────────────────►│
                                   │ 5  shape (F_u, F_v) -> spring on the stick    │
                                   └───────────────────────────────────────────────┘
```

**Step 2: the hand becomes `fx, fy`.** A background thread reads the stick
over USB, so the main thread only picks up the latest cached position, `jx, jy`,
each in −1…+1. That is scaled to a force:

```python
fx = jx * max_input_force          # 4.0 (reduced units) on the seven-bead patch
fy = jy * max_input_force
```

and written into the two mailboxes for the plane's axes:

```python
lmp.set_internal_variable("drive_x", fx)   # u axis
lmp.set_internal_variable("drive_y", fy)   # v axis
lmp.set_internal_variable("drive_z", 0.0)  # pinned axis
```

With the joystick there is one correction. The stick will *also* push your hand
with the membrane's force (step 5), and your hand gives way a little, which
changes `jx, jy`. So the membrane's force already reaches the bead once through
your hand. Applying it fully again, inside LAMMPS, would count it twice. So half
of it is taken off what goes to LAMMPS (`JOYSTICK_MD_FORCE_FELT_FRACTION = 0.5`):

```python
fx = jx * max_input_force - 0.5 * F_u      # F = the reaction force from last frame
fy = jy * max_input_force - 0.5 * F_v
```

With the mouse there is no force feedback, so the full force stays on the bead.

**Inside LAMMPS, for each of the 20 steps.** The held bead's total force is

```
f = F_membrane + (fx, fy, 0) − gamma · v            then setforce zeroes the pinned axis
    pair style   fix addforce   fix viscous
```

and the integrator moves it. After the 20 steps (still on the worker thread),
`constrain()` writes straight into LAMMPS' memory through the numpy view:
it puts the bead back exactly in its plane, clamps it to the leash, and caps its
speed. The view is writable, so `x[i][2] = z0` really moves the atom.

**Step 3: reading the force back.** What the hand should feel is
`F_membrane`, the force the *other beads* exert on the held one. LAMMPS can
report that directly with `compute group/group`, but only for pair styles that
have a `single()` method, and MesoMem's has none. So it is **reconstructed**
from the total force, by taking away the two forces this app added itself:

```python
f = lmp.numpy.extract_atom("f")      # total force on every atom, last step
v = lmp.numpy.extract_atom("v")
F_u = f[i][u] - fx_applied + gamma * v[i][u]     # undo addforce, undo viscous
F_v = f[i][v] - fy_applied + gamma * v[i][v]
```

`fx_applied` is the drive that was in effect *during that run*. This is exact as
long as nothing else acts on the bead in the plane. That is why the bead is kept
out of the thermostat. (A regression test pins the arithmetic.)
`setforce` only touches the pinned axis, which is not read.

Two adjustments are about the hand, not the physics. A released bead reports
zero (you are not holding it). And the force fades out as the bead nears the
edge of its leash, so reaching the limit is quiet instead of a constant shove.
On the twist patch the stick drives a *torque* instead. There the reaction is
read straight from LAMMPS' per-atom `torque` array, with no reconstruction
needed, and the rest of the chain is identical.

**Step 5: from `F_u, F_v` to the motors** (`forcefeedback.py`, values for the
MesoMem scenes):

1. **Soft-saturate:** `|F| → 120 · tanh(1.3 · |F| / 4.0)`, in the direction of
   `F`. Small forces are exaggerated enough to feel, and spikes are capped
   smoothly instead of clipping. The result is in device units (±127).
2. **Add a velocity brake:** a bias opposing the bead's own velocity, up to half
   of the full range. You feel the bead decelerate.
3. **Smooth:** a low-pass filter with a 0.1 s time constant, which takes out the
   thermal rattle of a single 20-step chunk.
4. **Stiffness from contact:** 0 while `|F| < 0.3` (nothing touching, the stick
   is limp), rising as `127 · tanh((|F| − 0.3) / 2.5)` in contact. The damper
   uses the same signal, 10 % to 55 % of its maximum.
5. **Send** the result as the spring's **centre offset** `(cx, cy)` and
   stiffness `k`. `y` is flipped back to the device's convention, and everything
   is rounded to the signed bytes the HID report carries.

The stick's own firmware then runs `F_motor = −k · (stick_position − c)` against
its own position sensor, much faster than our 60 fps (section 4). So we do not
stream motor forces. We move the spring's centre, and the stick does the rest.
The centre sits where the membrane is pulling, so the stick leans that way.
A small sine vibration on top, scaled with √T, adds the feel of temperature.

The USB writes happen on the joystick's background thread, and a write is
skipped if the bytes did not change since the last one, because each is a
blocking USB transfer.

### Timing

The force you feel in frame *n* was computed in the run that finished at the
start of frame *n*. That run used the stick position from frame *n − 1*. With
the 0.1 s smoothing on top, the whole loop answers within roughly 0.1 s. That is
quick enough that the membrane feels like it is pushing back, not like a
delayed echo.

*One-liner:* "Your hand sets a force, LAMMPS adds it to the membrane's, we take
our own force back out of the total and what's left is the membrane. That moves
the centre of a spring inside the stick."

---

## 11. Cheat sheet: the whole loop in 40 lines

Sections 8 to 10 in one runnable script, to remember the commands by. Plain
Lennard-Jones stands in for MesoMem, and a sine wave stands in for the stick.
It runs as is with `./venv/bin/python`.

```python
from lammps import lammps
import math, numpy as np

# ---- build once ---------------------------------------------------------------
lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])  # LAMMPS as a library
for c in ["units lj", "atom_style atomic", "lattice fcc 0.8442",
          "region box block 0 6 0 6 0 6", "create_box 1 box", "create_atoms 1 box",
          "mass 1 1.0", "pair_style lj/cut 2.5", "pair_coeff 1 1 1.0 1.0",
          "group probe id 1",                       # the bead the hand holds
          "group bath subtract all probe",          # everyone else
          "fix nve all nve",                        # integrator
          "fix heat bath langevin 0.5 0.5 1.0 42",  # thermostat, NOT on the probe
          "variable fx internal 0.0",               # mailboxes Python writes
          "variable fy internal 0.0",
          "fix drive probe addforce v_fx v_fy 0",   # hand force, re-read every step
          "fix damp probe viscous 4.0",             # drag -gamma*v on the probe
          "fix plane probe setforce NULL NULL 0",   # keep it in the xy plane
          "run 0"]:                                 # one full setup
    lmp.command(c)
GAMMA, MAX_F = 4.0, 4.0
i = int(np.flatnonzero(lmp.numpy.extract_atom("id")[:lmp.get_natoms()] == 1)[0])  # row of atom 1

# ---- every frame (60 per second) ----------------------------------------------
for frame in range(60):
    jx, jy = math.sin(frame / 10), 0.0          # joystick.poll(), -1..+1
    fx, fy = jx * MAX_F, jy * MAX_F             # hand -> force
    lmp.set_internal_variable("fx", fx)         # memory write, no command:
    lmp.set_internal_variable("fy", fy)         #   the fix stays untouched
    applied = (fx, fy)                          # what THIS run integrates with
    lmp.command("run 20 pre no post no")        # 20 steps, setup skipped
    n = lmp.extract_global("nlocal")            # real atom count (arrays are longer)
    f = lmp.numpy.extract_atom("f")[:n]         # views into LAMMPS memory, no copy
    v = lmp.numpy.extract_atom("v")[:n]
    x = np.array(lmp.numpy.extract_atom("x")[:n])  # copy: stable snapshot to draw
    Fu = f[i, 0] - applied[0] + GAMMA * v[i, 0]    # take out our push and our drag:
    Fv = f[i, 1] - applied[1] + GAMMA * v[i, 1]    #   what's left is the material
    mag = math.hypot(Fu, Fv) or 1e-9
    s = 120 * math.tanh(1.3 * mag / 4.0) / mag     # soft cap -> device units (±127)
    # joystick.send_force(Fu * s, Fv * s, k)       # spring centre on the stick
```

Two things the real app does that this leaves out. It runs the `run` on a worker
thread while the previous frame is drawn (section 9). And with the joystick it
takes half of last frame's `Fu, Fv` off `fx, fy`, so the membrane's force is not
counted twice (section 10).
