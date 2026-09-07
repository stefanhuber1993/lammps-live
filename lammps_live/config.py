"""Global constants that apply regardless of which system is active. Values
that vary by system (lattice spacing, damping range, force-feedback tuning,
...) live on each system's SystemSpec instead -- see systems/base.py."""

WINDOW_SIZE = (1300, 900)

# Real sim-time advanced per rendered frame, in ps. Converted to a per-system
# step count (SIM_TIME_PER_FRAME / system.spec.timestep) rather than a fixed
# step count, so systems with a smaller timestep (see lj_argon.py) still
# advance at the same physical rate instead of appearing in slow motion.
SIM_TIME_PER_FRAME = 0.003  # ps

# Joystick mode drives the puller mainly by the stick's input force, feeding the
# MD interaction force back INDIRECTLY through force feedback (see app.py). This
# is the fraction of that MD force also applied DIRECTLY to the puller in the
# simulation: 0.0 = puller feels only the stick (a pure haptic loop -- felt too
# detached in practice), 1.0 = puller feels the full MD force directly (like
# mouse mode). A middle value keeps some direct contact coupling. Mouse mode
# ignores this and always feels the full force.
JOYSTICK_MD_FORCE_FELT_FRACTION = 0.5

TEMP_KEY_RATE_FRACTION = 0.05     # fraction of a system's temperature range, per second, while Up/Down is held
TEMP_WHEEL_STEP_FRACTION = 0.02   # fraction of a system's temperature range, per mouse-wheel notch
HISTORY_WINDOW_SECONDS = 20.0
TRAIL_WINDOW_SECONDS = 2.0   # how far back every atom's fading motion trail reaches, in wall-clock seconds
# Trail snapshots are recorded every Nth rendered frame rather than every
# frame: with ~250+ atoms in cu_eam, drawing a full-rate (60Hz) trail for
# every atom costs several ms/frame just in per-segment pygame.draw.line
# calls (measured). Sampling at a lower rate cuts that roughly linearly
# while staying visually smooth -- the trail is a short, fast-fading smear,
# not a precision plot.
TRAIL_SAMPLE_EVERY_N_FRAMES = 5

# Force-feedback signal smoothing (see forcefeedback.py): recomputed from
# scratch every frame from instantaneous LAMMPS state, which is jerky
# frame-to-frame (individual atom vibrations, contact transients). Low-pass
# it with a simple exponential filter before it reaches the device, in real
# time rather than frame count so it's independent of frame rate.
FF_SMOOTHING_TAU = 0.1   # seconds

# Run the simulation on a worker thread so it overlaps the frame's drawing,
# rather than the two taking turns. LAMMPS releases the GIL inside `run`, so the
# step genuinely proceeds on another core while pygame draws -- worth ~5 ms a
# frame on the big membrane systems. Costs one frame of latency between an input
# force and seeing its effect. See stepper.py. False = take turns.
OVERLAP_SIM_AND_RENDER = True

# Start / stop the simulation, the same action as the Space key. 1 is the trigger
# on the Sidewinder FF2, and this is what it means on EVERY playground: the
# trigger is the most prominent control on the device, so it has to mean the one
# thing every scene has in common. (It used to double as grab-the-bead on the
# interactive playgrounds, which made the same finger do two unrelated things
# depending on which scene was loaded. The puller is B, and moving the focus off
# the viewport.)
JOYSTICK_PLAY_PAUSE_BUTTON = 1
# Reset the run to a fresh state, the same action as the R key. Every playground,
# for the same reason as the trigger.
JOYSTICK_RESET_BUTTON = 2
# Switch playground, the same action as Tab (forward) and the number keys.
JOYSTICK_PREV_PLAYGROUND_BUTTON = 3
JOYSTICK_NEXT_PLAYGROUND_BUTTON = 4
# The first device button that fires a HERO KNOB (see playground/spec.py), with
# the rest following it upward: 5 is a scene's first knob, 6 its second, and so
# on. Chosen as the first block that was still free, and left as a block so a
# scene can declare more than one without the mapping moving.
#
# The number is drawn ON each button in the row (see Renderer.draw_hero_knobs), so
# this constant and what the screen says cannot drift apart.
JOYSTICK_HERO_FIRST_BUTTON = 5
# How many of them are reachable that way. A scene may declare more knobs than
# this -- they still draw and are still clickable, they just have no device button
# -- but four is where the Sidewinder's numbered buttons run out.
JOYSTICK_HERO_BUTTONS = 4
# What the stick feels while it is NOT holding a bead -- flying the camera or
# setting a slider (see app._route_stick). The spring goes to full stiffness and
# true centre, because both of those controls are rate controls read off the
# stick's own position: "let go" has to mean exactly zero, and the hand should not
# have to find the middle itself. This is the damper that goes with it, as a
# fraction of the device maximum -- enough that the return to centre lands rather
# than rings, without making the stick feel like it is in treacle.
JOYSTICK_CENTERING_DAMPER_FRACTION = 0.35
