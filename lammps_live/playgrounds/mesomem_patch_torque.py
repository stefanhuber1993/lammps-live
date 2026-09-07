"""The same 7-bead patch, driven by TORQUE instead of force.

Physically identical to `mesomem_patch` -- the same force field, the same seven
beads at the same hexagonal spacing, the same box, the same camera, the same
presets. One thing changes, and it is what the stick does:

    mesomem_patch         the two axes PUSH the centre bead, in the xz plane,
                          inside a leash drawn as a net. Its director is a
                          secondary control, on the twist axis, and it swings only
                          within that plane.
    this one              the two axes TURN the centre bead's director, about two
                          world axes, and nothing pushes the bead at all. It is an
                          ordinary particle of the simulation that you happen to be
                          able to twist; where it goes is the membrane's answer,
                          not yours.

WHY THAT IS A DIFFERENT DEMO AND NOT A DIFFERENT BUTTON. Pulling a bead out of a
membrane is a question about the ISOTROPIC part of the potential: how hard the
4-2 well holds on, and how far you get before it lets go. Twisting one is a
question about the ORIENTATIONAL part -- the tilt term, which wants a director
along the local normal and is bistable (both +n and -n are minima), and the splay
term, which couples one director's deflection to its neighbours'. Twisting the
middle director and watching the ring splay after it, then dome, then let the flip
happen at the barrier, is the paper's orientational physics with a hand on it.
Under force control you can only reach that sideways, by pulling the bead until
the geometry does the twisting for you.

WHAT A FULL DEFLECTION ACTUALLY DOES, since it is worth knowing before standing in
front of it. The command is an angular-velocity kick rather than a torque added to
the pair style's (the same mechanism the twist axis has always used), so a stick
held at full travel does not stall against the tilt term -- it rotates the director
right around, roughly 90 degrees of it per three tau. What decides the outcome is
where you LET GO: released short of the barrier it springs back to +n, released past
it, it settles on -n, and both are real minima of the tilt term. A gentle deflection
is the one that finds the balance point and holds there against the reaction.

WHAT IS DELIBERATELY UNCONSTRAINED. Two of `mesomem_patch`'s constraints are gone,
and the same sentence removes both: nothing is holding the bead against an input
whose two axes have to determine a position, because the input is no longer a
position.

  NO PLANE AND NO LEASH (`confine=False`). The bead is free in three dimensions,
    and so there is no net -- the scene draws one only where there is a boundary to
    mark. The bead does not run off: nothing is pushing it, and the membrane it
    sits in is what holds it.
  NO PLANE FOR THE DIRECTOR EITHER. Under force control the director's swing is
    projected back into the control plane every frame, so that the two axes moving
    the bead are not also rotating it. Here the two axes ARE the rotation, they
    cover both of a director's degrees of freedom, and it tumbles freely -- which is
    the only way to take a director off the tilt term's great circle and see that
    the barrier it flips over is a barrier in two directions, not one.

The reaction torque is what reaches the hand and the red ring (see
modes.GameMode.interaction_force): the membrane's own restoring twist, read off the
pair style, with none of it subtracted from what you command.

WHAT YOU SEE AT THE BEAD, and why it is not what the force playgrounds show. No
straight arrows: nothing here pushes, and an arrow drawn along a torque's axis is a
picture of a push along a direction nothing moves in. Two rings instead -- green for
your command, red for the membrane's answer -- each a real circle in the scene,
lying in the plane its rotation happens in, and so seen by the camera as the ellipse
that plane is seen as: face on when you turn the director about the view axis, edge
on when you turn it in the plane of the screen, and tipping between the two as you
work the stick. That is the drive's second axis made visible, which a circle drawn
flat on the screen cannot do.

Units are the paper's LJ-reduced units (sigma = eps = m = 1).
"""
from ..playground import Control, Lesson, Playground, hex_patch
from .mesomem_patch import STYLE

PLAYGROUND = Playground(
    name="MesoMem membrane patch, torque control (3D)",
    description="Real MesoMem force field: twist the center bead's director and "
                "feel the tilt term resist -- the patch splays, domes and flips.",
    force_field="mesomem",
    # Every one of these is mesomem_patch's, unchanged and for its reasons -- see
    # that file. The scene has to be the same scene, or the comparison the pair of
    # playgrounds exists to make is not a comparison.
    force_field_options={"bead_diameter": 2.0},
    scenario=hex_patch(n_rings=1, a=1.0, box=4.2,
                       view_center_z=0.0,
                       settle_steps=300),
    mode="game",
    control=Control(
        atom="first",           # hex_patch orders the centre site first
        # Still the plane the camera faces, and it still matters: it is what
        # `puller_state` projects the bead's position and velocity onto for the
        # velocity-damping half of the force feedback, and it is what the default
        # torque axes below are chosen against.
        plane="xz",
        drive="torque",
        # The default pair, written out because this is the playground that
        # establishes it: (+y, -x) is the trackball mapping for an xz plane seen
        # from -y. Stick right tips the director right; stick forward tips it away
        # into the screen. Neither is the axis the directors start along (+z) --
        # a torque about the director itself does nothing.
        torque_axes=("y", "-x"),
        # The patch's own yaw_torque, which is picked so that a firm twist carries a
        # director over the tilt term's barrier at 45 degrees and flips it, and a
        # gentle one just deflects and springs back. That is the whole feel of this
        # playground, so it is the same number.
        max_input_torque=1.0,
        confine=False,          # no plane, no leash, no net -- see the docstring
    ),
    observables=["mean_tilt_deg", "thickness", "coordination"],
    params={},
    # DELIBERATELY THE SAME EXPOSURE AS mesomem_patch -- the same two dials, the
    # same absent plots -- because the comparison IS the lesson here (see the
    # module docstring: these are the same seven beads and the same force field,
    # and Tab between them is the experiment). A panel that rearranged itself
    # between the two would hide the one thing that changed.
    lesson=Lesson(
        title="Twist",
        claim="Tilt resists a twist -- and past 45 degrees the director flips over.",
        instruction="Twist the centre director until it snaps to the other side.",
        hook="Seven beads hold together. Does a real membrane stay flat?",
        everyday_params=("k_tilt", "k_splay"),
        plots=False,
    ),
    presets={
        # mesomem_patch's, verbatim: the same four settings mean the same four
        # things here, and "floppy" is if anything more legible under twist than
        # under pull.
        "paper": {},
        "floppy": {"k_tilt": 2.0, "k_splay": 0.1},
        "rigid": {"k_tilt": 40.0, "k_splay": 2.5},
        "isotropic_only": {"k_tilt": 0.0, "k_splay": 0.0, "wc": 0.0},
    },
    temperature=(0.0, 0.5),
    temperature_default=0.001,
    melt_temp=0.3,
    particle_radius=0.5,
    reduced_units=True,
    trajectory_smoothing=True,
    render_style=STYLE,
)
