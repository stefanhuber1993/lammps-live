"""Two MesoMem beads -- one in your hand, one nailed down. The whole force field,
one interaction at a time.

This is the tutorial's opening slide, and it now has something to point at. A
directored bead you drive, with the control net it moves on drawn underneath it and
no box outline to distract from either -- and, at the net's right-hand edge, a
SECOND bead of exactly the same kind, held absolutely still. That partner is the
difference between a controls demo and a physics one: a lone bead has no
neighbours, so every pair term in the force field is silent, and with one
neighbour every term speaks in turn as you close the gap.

WHAT TO DO WITH IT. Push right, and read the numbers written between the two
beads -- the whole force field, term by term, as a function of the one separation
and the one angle your hands control (see WHAT IS DRAWN BETWEEN THEM, below).
Measured on the real pair style at the paper's coefficients, driving straight in
with the director broadside, this is the story those numbers tell:

    r         2.0     1.8     1.6     1.4     1.2     1.0     0.9
    U_vdW   -0.00   -0.02   -0.12   -0.41   -0.80   -1.00   -0.95
    F_vdW   -0.02   -0.21   -0.91   -1.89   -1.79   -0.00   +1.29

(F negative pulls the pair together, positive shoves it apart.) So: nothing at
the start, worth saying out loud because the bead starts at 2.0 and every term
there is zero to two decimals -- the pair is inside rc = 2.5 but the attraction at
that range is a thousandth of eps, and the tutorial's old claim that it "turns on
at 2.5" was a statement about the cutoff, not about anything a hand can feel. The
pull arrives between 1.8 and 1.6, and it arrives fast: by 1.4 it is the strongest
attraction the pair ever produces, which is well over the 1.5 your own push is
capped at. The well bottom is at 1.0 -- exactly sigma, exactly -1 eps, the force
field's own two units in one place -- and inside that the 4-2 core turns the whole
thing round and refuses, and the leash cannot drive through it. That is worth
feeling once.

Then twist. The tilt and splay terms are zero in everything above, not because the
beads are too far apart but because the pair as it sits costs nothing in either:
tilt is built on (n . rhat) and both directors are perpendicular to the bond, splay
on (n_i . n_j) and the two are parallel. Turn the driven director with Q/E (or the stick's twist axis) and both terms come
alive, and they are the ones with something to say about ORIENTATION -- at 1.2
sigma with the director at 45 degrees the tilt term holds +0.57 eps, pushes +2.05
along the bond (outward: turning the director makes the pair repel where it was
attracting), and puts -1.15 of restoring torque on the bead trying to lay it flat
again. Splay is the same shape an order of magnitude down, because k_splay = 1
against k_tilt = 12. The van der Waals term, meanwhile, does not move at all under
a twist and its torque column stays at exactly zero -- it depends on nothing but r.
That contrast, seen in one glance, is what "orientational" means.

WHAT IS DRAWN BETWEEN THEM. The pair itself is the instrument (`pair_annotation`
below, and playground/pair_probe.py for the arithmetic):

  * the bond, with the separation on it, drawn solid while the pair is inside rc
    and DASHED outside it -- so "every number is zero because they cannot see each
    other" is a thing you can see rather than infer;
  * sigma, wc and rc as rings around the FIXED partner, coloured by the term each
    one belongs to, so the shells are visible before the bead reaches them. They
    are drawn in the control plane, which is the plane the bead is held in, so
    where a ring crosses the bead's path is exactly where the crossing happens;
  * and the callout: one row per additive term, with its energy, the force it
    contributes along the bond, and the torque it puts on the driven bead. The two
    energy panels in the upper left are switched OFF here (see
    PlaygroundSystem.get_pair_annotation): on a scene of one pair they and this are
    three copies of the same three numbers.

The two HUD numbers are the axes of that story: the separation (compare it against
sigma = 1, wc = 2, rc = 2.5) and the angle between the driven bead's director and
the line joining the pair, which is the quantity the tilt term is built on -- 90
degrees is what a flat membrane looks like locally.

WHY THE PARTNER IS FIXED, AND HOW. Fixed because the reading has to hold still:
any motion of the partner turns "the force field at this separation and this
angle" into a two-body problem whose answer moves while you are reading it. How, is
`BeadAndPartner.integrator_commands` -- the partner is simply left out of the
integrator group, so nothing integrates the forces and torques it accumulates. Its
position and its director are exactly where they were put, with no restraining
spring to fight and no spring timescale added to the deck.

WHY A HEX PATCH WITH NO RINGS UNDER IT. `BeadAndPartner` extends `HexPatch`, and
`n_rings=0` is exactly one site -- the centre -- so the driven bead, the camera,
and the framing dials are all the patch's, one shell fewer. The patch's three soft
correction forces are off: they act on everything EXCEPT the driven particle, which
here is only the fixed partner, and pushing on something that cannot move is not a
force. They are set to zero below as well, so the declaration says what it means
rather than relying on that.

WHAT IS DELIBERATELY SMALL -- AND WHAT WAS TOO SMALL.

  THE FORCE  a bead in a membrane is held by its neighbours, and the patch's 4.0
    is picked to win against them. A lone bead is held by nothing, so the same
    number flings it to the leash before the hand has registered pushing. 1.5
    against the default viscous damping settles it to under half a sigma per tau:
    slow enough to steer deliberately, fast enough not to feel broken -- and, at
    half again the 1.0 this scene opened with, enough to push INTO the core and
    hold there rather than being turned away at the first sign of it. The table
    above is why that mattered: the attraction alone peaks at 1.89.
  THE TORQUE  same argument, more sharply. The membrane's tilt term is what
    absorbs a twist on the patch, and it is absent here; the patch's yaw_torque of
    1.0 would spin the director roughly 24 degrees per frame with nothing to stop
    it. 0.45 -- again half again what this scene opened with -- turns the director
    once in about two seconds of real time, and holds it at 45 degrees against a
    restoring torque of order 1 instead of sagging back to broadside.
  WHAT PUSHES BACK, though, is deliberately LOUD, and that is the other half of
    the same change: one pair's forces are a tenth of a membrane's, so the scene
    carries its own force-feedback calibration rather than the shared one. See
    PAIR_FEEDBACK below.
  THE BOX AND THE NET  sized by the FORCE FIELD's reach, not by taste. The net's
    right-hand half is 2 sigma, the partner sits at that edge, and the driven bead
    starts at the net's centre -- so it starts exactly wc away, where (see the
    table) every term reads zero to two decimals and there is a real "watch it turn
    on" to watch. Driving LEFT instead takes the pair past rc, where the connector
    goes dashed and the zeros are exact rather than merely small. The framing
    rectangle below is derived from the same number rather than typed twice.

Units are the paper's LJ-reduced units (sigma = eps = m = 1).
"""
from ..mdsystem import ForceFeedbackProfile
from ..playground import Control, Playground, bead_and_partner
from .mesomem_patch import STYLE as PATCH_STYLE

# The net's right-hand edge, which is also where the partner is and also half the
# framed width. One number, because three declarations have to agree: put them out
# of step and either the bead starts already inside the interaction (nothing to
# watch turn on) or the partner sits outside the frame.
#
# 2 sigma, which is wc: the pair starts with every term zero to two decimals (the
# van der Waals energy there is -0.001 eps) and with one sigma of empty net to
# cross before anything happens. Further out would be more honestly "outside rc"
# and would buy nothing -- the potential is already silent here, and every extra
# sigma is both empty net to cross and smaller beads in frame.
NET_HALF_WIDTH = 2.0

# The patch's look, with five things changed for a scene that is two spheres.
#
#   THE RING IS KEPT FOR THE RELEASED STATE ONLY. The cyan ring answers "which of
#     these am I holding?" -- and now that there ARE two beads on screen that is a
#     real question, but the driven one is also the one with the arrows and the
#     torque arcs on it, so the ring would be a fourth mark on an already-busy
#     bead while the partner sits bare. Released, it is the only mark left and it
#     says the thing the picture does not: that you have let go.
#   NO BOX. There is nothing for a container outline to give scale to, and at this
#     zoom its near face crosses the frame behind the beads. `box_alpha=0` here and
#     not upstream because these values come AFTER `.on_light()`, which sets its
#     own box and net alphas -- so a varied() before it (as mesomem_patch does) is
#     overwritten by it, and only a varied() after it sticks.
#   A NET THAT READS. It is what the tutorial points at when it says where the bead
#     can go, and its right edge is where the partner is, so it is taken well up
#     from the light theme's 120.
#   DEPTH OF FIELD  off entirely. Both beads are at the same depth, on the control
#     plane, so there is no depth for a blur to describe -- only a soft edge on the
#     two objects the scene is about.
#   DEPTH CUE  off, for the same reason: nothing is behind anything.
STYLE = PATCH_STYLE.varied(puller_ring="released", box_alpha=0, net_alpha=200,
                           dof_bokeh_px=0.0, cue_strength=0.0)

# HOW HARD THE PAIR PUSHES BACK, in the hand. The shared REDUCED_UNIT_FEEDBACK
# profile (playground/spec.py) is knee'd at 4.0, which is right for a membrane: a
# bead there is held by six neighbours and the load on the puller runs O(1-10). One
# neighbour is a tenth of that, and on the shared profile the whole interaction
# arrived as a murmur -- the attractive peak filled half the spring's travel and
# first contact filled none of it, because it sat under the stiffness threshold.
#
# So this scene gets its own profile, scaled to what ONE pair actually does.
# Measured on the real pair style at the paper's coefficients, driving in along the
# net with the director broadside (see the module docstring for the geometry):
#
#     r      2.0    1.8    1.6    1.4    1.2    1.0    0.9
#     F_r   0.00  +0.19  +0.90  +1.87  +1.77  -0.02  -1.30     (broadside)
#     F_r   0.00  +0.17  -0.20  -1.09  -2.68  -5.43  -6.99     (director turned 90)
#
# (positive pulls the bead onto its partner, negative shoves it off) -- so the
# whole story lives between 0.2 and 2 with the core running out to 7 beyond it.
#
#   ff_knee 1.6      the attraction's own peak, near enough. tanh(F/knee) then
#     spends its useful range over exactly that 0.2-2 window instead of over a
#     membrane's, and the core saturates -- which is the truth about a core.
#   ff_exaggeration 1.5   up from 1.3, because the thing being exaggerated is a
#     tenth the size. Together with the knee this is 2-3x the deflection per unit
#     of force the shared profile gave, everywhere it matters.
#   stiffness_threshold 0.05   first contact at r = 1.8 is worth 0.19, and under
#     the shared 0.3 the spring stayed limp right through it. The moment the beads
#     notice each other is the moment the stick should firm up.
#   stiffness_knee 0.8    so it is fully firm by ~1, not by ~5.
#
# The arrow scale rides on ff_knee too (see Renderer._draw_arrow_3d), so the red
# reaction arrow and the connector's per-term force glyphs grow with the same
# number the hand feels -- one calibration, three instruments.
PAIR_FEEDBACK = ForceFeedbackProfile(
    ff_exaggeration=1.5,
    ff_knee=1.6,
    ff_max_mag=120.0,
    stiffness_threshold=0.05,
    stiffness_knee=0.8,
    damper_min_fraction=0.10,
    damper_max_fraction=0.55,
    vel_damp_max_fraction=0.5,
)

PLAYGROUND = Playground(
    name="Two MesoMem beads (3D)",
    description="One bead in your hand and one nailed down: drive them together "
                "and feel the pair potential turn on, term by term.",
    force_field="mesomem",
    # Sphere radius = sigma, as on the patch: it fixes the moment of inertia the
    # director's swing is felt through, and the twist here is meant to feel like
    # the twist there with the membrane taken away.
    force_field_options={"bead_diameter": 2.0},
    # This scene's own force-feedback calibration, scaled to one pair rather than to
    # a membrane -- see PAIR_FEEDBACK above for the measurements it comes from.
    force_feedback=PAIR_FEEDBACK,
    scenario=bead_and_partner(
        n_rings=0,                  # one site: the driven bead, on its own
        a=1.0,
        # Room for both beads and the whole net inside it, with margin. Invisible
        # (box_alpha=0) and only a backstop -- nothing here is meant to reach it.
        box=10.0,
        # THE PARTNER: middle-right of the net. On the control plane at the same
        # height the driven bead starts at, so the approach is one joystick axis
        # and the net's own right edge marks how far there is to go.
        partner_x=NET_HALF_WIDTH,
        partner_z=0.0,
        # Along +z, as a membrane bead's would be. The driven bead therefore
        # arrives BROADSIDE to it, which is the configuration the tilt term likes
        # (see the module docstring) -- so twisting away from it costs, and the
        # cost is visible on the panel.
        partner_director=(0.0, 0.0, 1.0),
        # The patch's three soft corrections, off. They act on everything but the
        # driven particle, which here is only the fixed partner, so they would push
        # on something that cannot move -- but a tutorial scene should not have a
        # homing spring in it that the text does not mention, and switching them
        # off here is what guarantees that.
        k_center=0.0, k_align=0.0, k_home=0.0,
        # Long enough for the partner's own director to be written and the pair to
        # take one honest force evaluation, and no longer: the beads start out of
        # range of each other, so there is nothing to relax.
        settle_steps=50,
        # Framing: a rectangle in the control plane, spanning the net and a little
        # more, so the partner at its right edge is comfortably inside the frame
        # (see Camera3D.fit_to_points -- the tighter of the two axes binds, and it
        # is height at the default window). Derived from NET_HALF_WIDTH rather than
        # typed, because these two must not drift apart.
        view_center_z=0.0,
        view_half_width=NET_HALF_WIDTH * 1.25,
        view_half_height=NET_HALF_WIDTH,
    ),
    mode="game",
    control=Control(
        atom="first",               # the driven one; the partner is last
        plane="xz",
        # The x half-extent IS where the partner is, so the right-hand edge of the
        # net is the contact and the bead can be driven exactly onto it -- it never
        # gets there, because the 4-2 core stops it a little short, and finding
        # that out is the point. The z half is smaller: the interesting travel is
        # along the line joining the pair, and up/down is for coming at it from a
        # different angle rather than for going anywhere.
        leash=(1.5*NET_HALF_WIDTH, 1.0 * NET_HALF_WIDTH),
        # Half again what this scene shipped with (1.0 and 0.3), and that is the
        # whole change: the hand needs enough authority to push INTO the core and
        # hold there rather than being turned away at the first hint of it, and
        # enough twist to hold a director at 45 degrees against the tilt term's
        # restoring torque instead of drifting back to broadside. Still well under
        # the patch's 4.0 / 1.0 for the reason in the docstring -- one bead is held
        # by nothing, and the patch's numbers are picked to beat six neighbours.
        max_input_force=1.5,        # (4.0 on the patch -- see the docstring)
        yaw_torque=0.45,            # (1.0 on the patch -- likewise)
        # What fills the reaction-torque arc. The pair's restoring torque was
        # measured at 1.1 with the director at 45 degrees and 1.2 sigma out, and
        # peaks near 2.5 against the core -- so on the shared default of 6.0 the arc
        # never got past a third and the twist read as unopposed. 2.5 is the
        # measured peak: a firm twist near contact now fills it.
        reaction_torque_max=2.5,
        grid_step=0.5,              # ~12 cells across the 6-sigma net
    ),
    # The two axes of the story, and the only two observables in this app that are
    # statements about a PAIR rather than about a membrane (see observables.py --
    # they read the first two particles, which is meaningless anywhere else and is
    # exactly right here). Separation against sigma, wc and rc; and the angle the
    # tilt term is built on.
    observables=["pair_separation", "pair_director_angle"],
    # THE READOUT THIS SCENE EXISTS FOR: the three additive terms written between
    # the two beads, with the separation and the director angle they belong to, the
    # energy in each and the force each one is contributing along the bond -- plus
    # sigma / wc / rc drawn as rings around the fixed partner, so the shells are
    # visible before the bead reaches them. See playground/pair_probe.py.
    pair_annotation=True,
    # Declaring it also switches OFF the two additive-energy panels in the upper
    # left (see PlaygroundSystem.get_pair_annotation): on a scene of one pair they
    # and the connector are three copies of the same three numbers, and the copy
    # that is drawn between the beads it describes is the one worth keeping.
    presets={"paper": {}},
    temperature=(0.0, 0.5),
    # Cold, and the slider is inert -- which is worth knowing before someone reaches
    # for it in front of an audience. The bath is `mobile` minus the driven particle
    # (see BeadAndPartner.thermostat_group), and on this scene that is the empty
    # set: the driven bead is deliberately kept out of the bath so its director has
    # no thermal noise, and the only other particle is the fixed partner, which
    # nothing integrates. Left on the panel because it is the same panel every other
    # playground has, and starting at zero so it says nothing rather than something
    # wrong.
    temperature_default=0.0,
    melt_temp=0.3,
    particle_radius=0.5,
    reduced_units=True,
    # Off, and not offered: this is the one scene where an individual bead's own
    # motion IS the subject, and smoothing it would soften exactly the response the
    # tutorial is pointing at -- including the moment the attraction takes over.
    trajectory_smoothing=False,
    render_style=STYLE,
)
