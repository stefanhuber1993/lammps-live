"""MesoMem membrane wrapping a rod -- steer a bacterium into a membrane.

The membrane of `mesomem_sheet` with one rigid rod added: a long, non-spherical
particle -- think of a rod-shaped bacterium against a cell membrane -- interacting
with the beads through Pietro Sillano's `rod_lj` pair style. That style is a
Lennard-Jones between a bead and the rod's AXIS SEGMENT, applied at the closest
point on it, so the rod feels a torque as well as a force and the membrane can
genuinely wrap it rather than just be dented by it.

WHAT THERE IS TO DO. The rod starts a little above the membrane, out of reach.
Bring it down (the joystick's two axes slide it in the world xz-plane, the same
control plane as the patch, drawn as the net) until adhesion grabs it. From there:

  * push it in, and the membrane tents around the body -- the reaction on the
    stick is the membrane's own resistance, recovered from the pair style;
  * twist (Q/E, or the stick's twist axis) to rotate the rod within the plane.
    Lying flat is what adhesion wants and what gets wrapped; stand it on end and
    you are pushing a blunt tip through a bilayer, which needs far more force;
  * once it is in, let go of the drive (B, or the trigger) and watch what the
    membrane does with it on its own. At the reference conditions it holds the rod
    in a groove with about a hundred beads on it; turn `eps_rod` up and the beads
    climb further round, turn it down and the rod barely dents the surface.

WHAT THERE IS TO WATCH FOR, in order. Sideways engulfment first: the rod lies
flat, the beads climb its flanks and close over the top of it. Then a NECK -- the
membrane pinching shut above the rod rather than merely draping over it. And then,
if it gets that far, the rod standing UP inside the pit it has made, because a
vertical rod inside a closed invagination costs less membrane area than a
horizontal one does. Nothing here forces that sequence; what this file guarantees
is that there is room for it -- the leash reaches two rod lengths down, the
container is deep enough for the rod to turn end-over-end at the bottom of it, and
the cell is wide enough that the dimple dies away before it meets its own image.

THE MEMBRANE IS AT CONSTANT LATERAL PRESSURE, not in a fixed cell, and that is
what makes an invagination possible at all: covering a rod costs area, and in a
frozen periodic cell the only place that area can come from is stretching the
lattice, so the membrane dents instead of engulfing. A barostat runs throughout
here -- `fix baro membrane press/berendsen x 0 0 <damp> y 0 0 <damp> couple xy
dilate partial`, installed in RodOnSheet.post_control_settle and never removed, as
it is through all three rod phases of the reference deck -- so the projected area
shrinks as the wrap grows. You can watch it: the cell is visibly smaller by a
couple of per cent once the rod is in. `baro_press` is the dial for putting the
membrane under tension instead, which suppresses wrapping.

BUT THE BAROSTAT IS NOT WHAT MAKES THE MEMBRANE FEEL STIFF, and it is worth
saying so because it is the natural suspect -- twice over, since the barostat's own
relaxation time was for a while set here in the belief that it was. Three things
resist a wrap; one of them is the physics, one was an artefact worth an order of
magnitude, and one turned out not to resist at all:

  `k_plane`  the sheet's plane-centring spring, pulling every bead back toward
    z = 0. It is bookkeeping on a flat sheet and a SUBSTRATE on this one -- an
    invagination is a piece of membrane leaving the plane. Turned down by two
    orders of magnitude below (see the scenario), which is where most of the
    "it will not deform" went.
  `baro_damp_run`  how fast the cell may give up area. This was set here for a
    while, an order of magnitude slower than the sheet's, on the argument that
    giving area away slowly is what makes a membrane feel stiff to push into.
    MEASURED, IT IS NOT -- and the same measurement found what the slow cell was
    really costing, so both halves are worth writing down.

    Driving the rod straight down at full stick for 32 tau (400 frames), from the
    height it starts at:

        baro_damp_run     5.0     1.0     0.2
        rod centre       2.58    2.52    2.46      (started at 6.04)
        |F| in the hand    32      37      46
        cell area        99.4%   98.3%   97.8%

    The depth differs by a tenth of a sigma in three and a half, and the hand feels
    MORE from the quick one rather than less (a deeper rod is a rod with more beads
    on it). A wrap is limited by bending and adhesion -- which is what the section
    above says -- not by how fast the box may shrink, and the 2.2% of area the
    quick one gives up is where the `reference` preset says a settled wrap leaves
    it, against 0.6% for the slow one.

    What the slow cell DID do was quietly stop the membrane being tension-free
    whenever the temperature slider moved. The tension-free cell at T = 0.2 is 11%
    wider than the built one; ramping there and standing still for 66 tau, the
    barostat at 5 tau reached 1.8% of that with the lateral pressure stuck between
    +0.08 and +0.02, and the pressure plot never came back down. That is a
    laterally COMPRESSED membrane, which is exactly the state HexSheet's docstring
    works through: gamma < 0, long undulation modes with negative stiffness, a
    standing ripple instead of thermal fluctuation. At the sheet's 0.2 the same
    ramp settles at the tension-free cell (1.112) with the pressure back to +0.001
    inside ~500 frames.

    So it is the sheet's number now, and this file does not override it. Above
    `melt_temp` the cell then inflates without bound instead of holding still --
    honest rather than a fault, and the same thing the sheet does, for the reason
    HexSheet gives: a melted membrane has no cohesion left to hold at zero tension.
  `k_tilt`  the membrane's own bending modulus, and the one that should resist:
    it is the real physics of the wrapping transition, it is on the everyday
    slider, and turning it down is how you find where the transition sits.

Where the wrapping transition sits -- how much adhesion it takes to make the
membrane pay the bending cost of covering a rod of a given radius -- is a live
question rather than something this file can assert. One thing worth knowing
while looking for it: a rod already held by adhesion cannot be twisted, because
the restoring torque on the arc is bigger than the steering. That clamping IS the
wrap.

The rod is drawn as a mottled capsule rather than in the membrane's own colours,
because it is not made of membrane: see `body_material` in STYLE below.

The HUD's three numbers are the wrapping story: how deep the rod sits (negative
once the mean membrane surface has closed over its centre), how many beads are
touching it, and whether it is lying flat or standing up.

THE WHOLE MEMBRANE IS DRAWN, near rows included. This scene used to cut the near
half of the cell away (`section_min`) because a monolayer is opaque and an
edge-on camera puts those rows exactly in front of the rod. The cut is gone: the
foreground beads ARE the membrane, and hiding them permanently to make one camera
angle work is the app deciding something the viewer is better placed to decide.
What replaces it is a camera lifted far enough above the plane to see over the
near rows (`view_elevation_deg` below), and the joystick's thrust lever, which
slides a slab through the box on demand (lammps_live/view_slice.py) -- so a
section is one lever away when the wrap needs to be read in profile, and the rest
of the time the picture is the whole membrane.

Units are the paper's LJ-reduced units (sigma = eps = m = 1). The collaborator's
original LAMMPS deck is kept beside the pair style, at
`forcefields/mesomem_ff/planar_wrapping_rod.lmp`.
"""
from ..playground import Control, Lesson, Playground, rod_on_sheet
from ._knobs import HEAT
from ..render_style import DEFAULT_STYLE

# The sheet's look, with the two depth effects pulled back. The subject here is a
# single object at a known distance -- the rod, and the dimple under it -- not the
# receding surface the sheet playground is a picture of, so a strong tilt-shift
# blur would soften exactly the contact the demo is about.
#
#   DEPTH OF FIELD  focused mid-scene, where the rod is: the cell's 46 sigma of
#     depth lies mostly along the view axis, so the far rows recede into a soft
#     horizon and the near ones blur out of the bottom of the frame, which is what
#     keeps the near half readable now that it is drawn rather than cut away --
#     the rod's own plane is the sharp part, and the beads in front of it are
#     softened rather than removed.
#   NO SECTION  every bead is drawn. The near rows do stand between the camera
#     and the rod, which is what the camera's elevation is for; a cut on demand
#     is the thrust lever's job, not the playground's.
#   PERIODIC IMAGES  OFF, unlike the sheet. The sheet tiles its cell because 900
#     beads is a small raft and the copies are what make it read as a piece of
#     something endless. This cell is 3600 beads and 54 sigma across, the camera
#     frames the middle third of it, and the membrane already runs off all four
#     edges of the frame -- so the copies would be paying for geometry nobody can
#     see, and every one of them would carry another rod. One cell, drawn once.
#   BOX OUTLINE  off with them, for the same reason: at this size the container's
#     near face is behind the camera and its far face is off in the haze, so the
#     outline is four lines nobody can connect into a box.
STYLE = DEFAULT_STYLE.varied(
    periodic_images=(0, 0, 0),
    box_alpha=0,
    # NO CUT. `section_min=None` is the default and is left at it deliberately --
    # see the docstring. The camera's elevation and the viewer's own slab lever do
    # the job the fixed cut used to.
    net_alpha=150,
    dof_focus=0.5,
    dof_range=1.5,
    dof_bokeh_px=4.0,
    cue_end=0.90,
    cue_strength=0.55,
    ao_strength=9.83,           # contact darkening where the beads meet the rod
    outline_strength=12.0,
    outline_edge_fraction=0.90,
    # THE ROD IS NOT A BEAD, so it is not painted like one. Its body is drawn in
    # its own mottled material -- see RenderStyle.body_material for the two
    # reasons the bead colourings cannot serve it, the second of which is a hard
    # one: on the energy ramp the rod's own potential runs to several hundred eps
    # against a bead's single digits, so it sits pinned at the bright end in every
    # configuration and the brightest object in the frame is the one whose colour
    # means nothing. A cell wall instead, whichever colouring the beads are in.
    body_material="bacterium",
).on_light()

# Rod geometry, in sigma. Repeated here because three declarations below need to
# agree with each other: the force field's dials, the height the scenario places
# the rod at, and the leash that has to reach from there to the membrane. The
# reference deck's own numbers (L = 5, D = 3).
ROD_LENGTH = 5.0
ROD_RADIUS = 1.5
# Comfortably outside the rod-membrane cutoff (~3.24 at this radius), and inside
# the leash below -- the leash is centred on the origin, so a rod placed above it
# would be yanked down to the limit on the first frame. tests/test_rod_wrapping.py
# pins both, via RodOnSheet.verify_reach.
ROD_HEIGHT = 3.5
# THE NET, and therefore three other numbers. Twice what it was (7, 5) -- see the
# leash comment below for why -- and written here because the container has to hold
# it, the camera has to frame it, and the Control has to declare it, and those three
# drifting apart is how the rod ends up driven somewhere the picture does not go.
LEASH = (14.0, 10.0)

PLAYGROUND = Playground(
    name="MesoMem membrane + rod (3D)",
    description="Steer a rod-shaped 'bacterium' into a MesoMem membrane and feel "
                "it wrap: adhesion vs bending, live.",
    force_field="mesomem_rod",
    # Beads at diameter = sigma, as on the other sheets, so the overlapping
    # monolayer reads as a continuous membrane rather than a raft of marbles. The
    # rod's mass is the force field's business (see MesoMemRod.__init__): heavy
    # enough to feel substantial against a bead, light enough for a stick to
    # steer, which the reference deck's density matching is not.
    force_field_options={"bead_diameter": 1.0, "rod_mass": 6.0},
    # 3600 beads over 54 x 47 sigma. Four times the sheet's count, and the size is
    # the reason: an invagination is a long-ranged deformation, and at 24 sigma
    # across the rod was wrapping into a cell barely three body-lengths wide, so
    # the dimple met its own periodic image before it had decayed. This gives it
    # room to die away, and gives the barostat enough membrane that the area it
    # gives up to a wrap is a small strain rather than a visible squeeze.
    scenario=rod_on_sheet(
        n_cols=60, n_rows=60,
        # THIS FORCE FIELD's TENSION-FREE SPACING, measured. The reference deck's
        # lattice is 0.9, and this used to be too, on the reasoning that the
        # barostat settle would relax it to whatever this force field actually
        # wants. It never got there. 0.9 is 26% too large in AREA: left to
        # converge, the cell contracts to 0.861 of what it starts at, and the
        # settle's 1000 steps moved it by under 1% of that. So the whole demo ran
        # with the membrane stretched, at a lateral tension of -0.07 that drifted
        # for as long as anyone watched -- and lateral tension is exactly what
        # SUPPRESSES wrapping (see RodOnSheet, and `baro_press`, which is the dial
        # for asking for it on purpose). The rod was being pushed into a membrane
        # pulled taut by an accident of the starting lattice.
        #
        # Starting at the answer costs nothing and leaves the settle a fraction of
        # a percent to remove, where converging from 0.9 costs ~4000 steps and
        # eight seconds of build time on 3600 beads. It is a starting point rather
        # than a claim in the other direction now: it is right for the parameters
        # this file ships, and every one of those is a live slider, which is
        # precisely why the barostat below has to keep running anyway.
        a=0.775,

        # --- two housekeeping terms turned down, because on THIS scenario they
        # are the thing that reads as "the membrane refuses to be deformed" -----
        #
        # `k_plane` is the sheet's plane-centring: a per-bead spring pulling every
        # bead (except the driven one) toward z = 0. On the sheet playground that
        # is harmless bookkeeping against slow drift. Here it is a SUBSTRATE. An
        # invagination is exactly a piece of membrane leaving the plane, and at the
        # sheet's 0.1 a bead three sigma down feels 0.3 of restoring force pushing
        # it back up -- comparable to the adhesion holding it on the rod, applied
        # to every bead in the dimple at once. It is not the barostat resisting the
        # wrap (that runs throughout, see RodOnSheet.post_control_settle); it is
        # this. Kept, but an order of magnitude weaker, which is still enough to
        # stop the sheet wandering out of the frame over thousands of tau.
        k_plane=0.01,
        # `baro_damp_run` USED TO BE SET HERE, at 5 tau against the sheet's 0.2, and
        # it is deliberately not any more: the slow cell was measured to do nothing
        # for the wrap and to break the temperature dial. The numbers are in the
        # module docstring above; the value now comes from HexSheet, like the rest
        # of the barostat.
        # DEEP ENOUGH FOR THE WHOLE STORY AND NO DEEPER, and the second half of
        # that is not tidiness -- it is measured.
        #
        # The floor: the wrap is meant to run past a dent (engulfment sideways,
        # then a neck closing over the rod, then the rod standing UP inside the
        # invagination). Standing up costs half a rod length -- 2.5 -- of headroom
        # at whatever depth it has reached, and the leash below reaches 10 sigma
        # down, so nothing the demo can ask for goes past 12.5. (The sheet's own
        # 4.0 is for a bead being lifted out of a lattice; this is a bacterium
        # being swallowed.)
        #
        # The ceiling: an emptier container is not free. The pair list is identical
        # -- 97,200 pairs, 27 a bead, whatever the depth, since the membrane is a
        # monolayer and the extra volume is vacuum -- and the pair LOOP still
        # varies by 20% with it, which is a cache-locality effect of how LAMMPS
        # bins and sorts the atoms rather than any extra work. Measured on this
        # playground, interleaved: 26 sigma deep runs a 16-step chunk in 39 ms
        # (2.5 tau/s) and 30 sigma in 48 (2.1). So the rule for this number is
        # "the least the demo needs", not "round it up for safety".
        z_half=13.0,
        # LIFT THE MEMBRANE OFF THE MIDPLANE, because this scenario's interesting
        # direction is DOWN and it was spending half its container on headroom
        # nothing uses.
        #
        # The rod needs 3.5 sigma above the membrane -- that is the whole of what
        # it wants up there, just enough to start out of contact -- and everything
        # after that is depth to invaginate into. With the plane at 0 the leash
        # gave it 10 sigma down, two rod lengths, and 10 sigma up that nothing
        # ever went near. At 5.0 the same leash gives 15 down (three rod lengths)
        # and 5 up, which is still 1.5 sigma of room above where the rod starts and
        # comfortably clear of the rod-membrane cutoff, so it can still be lifted
        # off the membrane entirely. RodOnSheet.verify_reach pins all three of
        # those, and tests/test_rod_wrapping.py runs it.
        #
        # It costs nothing. The container is the same depth as before -- which
        # matters, because an over-deep box costs ~20% in the pair loop for no
        # extra pairs (see `z_half` below) -- and the leash and the frame are
        # centred on the ORIGIN rather than on the membrane, so raising the plane
        # also lifts the membrane's image into the top of the picture and puts the
        # space the invagination happens in where the eye is already looking.
        plane_z=5.0,
        settle_steps=1000,
        # HOW FAST THIS RUNS, and where the limit actually is. Two things set the
        # pace and only one of them is here.
        #
        # The step size is fixed by STABILITY, not by taste: measured, dt = 0.0075
        # and 0.01 both blow the membrane up at the top of the adhesion dial with the
        # temperature raised, where 0.005 survives. So it stays.
        #
        # The steps per frame (16 at these two numbers) buys picture smoothness and
        # nothing else. LAMMPS costs the same per step whatever the chunk size, so
        # halving the chunk halves the frame time and halves the physics per frame:
        # the wrap takes the same wall-clock time to form either way, it just arrives
        # in more frames. 0.08 tau a frame lands at ~30 fps on 3600 beads.
        #
        # WHAT DID MOVE THE PACE was the pair style. Measured on this playground, a
        # 16-step chunk was 44 ms, 92% of it inside `Pair` -- and a good quarter of
        # that was the isotropic branch re-deriving four things per pair that depend
        # only on the two atom TYPES: a sqrt of cutsq, a divide, a sin(), and a libm
        # pow(). All four are now hoisted or replaced (see
        # mesomem_ff/pair_membrane_sillano_v2.cpp's compute and init_one), the
        # numbers agree to a few ULP (`--verify` pins that), and the chunk is 32 ms:
        # 1.8 -> 2.5 tau/s at the same system size. What is left is genuine
        # arithmetic over 27 neighbours a bead; the next real step up would be
        # vectorising the neighbour loop, or fewer beads.
        timestep=0.005,
        sim_time_per_frame=0.08,
        rod_height=ROD_HEIGHT,
        # Lying flat, along the control plane's horizontal axis -- the orientation
        # adhesion wants, so the demo starts from the interesting configuration
        # and twisting away from it is what costs. `(0, 0, 1)` starts it on end.
        rod_axis=(1.0, 0.0, 0.0),
        # No diffusion tracer: on this playground the rod is the thing to watch,
        # and a second highlight elsewhere on the membrane only competes with it.
        tracer_fraction=None,
        # STAND BACK, because the net is now twice the size it was. `view_span`
        # scales the camera's DISTANCE as well as the zoom, so this is the same
        # picture from further out: what has to fit in the frame is the whole of
        # the rod's travel -- 28 sigma of net across and 20 down -- and at the
        # old 0.38 (a 20-sigma window on a 54-sigma cell) the rod left the frame
        # long before it reached the leash. 0.75 frames roughly the net, with
        # membrane still running off both sides. Aimed straight at the middle,
        # since there are no receding copies to leave room for.
        view_span=0.75, view_aim_ahead=0.0,
        # NOT edge-on any more, and this is the price of drawing the near rows.
        # At 8 degrees a monolayer is a wall: the beads between the camera and the
        # rod sit at the height the rod is being pushed to, and with no section cut
        # they hide it completely. 24 degrees looks over them -- steep enough that
        # the invagination is a visible pit in a surface rather than a line, shallow
        # enough that the DEPTHS the demo is about (how far the rod has sunk, how
        # far the beads have climbed) still project onto the screen instead of
        # foreshortening away. Push the thrust lever to get the profile back.
        # See RodOnSheet.camera.
        view_elevation_deg=24.0,
        # And frame the WHOLE net vertically, not just the rod's clearance. The
        # default here is `rod_height + 1.5`, which was the travel back when the
        # leash was 5; with the leash at 10 it would put the bottom half of the
        # drawn net -- and the rod, once it is pushed down there, which is the whole
        # demo -- below the bottom of the picture. A sigma of margin past the limit
        # so arriving at it is something you watch rather than something that
        # happens off screen. This costs nothing in zoom: at the window's aspect
        # ratio the in-plane extent above is what binds the fit.
        view_z_half=LEASH[1] + 1.0,
    ),
    mode="game",
    control=Control(
        atom="last",            # rod_on_sheet appends the rod after the sheet
        plane="xz",
        # TWICE WHAT IT WAS (7, 5), and the reason is the whole scenario rather
        # than comfort: a dent needs a couple of sigma of travel, an invagination
        # needs the rod to go DOWN past the membrane's own surface until a neck can
        # close over it, and then to stand upright inside what it has made. 10
        # sigma of z is two rod lengths of depth, which is enough room for all
        # three stages and still leaves the container floor (z_half = 13 above)
        # clear -- the rod sweeps to 12.5 when it stands end-on at the limit. The 14 across is the same doubling: it is what lets a wrapped rod
        # be dragged sideways far enough to see whether the invagination travels
        # with it or the membrane hands it on.
        leash=LEASH,
        # A rod pushing on ~50 beads at once meets far more resistance than a
        # single bead does, and it weighs a dozen beads. Both want more authority
        # than the sheet's 7.
        max_input_force=30.0,
        # It is heavy, so it wants more damping than a bead to stop it coasting
        # after the stick is released.
        damping_default=8.0,
        damping_range=(0.0, 20.0),
        # Steering is an angular-velocity kick, so the rod's inertia does not
        # blunt it -- and at this scenario's 20 steps a frame the patch's 1.0
        # would spin the rod most of a radian per frame. 0.3 turns a free rod
        # about half a revolution a second, which is a rod turning rather than
        # flailing, and leaves adhesion able to win: a rod pressed into the
        # membrane holds its orientation against a sustained twist (measured:
        # ~60 of restoring torque against this steering), which is the clamping
        # that a wrap IS. Raise it and the rod plows through instead.
        yaw_torque=0.3,
        rot_damp=0.85,
        # Display normalisation for the reaction-torque arc, not a limit: the
        # restoring torque here is adhesion pulling on a lever arm of half the
        # rod, which runs an order above the membrane's own tilt term. Measured
        # peak while twisting a rod pressed into the membrane: ~60.
        reaction_torque_max=60.0,
        grid_step=1.0,
    ),
    # The three numbers a wrap is told in. `coordination` would work here too --
    # the rod finds its own long-ranged pairs rather than widening the membrane's
    # list to reach them (see MesoMemRod.extended_pairs) -- but it says nothing
    # about the rod, which is the subject.
    observables=["rod_height", "rod_contacts", "rod_tilt_deg"],
    # The membrane is one sheet and the rod is one body, so cluster colouring has
    # two colours to give and the species colours already say which is which.
    bead_colors=("director", "energy"),
    params={"rod_length": ROD_LENGTH, "rod_radius": ROD_RADIUS},
    param_ranges={
        # The wrapping transition is the thing worth finding, and it sits well
        # below the membrane's own default stiffness -- a floppy membrane wraps a
        # rod that a stiff one merely holds.
        "k_tilt": (0.0, 30.0),
    },
    # WHERE THE DEMO BECOMES BIOLOGY. Everything up to here has been a material;
    # this is the material doing the thing it exists to do, and the claim says so
    # in the audience's own words rather than in the model's.
    #
    # HEAT AGAIN, and it earns its place here more than anywhere: a wrap is the
    # membrane FLOWING around the rod, and a cold membrane deforms like a sheet of
    # foil instead. Same numbers as the sheet's, from the same dial.
    lesson=Lesson(
        title="It wraps",
        claim="The membrane sticks to the rod and bends around it. Cells eat this way.",
        instruction="Steer the rod into the sheet. Cut it open with the lever to look.",
        hook="Three thousand beads on a laptop. Does it hold at ten times that?",
        hero_knobs=(HEAT,),
    ),
    presets={
        # The reference deck's conditions: eps = 3, L = 5, D = 3, paper moduli.
        # Brought into contact and released, the membrane invaginates until the
        # rod's centre sits at the mean surface, with the beads closed most of the
        # way round it (measured: height +0.07, ~120 touching, and the cell 2%
        # smaller than it started -- that shrinkage IS the area the wrap took).
        "reference": {},
        # Adhesion too weak to pay for any bending: the rod rests on the surface
        # and dents it.
        "weak_adhesion": {"eps_rod": 0.8},
        # More adhesion is NOT more wrapping, which is worth seeing: at the top of
        # the dial the rod is gripped so hard at first contact that the beads pile
        # onto its flanks instead of closing over it, and it ends up sitting
        # HIGHER than at the reference (measured: +1.4 against +0.07). The
        # transition is a balance, not a monotone.
        "strong_adhesion": {"eps_rod": 8.0},
        # Strong adhesion against a membrane with no orientational stiffness to
        # resist it: the beads go all the way round (measured: ~210 touching,
        # against ~120 at the reference). A plain isotropic fluid rather than a
        # bilayer, so this is the adhesion-wins limit rather than physics.
        "engulfed": {"eps_rod": 8.0, "k_tilt": 0.0},
        # Twice the radius at the same adhesion: twice the curvature to pay for,
        # over more area. The comparison that shows the transition is adhesion
        # AGAINST bending rather than adhesion alone.
        "fat_rod": {"rod_radius": 3.0},
        # A short, stubby rod, which is nearly the spherical-nanoparticle limit
        # the wrapping literature usually treats.
        "stubby": {"rod_length": 1.5},
    },
    temperature=(0.0, 0.5),
    temperature_default=0.001,
    melt_temp=0.3,
    particle_radius=0.5,
    reduced_units=True,
    # The rod's panel is dominated by adhesion: it is one particle in contact with
    # tens of beads at once, where a bead touches a dozen, so its share of the
    # energy runs two orders above the force field's per-particle scale and the
    # panel would otherwise sit pinned at full deflection. Measured: the adhesion
    # bar reaches ~-370 on a rod driven hard into the membrane.
    pulled_energy_scale=450.0,
    # The energy panels are a pass over every pair, and at 3600 beads that is the
    # one thing in the frame big enough to be felt as a hitch when it lands. The
    # aggregate barely changes frame to frame, so it is evaluated half as often as
    # the default. (The other half of that problem -- the rod's long reach
    # dragging the whole pair list out with it -- is solved in the force field,
    # by MesoMemRod.extended_pairs.)
    analysis_energy_every=8,
    # The membrane ripples thermally at every wavelength at once; smoothing the
    # drawn beads leaves the wrap itself -- a slow, collective change -- legible
    # without the shimmer on top. Advanced slider, 0 (off) by default.
    trajectory_smoothing=True,
    render_style=STYLE,
)
