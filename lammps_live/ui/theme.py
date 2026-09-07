"""Shared colors and layout constants for the pygame UI."""
import math

BG = (18, 18, 24)
CRYSTAL_COLOR = (230, 140, 40)   # fallback flat atom color; systems set their own (spec.crystal_color)
INPUT_VEC_COLOR = (80, 220, 120)   # joystick/mouse-commanded input force
REACTION_VEC_COLOR = (255, 90, 90)  # crystal's interaction-force reaction
BOX_OUTLINE = (90, 90, 100)
PANEL_BG = (24, 24, 30)
PANEL_DIVIDER = (60, 60, 68)
TEXT_COLOR = (200, 200, 200)
DIM_TEXT_COLOR = (130, 130, 140)
HEADER_TEXT_COLOR = (235, 235, 240)
GRID_COLOR = (45, 45, 52)
SLIDER_TRACK = (70, 70, 80)
SLIDER_HANDLE = (230, 230, 235)
SLIDER_HANDLE_ACTIVE = (255, 210, 90)
MELT_MARK_COLOR = (255, 110, 90)
# What the joystick is currently driving (see control_focus.py): a bright cyan
# frame around the sim viewport, or around the focused slider's row. Cyan because
# nothing else in the panel is -- the point of the marker is that it is findable
# from across a room, mid-demo, without reading anything.
FOCUS_COLOR = (60, 230, 255)
FOCUS_WIDTH = 3
# Marker for a slider's recommended "optimum" value (the paper's sweet spot),
# drawn as a distinct tick + "opt" label so it reads differently from the red
# melt marker on the temperature slider.
OPTIMUM_MARK_COLOR = (120, 230, 170)

# Faint "bond" lines between atom pairs, with opacity encoding how close their
# separation is to the system's equilibrium nearest-neighbor spacing
# (spec.lattice_spacing -- the optimal bonding distance). Alpha peaks at
# BOND_PEAK_ALPHA when a pair sits exactly at the optimum and falls off
# exponentially in either direction:
#     alpha = BOND_PEAK_ALPHA * exp(-|d - d_opt| / (BOND_FALLOFF * d_opt))
# so BOND_FALLOFF is the decay length (lambda) as a fraction of d_opt. Pairs
# whose alpha would drop below BOND_MIN_ALPHA are skipped -- both a visibility
# floor and the distance cutoff that bounds how many pairs get drawn. All
# tunable:
BOND_LINES_ENABLED = True
BOND_COLOR = (255, 255, 255)   # white
BOND_PEAK_ALPHA = 128          # out of 255 -> 50% opacity exactly at the optimum
BOND_FALLOFF = 0.10            # decay length (lambda) as a fraction of the optimal distance
BOND_MIN_ALPHA = 3             # skip lines fainter than this (also sets the effective cutoff)
BOND_WIDTH = 1                 # line thickness, px

# Glyph color for per-species atom labels (e.g. the +/- stamped on ions).
# Per-species fill colors themselves live on each system's SystemSpec
# (species_colors), since they're system-specific.
ION_LABEL_COLOR = (20, 20, 26)

# Explicit molecular backbone bonds (e.g. lipid head-tail-tail chains):
# subtle gray for ordinary molecules, bright cyan for the control lipid so
# "your lipid" and the way it points stand out.
BOND_STICK_COLOR = (95, 95, 110)
PULLER_BOND_COLOR = (110, 220, 255)

# Hydrogen-bond overlay (water demo): a lighter, cooler, thinner line than the
# solid intramolecular bonds, drawn dashed so a transient H-bond reads as
# "weak/breakable" rather than a fixed covalent stick.
HBOND_COLOR = (150, 200, 245)
HBOND_WIDTH = 1
HBOND_DASH = 6   # px on/off dash period; 0 -> solid

# Small per-system HUD lines drawn in the sim view (etch tally, water phase...).
HUD_TEXT_COLOR = (210, 214, 224)
HUD_BG = (0, 0, 0, 130)

CRYSTAL_RADIUS = 5    # fixed-pixel fallback when a system sets no physical radius
PULLER_RADIUS = 8

# The interactively-controlled ("puller") atom is drawn in its true species
# color -- a deposited Cu atom IS a Cu atom, the pulled Na+ IS a cation -- so it
# no longer masquerades as a differently-colored species. What marks it as
# "the one you control" is instead a bright ring around it (a color no atom
# fill uses, so it always contrasts) plus a small size boost. This keeps the
# picture chemically honest while still making the controlled atom pop.
PULLER_RING_COLOR = (95, 230, 255)
# ...and how it reads once released (B / joystick trigger): still marked, because
# it is still the particle the ring will grab again, but dim and cool so "you are
# not holding this" is legible at a glance rather than only in the HUD line.
PULLER_RING_FREE_COLOR = (95, 110, 130)
PULLER_RING_WIDTH = 3
PULLER_RADIUS_BOOST = 3   # extra px on the controlled atom over its species size
# Per-atom energy readout stamped under the controlled atom.
PULLER_LABEL_COLOR = (240, 240, 248)
PULLER_LABEL_BG = (0, 0, 0, 120)

# Physical-radius drawing (spec.atom_radius_A / species_radii_A) is converted to
# pixels at the live box scale, then clamped to this band so atoms stay visible
# on a big box and don't swallow the view (and the bond overlay) on a small one.
ATOM_MIN_RADIUS = 4
ATOM_MAX_RADIUS = 22
# Arrow length: soft-saturating (tanh) rather than linear, since the
# interaction force can spike well past what the input force ever reaches --
# a fixed linear scale either makes small vectors invisible or huge ones fly
# off-screen.
VECTOR_MAX_PX = 130.0
ARROWHEAD_LEN = 8
ARROWHEAD_ANGLE = math.radians(25)

# FLAT circular "torque" arrows drawn around the puller bead on a FORCE drive:
# an arc starting at the top of a ring and sweeping up to a semicircle left/right
# in proportion to the applied (green) / reaction (red) twist about the control-
# plane normal. Radius is a fraction of the puller's on-screen radius; the two
# arcs use slightly different radii so they don't overdraw each other.
#
# Flat is defensible only here. On a force drive the twist is about the control
# plane's normal, which is the axis those scenes are looked down -- so the circle
# the camera would see IS a circle. A torque drive turns the director about two
# axes at once and neither of them is that one, which is what the rings below are
# for.
TORQUE_ARC_APPLIED_RADIUS = 0.62  # x puller radius (green, user's steering torque)
TORQUE_ARC_REACTION_RADIUS = 0.82  # x puller radius (red, membrane's restoring torque)
TORQUE_ARC_WIDTH = 3
TORQUE_ARC_HEAD_LEN = 9

# TORQUE RINGS -- what a torque drive draws instead (see Renderer._draw_torque_ring).
# A real curve in the scene, turning about the axis the rotation is about, so the
# camera foreshortens it exactly as much as that plane is foreshortened. Radii are
# multiples of the bead's WORLD radius, and both are outside the bead: the glyph
# has to pass behind its own bead for the near and far parts to be tellable apart.
# Well apart from each other, too -- seen nearly edge on the two lie almost along
# the same line, and then most of what separates them is how long that line is.
TORQUE_RING_APPLIED_RADIUS = 1.45   # x bead radius (green, your steering torque)
TORQUE_RING_REACTION_RADIUS = 2.05  # x bead radius (red, the membrane's answer)
# How far the glyph advances ALONG its own axis over a full turn, as a fraction
# of its radius -- it is one turn of a right-handed screw, not a flat circle. A
# flat circle has a degenerate view: seen edge on it is a straight line, which on
# this drive is the one thing the drawing must never look like, and the second of
# the two control axes is edge on from the scenes' own camera by construction. A
# helix has no such view -- edge on it is a slanted, offset curve, and the way it
# advances is the axial vector's own direction (right-hand screw), which is the
# last thing the axial arrow was there to say.
TORQUE_RING_PITCH = 1.4
# How far round a full-scale torque goes, in turns. Three quarters rather than the
# half a flat arc used: half a turn centred on the near point is exactly the
# monotone half of the coil, which seen edge on is a slanted stroke and still too
# close to a straight arrow. Past a half turn the curve doubles back at each end,
# and a stroke that hooks back on itself is not an arrow from any angle.
TORQUE_RING_MAX_TURN = 0.75
TORQUE_RING_WIDTH = 3.6             # px at the near edge, tapering with depth
TORQUE_RING_HEAD_LEN = 11
TORQUE_RING_MIN = 0.05              # hide below 5% of full scale, to declutter
# How far the far half of a ring fades toward the scene background, and how much
# of its width it loses. Both are depth cues and both are needed: a projected
# ellipse on its own is the same ellipse for an axis tipped toward the camera and
# one tipped away, so without them the sense of the rotation is a coin flip.
TORQUE_RING_DEPTH_FADE = 0.55
TORQUE_RING_DEPTH_TAPER = 0.45

# --- 3D scene (MesoMem membrane patch and future 3D systems) ---------------
# The 3D scene's own palette -- what it is drawn ON (and so what its fog fades
# INTO), plus the box outline, the control net and the bond spokes -- is NOT
# here: it lives on each system's RenderStyle (lammps_live/render_style.py),
# because a scene can be drawn dark or light (`STYLE.on_light()`) and all of
# those have to move together. What is left below is what every 3D scene shares
# whichever background it is on: the bead colouring and the shading constants.
#
# Depth cueing ("haze"): distant beads are blended toward the scene's own
# background so the scene recedes into it instead of reading as a flat cluster.
# STRENGTH caps how much of the original color is washed out at the far plane
# (1.0 = fully background). These two are the CPU fallback's ramp only -- the GL
# path takes its cue range and strength from the RenderStyle (`cue_*`), which is
# per system and anchored to the scene's own depth extent. Here it is anchored
# to the farthest bead: nothing hazes until DEPTH_FADE_START of that distance,
# then it ramps LINEARLY to a full wash-out at it.
HAZE_STRENGTH = 1.0
DEPTH_FADE_START = 0.5   # fraction of the max bead distance at which haze begins
# Screen-space edge vignette applied to the beads of periodic scenes (the
# membrane sheet): beads fade toward the background over the outer frame margin,
# uniformly in screen space, softening the frame edge and the periodic clip seam
# without touching the box outline (drawn after the composite). 0 disables it;
# 1 fully fades the very edge to the scene background. See gl3d._COMPOSITE_FS.
EDGE_VIGNETTE_STRENGTH = 0.65
# Light direction for the shaded-sphere sprite (points FROM the light, in
# screen space: up-left and toward the viewer). Sphere shading is baked once
# from this and reused for every bead.
SPHERE_LIGHT_DIR = (-0.5, -0.6, 0.62)
SPHERE_AMBIENT = 0.28          # floor brightness on the dark side of a bead
# The director "spike" (orientation vector n_i) drawn standing out of each
# bead, like the cones in the MesoMem figures.
DIRECTOR_COLOR = (245, 214, 66)     # warm yellow
DIRECTOR_LEN_R = 1.55               # spike length as a multiple of bead radius
DIRECTOR_BASE_R = 0.42              # spike base half-width as a multiple of bead radius
# The control "net": the plane (perpendicular to the membrane, facing the
# screen) that the joystick slides the active bead along, drawn as a faint grid.
# Its colour and line alpha are RenderStyle.net_color / net_alpha; this is the
# fill alpha of the plane itself.
NET_ALPHA = 60
# The net is redrawn, a little brighter, clipped to the active bead's disc, so
# the control plane visibly cuts THROUGH that bead (a plane through a solid
# sphere's centre is otherwise hidden by its near hemisphere) -- makes clear the
# bead rides in the plane.
NET_GHOST_ALPHA = 110
# Thin in-bead director arrow (cylinder shaft + cone head) marking which way n_i
# points -- so a flip between the two equivalent normals (+n vs -n) is visible.
DIRECTOR_ARROW_COLOR = (28, 30, 40)
# Live additive-potential breakdown panel (get_potential_terms): one distinct
# colour per term, plus a neutral colour for their sum, drawn as signed bars.
POTENTIAL_COLORS = ((120, 205, 225), (250, 180, 95), (200, 140, 240))  # isotropic, tilt, splay
POTENTIAL_TOTAL_COLOR = (225, 228, 236)
POTENTIAL_PANEL_BG = (10, 12, 18, 185)
POTENTIAL_TRACK_COLOR = (58, 62, 76)
# The two-bead scene's term-by-term connector (see
# Renderer._draw_pair_annotation and playground/pair_probe.py). It is drawn IN the
# scene, over whatever background that scene has, so the panel colours above --
# which live on a dark plate of their own -- cannot simply be reused: the term
# colours are dragged toward black by PAIR_INK_DARKEN on a light ground, where the
# pastels they are read as bright accents on vanish.
PAIR_CALLOUT_WIDTH = 360        # px at UI scale 1
PAIR_CALLOUT_BG_DARK = (10, 12, 18, 210)
PAIR_CALLOUT_BG_LIGHT = (250, 251, 253, 218)
PAIR_INK_DARKEN = 0.42
# How far off the bond the callout's near edge sits, measured perpendicular to the
# bond: one bead RADIUS to clear the silhouette of whichever bead is bigger, plus
# this fixed gap. It used to be three radii and no gap, which is the same rule with
# the wrong constant in it -- on a scene framed as tightly as the two-bead one a
# bead is ~50 px, so the panel sat 150 px off the pair with an empty band between
# them, at the bottom of the frame reading as a separate instrument rather than as
# this pair's numbers. Clearing the bead is the actual requirement; the gap is then
# the only free parameter, and it does not want to scale with the zoom.
#
# It does NOT clear the force arrows on the driven bead, which are as long as the
# force is: they point along the drive, this scene's drive is along the bond, and
# widening the gap enough to cover a straight-down shove would undo the change for
# every other frame. The plate is translucent and the arrows are drawn over it.
PAIR_CALLOUT_GAP_PX = 16.0
# The bond line between the two beads, and the arrowheads on the per-term force
# glyphs beside each number.
PAIR_LINE_WIDTH = 2
PAIR_GLYPH_MAX_PX = 15.0        # half-length of a full-scale force glyph
PAIR_GLYPH_HEAD_PX = 4.0
# The landmark rings around the fixed partner (ForceField.pair_landmarks): points
# per ring, and how solid they are drawn. Faint on purpose -- they are the scale
# the scene is read against, not the subject of it.
PAIR_SHELL_SAMPLES = 72
PAIR_SHELL_ALPHA = 105
PAIR_SHELL_LABEL_ALPHA = 235

# Membrane bead fill. (The stick backbone linking neighbours in 3D, and the
# simulation-box outline drawn around the scene, are RenderStyle.bond_color and
# .box_color/.box_alpha -- both are faded by depth cueing toward the scene's
# background, so they belong with it.)
MEMBRANE_BEAD_COLOR = (232, 104, 98)
# The corner of the box nearest the eye hangs in front of everything and streaks
# across the scene, so it is faded out. FADE_DEPTH is how much of the box's own
# depth span the fade covers, measured back from its nearest corner: 0 disables
# it, 0.3 dissolves roughly the near third of the three edges meeting that
# corner. SUBDIVISIONS is how many pieces each edge is cut into to carry the
# gradient -- enough that it reads as smooth, few enough to stay 12*n lines.
BOX_EDGE_FADE_DEPTH = 0.30
BOX_EDGE_SUBDIVISIONS = 12
# Paper (MesoMem) bead coloring: each bead is a little sphere whose POLES (along
# its director n_i) are hydrophilic -> blue, and whose EQUATOR (perpendicular to
# n_i) is hydrophobic -> yellow. The yellow band therefore tilts with the
# director, so tilt/splay show directly in the coloring. BAND_HALFWIDTH is the
# band's half-height as |cos(latitude)| (0 = equator, 1 = pole); BAND_SOFT is
# the soft transition width between yellow and blue.
BEAD_POLE_COLOR = (122, 165, 217)      # blue, hydrophilic poles
BEAD_EQUATOR_COLOR = (247, 225, 122)   # yellow, hydrophobic equator
BEAD_BAND_HALFWIDTH = 0.30
BEAD_BAND_SOFT = 0.07
# One pole (the +director pole) is over-painted WHITE, so which way a bead's
# director points is unambiguous (the two normals +n / -n no longer look
# identical). The cap is a small dot right at the pole: white where the SIGNED
# cos-latitude s = N.n exceeds BEAD_WHITE_POLE_MIN, with a BEAD_WHITE_POLE_SOFT
# transition. The projected on-screen diameter of the cap scales as sqrt(1 -
# MIN^2), so MIN = 0.954 (sin = 0.30) makes it half the diameter of the older
# MIN = 0.80 (sin = 0.60) dot. Only s > 0 (the +n hemisphere) is affected, so
# the opposite pole keeps the blue/yellow banding.
BEAD_WHITE_POLE_COLOR = (245, 246, 250)
BEAD_WHITE_POLE_MIN = 0.954
BEAD_WHITE_POLE_SOFT = 0.03

# --- the other bead colouring: potential energy -------------------------------
# Instead of the director banding, paint each bead by ITS OWN potential energy in
# the force field (LAMMPS' per-atom pe). The banding answers "which way is this
# bead pointing"; this answers "how bound is it" -- so a bead pulled out of the
# membrane brightens as its bonds stretch, a well-packed one stays dark, and on
# the assembly box the ordered patches read as dark against the loose gas. The
# +director pole keeps its white cap in both modes, so orientation is never lost.
#
# Matplotlib's `inferno`, sampled at 32 points and interpolated in the shader:
# perceptually uniform (equal steps in the number look like equal steps in the
# colour) and dark-to-bright, which is the right direction for "more energy".
# Baked in rather than imported so matplotlib is not a runtime dependency.
INFERNO = (
    (0.0015, 0.0005, 0.0139), (0.0140, 0.0112, 0.0719), (0.0423, 0.0281, 0.1411), (0.0820, 0.0433, 0.2153),
    (0.1358, 0.0469, 0.2998), (0.1904, 0.0393, 0.3614), (0.2450, 0.0371, 0.4000), (0.2972, 0.0475, 0.4205),
    (0.3540, 0.0669, 0.4309), (0.4039, 0.0856, 0.4332), (0.4537, 0.1038, 0.4305), (0.5035, 0.1216, 0.4234),
    (0.5596, 0.1413, 0.4101), (0.6093, 0.1595, 0.3936), (0.6585, 0.1790, 0.3727), (0.7065, 0.2007, 0.3478),
    (0.7584, 0.2291, 0.3153), (0.8019, 0.2587, 0.2831), (0.8420, 0.2929, 0.2486), (0.8780, 0.3321, 0.2123),
    (0.9130, 0.3816, 0.1698), (0.9387, 0.4301, 0.1304), (0.9591, 0.4820, 0.0895), (0.9742, 0.5368, 0.0484),
    (0.9846, 0.6011, 0.0236), (0.9879, 0.6603, 0.0517), (0.9856, 0.7208, 0.1122), (0.9775, 0.7823, 0.1859),
    (0.9625, 0.8515, 0.2855), (0.9487, 0.9105, 0.3953), (0.9517, 0.9606, 0.5242), (0.9884, 0.9984, 0.6449),
)

# --- the third bead colouring: which aggregate ---------------------------------
# Neither of the two above says what a bead is PART OF. The director banding is a
# property of one bead and the energy ramp is a property of one bead; a membrane
# and the gas around it are painted from the same two pictures. This one paints
# each connected aggregate in its own colour, so the coarsening on the assembly
# box reads directly: the number of colours falls, the patches grow, and a merge
# is two colours becoming one. See playground/clustering.py for how a cluster is
# found and, harder, how it keeps its colour from frame to frame.
#
# WHY THESE TEN. The requirement is categorical -- ten labels that are only ever
# compared, never ordered -- so it is the opposite problem from INFERNO above,
# where the whole point is that the colours line up on a scale. What a categorical
# set has to do is be equally strong: no member may look more important, closer or
# more "selected" than another, because a cluster is not more of an aggregate for
# being painted red.
#
# So they are ten hues spaced evenly (36 degrees apart) around the OkLCh wheel at
# one lightness and one chroma, which is the closest thing to a guarantee of that
# -- OkLCh being the space in which equal steps look equal. Two hand corrections
# on top, both the usual ones: the yellow-green arc is raised in lightness and the
# blue-violet arc lowered, because a yellow at the blues' lightness looks dirty
# and a blue at the yellows' looks chalky; and each hue's chroma is pulled back to
# 93% of what sRGB can actually hold there, so the saturated ones (magenta, green)
# are not silently clipped into a different hue than the ones next to them.
#
# The band they all sit in -- L 0.62 to 0.78 -- is chosen for what happens AFTER:
# these are albedos of lit spheres, multiplied by sun and sky and pushed through
# half an ACES curve. Darker and the ambient occlusion between packed beads takes
# them to mud; lighter and the specular highlight blows the hue out of them. It is
# also the band that survives the light-mode flip (render_style.LIGHT_MODE), where
# the background rises to near-white and a pale palette would have nothing to be
# darker than.
#
# STORED IN THE ORDER THEY ARE HANDED OUT, which is not hue order: every third hue
# (a stride of 3 over 10 is a cycle through all of them), so the first four
# clusters on screen are coral, green, blue, pink rather than four neighbouring
# reds. Adjacent hues are the pair most easily confused, and a scene with two
# clusters should never be showing them.
CLUSTER_COLORS = (
    (240, 115, 108),   # coral
    (132, 192,  83),   # green
    ( 36, 157, 227),   # blue
    (226, 113, 172),   # pink
    (209, 181,  42),   # gold
    ( 41, 177, 191),   # cyan
    (178, 113, 211),   # violet
    (238, 143,  45),   # orange
    ( 42, 185, 145),   # jade
    (114, 123, 227),   # indigo
)
# Beads in no cluster worth naming -- the monomer gas, and anything under
# clustering.MIN_CLUSTER_SIZE -- are NOT here: they are RenderStyle's
# `cluster_gas_color`, because unlike the ten above they have to recede toward
# whichever background the scene is drawn on. A desaturated slate rather than one
# more colour, either way: the gas is the ground of this picture, and it has to
# read as "not one of these" at a glance.
# Seconds for a bead to cross most of the way to a new colour. The slot changes in
# one step (it is an integer); this is what makes that step a dissolve instead of
# a pop, and it is the difference between a merge reading as two aggregates
# becoming one and reading as a glitch. Long enough to be legible as a transition,
# short enough that the picture is never lying about the current clustering for
# more than a moment.
CLUSTER_FADE_SECONDS = 0.45

# Play / Pause / Reset playback buttons (self-assembly system), drawn along the
# bottom of the sim view. The button matching the current state (e.g. Play while
# running) is highlighted with the active colors.
BUTTON_BG = (40, 42, 52)
BUTTON_TEXT = (215, 218, 228)
BUTTON_BORDER = (90, 94, 108)
BUTTON_ACTIVE_BG = (70, 150, 220)
BUTTON_ACTIVE_TEXT = (12, 14, 20)

PANEL_WIDTH = 480
PANEL_PAD = 14
PLOT_COLORS = {
    "temp": (255, 150, 90),
    "press": (120, 180, 255),
    "ke": (120, 220, 140),
    "pe": (230, 130, 230),
    "etotal": (240, 220, 100),
    "rdf": (100, 210, 255),
}


# ---- the taught sequence: the lesson card, the position rail, the thesis button

# THE LESSON CARD, top-left of the sim view (Renderer._draw_lesson_card): the
# stage number and title, the one-line claim, and the one-line instruction. Its
# COLOURS come from the scene's own RenderStyle (text_color / dim_text_color, like
# the header line above it), because every membrane scene is drawn on a light
# background and a fixed dark-theme grey would be unreadable on it. Only the sizes
# and the spacing are here.
#
# The title is set at 30 against the header font's 22 for one reason: this is read
# off a projector from the back of a room, in the two seconds before the presenter
# starts talking over it, and it is the only line that has to survive that. The
# claim follows at 20 -- large enough to be read, small enough that the title still
# wins -- and the instruction sits at the body size, dim, because it is for the
# person with their hands on the controls rather than for the room.
LESSON_TITLE_SIZE = 30
LESSON_CLAIM_SIZE = 20
# Baseline gaps down the card, in px at UI scale 1.
LESSON_TITLE_GAP = 6
LESSON_CLAIM_GAP = 4
LESSON_BLOCK_GAP = 8

# THE POSITION RAIL, top-right (Renderer._draw_chapter_rail): one mark per scene
# in the offered sequence, grouped into the three acts with a wider gap between
# groups.
#
# NOT A PROGRESS BAR, deliberately. A bar implies a task being completed, which
# frames every scene before the last as overhead on the way to the point -- the
# opposite of true here, where the middle IS the physics. A row of marks says the
# two things that are actually useful (where am I, and how much is left) and the
# grouping adds the one a bare "4 of 8" cannot: which third of the argument this
# is, and therefore what kind of thing is coming next.
#
# Small and dim, because the number itself is already large in the lesson card and
# an indicator that competes with the scene has misunderstood its job.
RAIL_MARK_W = 13
RAIL_MARK_H = 5
RAIL_MARK_GAP = 4
RAIL_ACT_GAP = 13
# The current scene's mark: taller (it grows downward from the same top edge, so
# the row's top stays a straight line) and drawn in the scene's own text colour
# against the others' dim one.
RAIL_CURRENT_H = 9
RAIL_LABEL_GAP = 7

# THE HERO-KNOB ROW, centred at the bottom of the sim view just above the playback
# controls (Renderer.draw_hero_knobs). One button per HeroKnob the scene declares,
# each carrying the input-device button number that fires it.
#
# LOUD, BECAUSE IT IS THE THING TO DO NEXT. These are not more panel widgets: a
# hero knob is the move whoever built the scene wants made on it, and a presenter
# who has to hunt for it will not make it. So the released state is already an
# accent button rather than another grey chip in the playback family, and the
# engaged state goes full amber -- the one alarm colour the rest of this UI keeps
# for a live cluster allocation, and which nothing in a membrane scene is.
HERO_W = 250
HERO_H = 42
HERO_GAP = 14                   # between two knobs
HERO_ROW_GAP = 14               # above the playback row
# Released: a filled accent, dark enough for its own light text, with a brighter
# rim. Not the playback buttons' grey, because it is not a playback control.
HERO_BG = (46, 58, 78)
HERO_TEXT = (226, 234, 246)
HERO_BORDER = (120, 160, 210)
# Engaged.
HERO_ENGAGED_BG = (208, 138, 30)
HERO_ENGAGED_TEXT = (16, 12, 4)
HERO_ENGAGED_BORDER = (245, 190, 90)
# The little device-button number on the left of each knob: "5" in a chip, so the
# mapping is on the button rather than in a manual.
HERO_BADGE_W = 26
HERO_BADGE_BG = (16, 22, 34, 190)
HERO_BADGE_TEXT = (170, 200, 235)
HERO_BADGE_ENGAGED_BG = (120, 76, 8, 210)
HERO_BADGE_ENGAGED_TEXT = (255, 226, 160)
# The caption drawn above the row while any knob is engaged, saying in numbers what
# that knob changed. Amber too, and on a plate, because it has to be legible over
# whatever the membrane is doing behind it.
HERO_CAPTION_COLOR = (250, 205, 110)
HERO_CAPTION_BG = (28, 20, 6, 216)
HERO_CAPTION_GAP = 10

# THE HOOK LINE in the panel -- the question this scene leaves open. Its own
# colour, and deliberately NOT the focus cyan: cyan already means "this is what
# the joystick is driving" everywhere else in this UI, and a line of prose in it
# would read as a control. A soft blue instead: clearly a note, clearly not a
# widget, and legible against the dark panel without competing with the header.
HOOK_COLOR = (140, 185, 235)

# HOW MUCH ROOM A SLIDER ROW NEEDS ABOVE ITS TRACK. `Slider.draw` blits its
# "label: value" line at `rect.y - 18` and `Slider.focus_rect` starts the row at
# `rect.y - 22`, so a track placed at the current y puts its own label 22 px into
# whatever was drawn last. Everything that hands a y to a slider has to leave this
# much headroom first, and the two places that did not were both visibly wrong:
# the temperature label sat on the divider above it, and the first slider of the
# expanded "Advanced" group sat on the word "Advanced".
SLIDER_LABEL_H = 22

# AND HOW TALL THE WHOLE ROW IS. A slider with an `optimum` (or the temperature's
# melt mark) draws a caption BELOW its track as well, and at the plain pitch that
# caption lands on the next row's label -- zeta's "opt" was sitting inside "splay
# symmetry (0=signed, 1=|dot|)". The temperature slider already used the taller
# pitch for exactly this reason; it is a property of the slider, not of which slot
# it happens to be in, so both loops now ask.
SLIDER_ROW_H = 34
SLIDER_ROW_H_MARKED = 46
