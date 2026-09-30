"""Tunable look of the 3D bead scenes -- one dataclass, per system.

These are the knobs of the deferred impostor renderer in `ui/gl3d.py`. The
defaults are the ones tuned in the standalone showreel (21_LearnModernGL/
showreel.py) on a dense cube of spheres, so the MesoMem self-assembly box -- the
scene that matches that one -- takes them unchanged. The patch (7 beads) and the
sheet (a flat, strongly receding lattice) are different pictures and override a
few fields; see each playground file. `LIGHT_MODE` / `STYLE.on_light()` at the
bottom draws any of them on a light background instead.

Everything here is a pure number: nothing in this module imports GL, pygame or
LAMMPS, so a playground can carry a RenderStyle without dragging the renderer in.

Two conventions worth knowing before you turn a dial:

  * COLOURS ARE LINEAR-LIGHT once inside the shader. Colours declared as 0..255
    bytes -- the bead bands in `ui/theme.py`, the scene palette at the top of
    the dataclass below -- are display-space, and the renderer raises them to
    `display_gamma` on the way in and encodes the final image back down by
    1/`display_gamma` at the end. So a "brighter" bead means a brighter *byte*,
    while the lighting values here (`sky_ambient`, `sun_color`, ...) are already
    linear and do not go through that conversion.
  * SIZES THAT SCALE WITH THE BEADS end in `_r` and are multiples of the bead
    RADIUS (spec.atom_radius_A). That is what keeps the ambient-occlusion reach
    and the contact shadows meaning the same thing on a 0.5-sigma bead as they
    would on a 5-Angstrom atom. Depth-based fractions (`cue_*`, `dof_*`) are
    fractions of the scene's own front-to-back depth span, for the same reason:
    they must not have to be re-tuned when the camera dollies in.
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class RenderStyle:
    """One system's rendering look. `dataclasses.replace(STYLE, ...)` to vary it."""

    # --- the scene's palette: dark or light -----------------------------------
    # These are DISPLAY-SPACE 0..255 bytes, like the colours in ui/theme.py --
    # they name things you would pick in a colour picker, not linear light. They
    # live here rather than in theme.py because they are per-system: a scene is
    # drawn on ITS OWN background, and a style handed `.on_light()` (see
    # LIGHT_MODE below) flips all of them together with the lighting that has to
    # agree with them.
    #
    # BACKGROUND is the void the scene sits in, and it is also what everything
    # recedes INTO: the depth cue, the periodic-image fade and the screen-edge
    # vignette all fade toward it, and the bond spokes, the control net and the
    # box outline are lerped toward it by the same ramp. Change it and the fog
    # follows -- there is no separate haze colour to keep in sync.
    background: tuple = (18, 18, 24)
    # A gentle vertical ramp across the void, as a signed fraction of the
    # background's brightness at the BOTTOM of the frame (the top is left alone).
    # Positive lifts the bottom, which on a dark scene reads as a soft glow the
    # beads sit in; negative sinks it, which is what a light scene wants -- a
    # lifted bottom there just clips to white.
    background_gradient: float = 0.35
    # The simulation-box wireframe: a subtle frame around the cell, not a hard
    # cage, hence the alpha. It has to hold against the background it is drawn
    # on -- near-white on the void, near-black on a light one.
    box_color: tuple = (240, 240, 245)
    box_alpha: int = 150               # out of 255
    # The faint spokes between bonded beads (alpha comes from how close the pair
    # is to equilibrium; see theme.BOND_PEAK_ALPHA).
    bond_color: tuple = (150, 96, 92)
    # The control "net": the plane the joystick slides the active bead along.
    net_color: tuple = (120, 150, 210)
    net_alpha: int = 100
    # WHEN TO RING THE CONTROLLED BEAD. The ring answers "which of these is the one
    # I am driving?", which is a real question in a membrane and no question at all
    # in a scene that holds one bead -- there it is a cyan circle drawn around the
    # only thing on screen, competing with the director spike and the arrows for the
    # eye. Three settings:
    #
    #   "always"    (the default) ringed whether it is being steered or not, dim
    #               once released -- see renderer._puller_ring_color.
    #   "released"  ringed ONLY while nobody is holding it. The ring then means one
    #               thing instead of two: not "this is the one", which is obvious,
    #               but "you have let go of it", which is not.
    #   "never"     never ringed.
    puller_ring: str = "always"
    # The two unbacked text lines drawn over the scene itself (the header and its
    # legend). Everything else in the sim view carries its own backdrop, so only
    # these have to follow the background.
    text_color: tuple = (200, 200, 200)
    dim_text_color: tuple = (140, 140, 140)
    # The bead colour of loose, unaggregated material under the CLUSTER colouring
    # (theme.CLUSTER_COLORS holds the ten aggregate colours, which are the same on
    # any background because they are what the eye is meant to land on). This one
    # is here, with the scene, because its whole job is the opposite: the gas is
    # the ground of that picture and has to recede toward the background rather
    # than compete with it -- which means a slate on the void and a much lighter
    # one on a bright ground, where this colour at its dark value reads as the
    # heaviest thing in the frame and buries the aggregates it is meant to set off.
    cluster_gas_color: tuple = (109, 118, 127)

    # --- material -------------------------------------------------------------
    # Gamma between the display-space colours in theme.py and the linear light
    # the shading maths runs in. 1.0 disables the conversion entirely (the flat,
    # washed-out look this renderer had before).
    display_gamma: float = 2.2
    # Sun direction in WORLD space (the beads' membrane normal is +z), pointing
    # FROM the surface TOWARD the light: right, toward the default viewer, above.
    # World-fixed rather than camera-fixed, so an orbiting camera moves through a
    # lit scene instead of dragging the lighting around with it.
    sun_dir: tuple = (0.45, -0.5, 0.75)
    sun_color: tuple = (1.0, 0.96, 0.88)
    sun_gain: float = 0.95
    # Ambient fill, standing in for skylight. This is the colour a bead shows
    # where the sun does not reach, and the only term ambient occlusion darkens.
    sky_ambient: tuple = (0.30, 0.38, 0.52)
    # Specular: the exponent is roughness (32 reads as plastic, 96 as a polished
    # liquid), the gain is how bright the highlight is. The fresnel term makes
    # every surface mirror-like at glancing angles, which is most of what makes
    # a bead read as wet rather than chalky.
    spec_power: float = 32.0
    spec_gain: float = 0.20
    fresnel_color: tuple = (0.55, 0.70, 0.95)
    fresnel_gain: float = 0.75

    # --- ambient occlusion ----------------------------------------------------
    # Screen-space AO: crevices where beads pack together darken. Cost is per
    # PIXEL and independent of the bead count, so 1500 beads cost what 7 do.
    ao_samples: int = 16               # 0 disables the pass entirely
    ao_radius_r: float = 2.0          # x bead radius -- how far a crevice reaches
    ao_bias_r: float = 0.08           # x bead radius -- clears a sphere's own curvature
    # Exponent on the 0..1 visibility term. A power, not a multiply: it pins
    # fully-open surfaces at full brightness and bends only the occluded ones
    # down, so contact shadows deepen without flattening the whole image.
    ao_strength: float = 5.83
    # Curvature AO: a sphere's own rim occludes it, which is free to compute
    # from the view normal and does most of the work of separating beads.
    curvature_ao: bool = True

    # --- screen-space contact shadows ----------------------------------------
    # Marched through the depth buffer alongside the ambient occlusion, at half
    # resolution, and smoothed by the same blur -- which is what keeps the
    # march's jittered, all-or-nothing hits from reading as noise.
    shadows: bool = True
    shadow_len_r: float = 5.5         # x bead radius -- how far a bead casts
    shadow_bias_r: float = 0.08       # x bead radius -- self-shadow acne guard
    shadow_thick_r: float = 2.6       # x bead radius -- assumed occluder thickness

    # --- outline --------------------------------------------------------------
    # Depth-discontinuity edge detect: where the depth changes fast between
    # neighbouring pixels, draw a dark line. STRENGTH is how black it goes.
    #
    # The other knob is where a BEAD'S OWN CURVATURE starts counting as an edge,
    # given as the fraction of its radius at which the outline begins -- 0.90
    # rings the outer tenth. It is quoted that way, rather than as the raw
    # depth-gradient threshold the shader wants, because the raw number means
    # completely different things at different bead sizes on screen: a threshold
    # that draws a crisp 1px ring on a distant bead swallows a third of a near
    # one (the ring is an axis-aligned band, so what is left of a swallowed bead
    # is a bright SQUARE -- unmistakable once seen). Beads here are drawn several
    # times larger on screen than in the showreel this look came from, and get
    # larger again in fullscreen, so the renderer converts this fraction into the
    # threshold using the camera's focal length and the picture stays the same.
    outline: bool = True
    outline_strength: float = 12.0
    outline_edge_fraction: float = 0.90
    outline_color: tuple = (0.01, 0.015, 0.02)

    # --- depth cue ------------------------------------------------------------
    # A linear fade to the background across the scene's own depth span -- a
    # legibility aid from technical illustration, not a physical fog. Start and
    # end are fractions of the span measured from the nearest bead; strength is
    # how far it fades (1.0 = fully background at `cue_end`).
    depth_cue: bool = True
    cue_start: float = 0.30
    cue_end: float = 0.55
    cue_strength: float = 0.75

    # --- depth of field -------------------------------------------------------
    # Circle-of-confusion blur. `dof_focus` places the sharp plane as a fraction
    # of the scene depth span (0 = nearest bead, 1 = farthest); `dof_range` is
    # how much of that span it takes to go from sharp to maximally blurred;
    # `dof_bokeh_px` is that maximum blur radius in pixels, at a 900px-tall
    # viewport (it scales with the viewport so fullscreen looks the same).
    dof: bool = True
    dof_focus: float = 0.15
    dof_range: float = 0.40
    dof_bokeh_px: float = 8.0

    # --- tonemap --------------------------------------------------------------
    # Half-strength ACES: the full filmic curve washes a near-monochrome scene
    # toward grey, so it is blended half-and-half with the linear colour, which
    # keeps the highlight rolloff at half slope and gives the colours back their
    # saturation. Then the gamma encode, then a corner vignette.
    tonemap: bool = True
    tonemap_exposure: float = 1.1
    tonemap_mix: float = 0.25
    vignette: float = 1.2

    # --- the energy colouring -------------------------------------------------
    # Range the inferno ramp spans when the beads are coloured by their own
    # potential energy instead of their director banding (see theme.INFERNO).
    #
    # The number being coloured is the WHOLE energy of the bonds touching a bead
    # -- the same quantity the additive-energy panel reports for the controlled
    # one, and twice LAMMPS' per-atom share (see PlaygroundSystem.get_bead_energies).
    # Measured spans, for retuning: the relaxed sheet sits in a narrow band near
    # -7.5, the patch runs -5.6 to -3.0, and the assembly box -6.8 to +0.3 as its
    # aggregates form. -8 to 0 covers all three on one scale, so a bead reads as
    # "more bound than that one over there" across the whole demo rather than
    # only within its own scene; narrow it per system to spend the whole ramp on
    # one of them.
    energy_range: tuple = (-8.0, 0.0)
    # A SECOND SCALE for the tinted species (the beads a playground paints with
    # `render_tints`, mix > 0.5), or None to put every bead on `energy_range`.
    # For a species whose number is a different quantity: the vesicle chain's
    # beads carry their BENDING energy (see ForceField.bonded_bead_energies), which
    # is 0 for a straight chain and has nothing to do with how bound a membrane
    # bead is. Painted on theme.BEND_RAMP, a different hue family, so nobody reads
    # one against the other.
    tint_energy_range: tuple = None

    # --- periodic image tiling ------------------------------------------------
    # A periodic cell is a window onto an infinite system, and drawing only the
    # one cell says otherwise -- the sheet looks like a small square raft rather
    # than a piece of an endless membrane. Draw its periodic IMAGES too and the
    # illusion is right, at the cost of one instanced copy per image (the whole
    # pipeline after the G-buffer is screen-space, so nothing else gets more
    # expensive).
    #
    # `periodic_images` is how many copies to draw per axis. Each entry is
    # either a number -- the same either side -- or a (before, after) pair, and
    # either may be FRACTIONAL:
    #
    #     (0, 0, 0)            just the real cell
    #     (1, 1, 0)            3x3 in x and y
    #     (0.5, 0.5, 0)        half of each neighbour: a fringe of surrounding
    #                          material, at a quarter of the instances of a 3x3
    #     (1, (0.5, 3), 0)     one either side in x; half a cell in front and
    #                          three behind in y -- which puts the real cell near
    #                          the FRONT, with its copies receding away behind
    #
    # That last shape is the useful one for a scene you look into rather than at:
    # the cell carrying the controlled particle, its control net and the outline
    # is the one you are close to, and everything behind it is context. Counts
    # only mean anything on axes the cell is actually periodic in, and the
    # renderer clamps them to that -- asking for images along a free axis would
    # be inventing matter.
    #
    # The copies fade to the background so the tiling has no visible outer edge.
    # Both bounds are fractions of HOW FAR THE COPIES REACH in that direction, so
    # 1.0 means "gone exactly where they are cut" -- whatever the counts are, and
    # separately for each side, which is what keeps an asymmetric block from
    # showing a straight edge on its short side. 0.0 is the real cell's own
    # boundary.
    periodic_images: tuple = (0, 0, 0)
    image_fade_start: float = 0.0
    image_fade_end: float = 1.0

    # --- section (cut-away) ---------------------------------------------------
    # Draw only the particles on the far side of a plane, so the scene reads as a
    # CROSS-SECTION rather than a surface. `section_axis` is a world axis and
    # `section_min` a coordinate along it: everything below that coordinate is not
    # drawn. None -> draw everything, which is every other scene.
    #
    # This exists because an edge-on view of a large flat membrane cannot show
    # what is happening INSIDE it. A monolayer is opaque, and a rod being
    # invaginated sits at the same height as the rows of beads between it and the
    # camera -- so those rows hide the very thing being looked at, and no camera
    # angle fixes it: raise the camera and the depth you are trying to measure
    # foreshortens away, lower it and the near rows are exactly in the way. Cut
    # the near half off and the remaining face IS the section: the membrane's
    # profile as a line, and the rod sitting in it.
    #
    # Particles drawn as BODIES (see MDSystem3D.get_glyph_spheres) are exempt --
    # the rod is the subject, and half a rod is a rendering artifact rather than a
    # more informative picture.
    #
    # This is the PLAYGROUND's cut: fixed, in a fixed world axis, declared because
    # the scene is always drawn this way. The viewer has one of their own -- a slab
    # they slide through the box with the joystick's thrust lever (see
    # lammps_live/view_slice.py) -- and the two compose by intersection rather than
    # overriding each other, since a sliced view of a scene drawn in section is a
    # section of the slab and that is what both asked for.
    section_axis: tuple = (0.0, 1.0, 0.0)
    section_min: float = None

    # --- the material a BODY is drawn in --------------------------------------
    # A particle drawn as a body (see MDSystem3D.get_glyph_spheres -- the rod) is
    # not a membrane bead, and the bead colourings are all wrong for it:
    #
    #   THE DIRECTOR BANDING is a statement about a bead's orientational degree of
    #     freedom. The rod has an axis, but no yellow hydrophobic equator and no
    #     blue poles -- painting it in the membrane's own colours says it is made
    #     of the same stuff as the thing it is invaginating into.
    #   THE ENERGY RAMP is worse, and it is a scale problem rather than a taste
    #     one. The ramp spans a bead's `energy_range` (single-digit eps); the rod
    #     is one particle in contact with a hundred beads at once, so its own
    #     potential runs to several hundred. It is pinned at the bright end of the
    #     ramp in every configuration, which is not a reading of anything -- and it
    #     drags the eye to the one object in the frame whose colour means nothing.
    #   THE CLUSTER COLOURING gives it whichever aggregate colour its owner drew.
    #
    # So a body gets a MATERIAL of its own instead, outside the colourings
    # entirely: `"bacterium"` is a mottled, procedurally textured surface (see the
    # `bacterium` branch of gl3d's geometry shader) in the two colours below. It
    # reads as a living thing with a wall rather than a giant bead, which is what
    # the rod is a picture of, and it stays the same object whichever colouring
    # the viewer has the beads in. "" leaves bodies painted like their owner.
    #
    # Only the GL renderer has this; the CPU fallback bakes one banded sprite per
    # radius and cannot vary a per-bead albedo at all -- the same reason it draws
    # no energy ramp and no per-bead tints.
    body_material: str = ""
    # The two ends of the mottling, DISPLAY-SPACE bytes like the rest of the
    # palette. A pale warm cell wall against an olive-brown shadow: enough
    # separation that the texture reads at a distance, not so much that the rod
    # looks like camouflage. Kept off `on_light()`: a bacterium is an object with
    # its own colour, not part of the scene's ground, and it has to stay the same
    # thing on either background.
    body_color_light: tuple = (214, 208, 168)
    body_color_dark: tuple = (108, 116, 72)
    # Mottle size, in units of the bead radius: the wavelength of the coarsest
    # octave of the noise. About two bead diameters, so a 1.5-sigma rod carries a
    # handful of patches along its body rather than one gradient or a fine speckle
    # that aliases the moment the camera pulls back.
    body_mottle_r: float = 4.0

    # --- antialiasing ---------------------------------------------------------
    # FXAA over the finished image. Every silhouette here is a hard `discard`
    # edge and every outline a saturating step, so without this they stair-step
    # -- worst on the sheet, where hundreds of small beads each contribute
    # several arcs of edge. Off gives a crisper but visibly jagged picture.
    antialias: bool = True

    def outline_threshold(self, focal_px):
        """The shader's depth-gradient threshold, from `outline_edge_fraction`.

        A bead of world radius r at distance D covers D/f world units per pixel,
        and its surface z = sqrt(r^2 - s^2) has slope s/sqrt(r^2 - s^2) at
        distance s off its centre -- so the central difference the shader
        measures is 2*(D/f)*u/sqrt(1-u^2) at u = s/r, while its threshold is
        `thresh * D`. The D and the r both cancel: one number holds for every
        bead in the scene, at every depth, and only the focal length (i.e. the
        on-screen size) is left."""
        u = min(max(self.outline_edge_fraction, 0.0), 0.999)
        return 2.0 * u / ((1.0 - u * u) ** 0.5) / max(focal_px, 1.0)

    def cue_range(self, d_front, d_back):
        """(near, far) view distances of the depth-cue ramp for a scene spanning
        d_front..d_back. The renderer fades the bond spokes, the control net and
        the box outline over this same range as it builds them, so the drawing on
        top of the scene recedes on exactly the ramp the beads do."""
        span = max(d_back - d_front, 1e-4)
        return d_front + self.cue_start * span, d_front + self.cue_end * span

    def focus_range(self, d_front, d_back):
        """(focus distance, CoC range) in world units for a scene spanning
        d_front..d_back."""
        span = max(d_back - d_front, 1e-4)
        return d_front + self.dof_focus * span, max(self.dof_range * span, 1e-4)

    def varied(self, **overrides):
        """A copy with some fields replaced -- what a playground calls."""
        return replace(self, **overrides)

    def on_light(self):
        """This style, redrawn on a light background (see LIGHT_MODE)."""
        return replace(self, **LIGHT_MODE)

    def on_dark(self):
        """This style, back on the default dark background."""
        return replace(self, **DARK_MODE)


DEFAULT_STYLE = RenderStyle()

# ---- light mode ---------------------------------------------------------------
# The same scenes on a bright, paper-like ground: `STYLE.on_light()`, or start
# from `LIGHT_STYLE.varied(...)` instead of `DEFAULT_STYLE.varied(...)`. Nothing
# about the SHAPE of the picture changes -- the depth-of-field, the cue ramp, the
# periodic tiling and the outline geometry are all background-agnostic -- so a
# per-system style keeps its own tuning across the flip.
#
# What does change is everything that assumed a void, and why:
#
#   BACKGROUND      a light neutral grey rather than white. White would leave the
#                   beads' white polar caps and their specular highlights with
#                   nothing to be brighter than.
#   GRADIENT        sunk at the bottom instead of lifted: on a light ground a
#                   lifted bottom runs straight into the top of the tonemap and
#                   flattens into a white band.
#   AMBIENT FILL    a bright surround bounces light back into the scene, so the
#                   sky term rises by half again and loses most of its blue -- a white
#                   room, not a night sky -- and the sun comes down to keep the
#                   lit faces off the top of the tonemap. How far to push that
#                   trade is the one judgement call in here: AO and the contact
#                   shadows only ever darken the AMBIENT term, so a scene lit
#                   mostly by fill has nothing left to carve the crevices with
#                   and the beads flatten into pale discs. Fill at 0.50 against
#                   a sun at 0.82 was the point where the packed sheet still
#                   read as rows of separate spheres.
#   FRESNEL         the rim of a bead mirrors what is around it, which is now
#                   bright rather than dark blue. Kept weaker, because a strong
#                   near-white rim on a near-white ground erases the silhouette.
#   OUTLINE         unchanged and doing more work than before: on a light ground
#                   the dark ink line IS the silhouette, since the beads can no
#                   longer separate themselves from the background by being
#                   brighter than it.
#   VIGNETTE        nearly off. Darkened corners on a light page read as a
#                   photocopy artefact rather than as a lens.
#   BOX / NET /     inverted from near-white to a dark slate, and the bond spokes
#   BONDS / TEXT    deepened, so all the line work stays legible against the
#                   ground it is fogged toward.
#   CLUSTER GAS     lightened, and it is the one colour here that moves the OTHER
#                   way from the line work. It is not drawing, it is the quiet
#                   majority of the beads under the cluster colouring, and it has
#                   to sit just off the background so the ten aggregate colours
#                   are what the eye lands on. Dark slate on a pale ground makes
#                   the gas the loudest thing in the picture.
LIGHT_MODE = dict(
    background=(232, 234, 239),
    background_gradient=-0.12,
    sky_ambient=(0.50, 0.53, 0.60),
    sun_color=(1.0, 0.97, 0.91),
    sun_gain=0.82,
    fresnel_color=(0.86, 0.89, 0.96),
    fresnel_gain=0.30,
    tonemap_exposure=1.05,
    vignette=0.18,
    box_color=(58, 62, 78),
    box_alpha=190,
    bond_color=(126, 74, 70),
    net_color=(58, 92, 165),
    net_alpha=120,
    text_color=(40, 44, 54),
    dim_text_color=(105, 110, 122),
    cluster_gas_color=(180, 187, 196),
)

# The same keys at their defaults, so `on_dark()` is an exact undo of
# `on_light()` and cannot drift away from it.
DARK_MODE = {k: getattr(DEFAULT_STYLE, k) for k in LIGHT_MODE}

LIGHT_STYLE = DEFAULT_STYLE.on_light()


@dataclass(frozen=True)
class CameraOrbit:
    """Turntable camera for a scene with nothing to steer (the assembly box).

    The camera is spherical about the scene centre: azimuth, elevation, radius.
    The auto-orbit and the mouse drag write those same three numbers, so they
    compose instead of fighting -- taking the mouse stops the animation, and C
    resumes it from wherever the drag left the camera rather than snapping back
    to a canned angle.
    """
    # Whether the turntable is already turning when the system loads. Off means
    # C starts it.
    autostart: bool = True
    speed: float = 0.16            # rad/s of azimuth while auto-orbiting
    drag_sensitivity: float = 0.006  # rad per pixel of mouse drag
    zoom_step: float = 0.12        # multiplicative, per wheel notch
    # Dolly limits, as multiples of the framing distance the scenario chose --
    # which for the assembly box is 3 cell widths (see RandomFill.camera: the
    # distance IS the perspective strength, so it is set long deliberately). These
    # are quoted against that, so 0.15 still reaches about half a cell width from
    # the centre -- i.e. inside the cell, flying between the membranes, which is
    # the close end worth having -- and 1.5 pulls back to four and a half widths,
    # where the whole box is a small object in the middle of the frame. Both were
    # tightened when the framing distance was lengthened, so that the range you
    # can actually reach is the same one as before.
    dist_min: float = 0.15
    dist_max: float = 1.5
    # Stop short of straight overhead: the view basis is built with a cross
    # product against world up, which collapses when you look exactly along it.
    elev_limit_deg: float = 84.0
    # Joystick flying (the viewport-focus stick, see control_focus.py). Rates,
    # not offsets: the camera keeps moving for as long as the stick is held. The
    # response is the same shape as a slider's -- deadzone, a wide slow plateau,
    # then a smooth ramp to full speed over the last of the travel (see
    # control_focus.band_rate) -- so one hand learns one behaviour and the same
    # push means "creep" whether it is aimed at the camera or at k_tilt. The slow
    # plateau is what a demo actually spends its time in: a nudge that eases the
    # view round a few degrees to show the other side of a membrane.
    stick_deadzone: float = 0.20
    stick_slow_end: float = 0.75
    stick_slow_speed: float = 0.25  # rad/s on the plateau (~14 deg/s)
    stick_speed: float = 2.0        # rad/s at the stop (~115 deg/s, a turn in 3 s)
    stick_zoom_slow_speed: float = 0.8   # wheel notches/s on the plateau
    stick_zoom_speed: float = 6.0        # wheel notches/s at full twist
    # Shift-drag panning: world units moved per pixel, per unit of camera
    # distance. Scaled by the distance so a pan covers the same fraction of the
    # picture whether you are dollied in on one membrane or out at the whole cell.
    pan_sensitivity: float = 0.0015
