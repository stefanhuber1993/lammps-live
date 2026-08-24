"""MesoMem membrane sheet -- the paper's planar-stability test, made interactive.

The same force field as the patch, at the scale the paper (Sec. IV) uses to check
that the interaction scheme supports a stable planar membrane: beads on a
hexagonal lattice at spacing a = 0.8 sigma, periodic in-plane, relaxed under
Langevin dynamics with a barostat that drives the lateral pressure to zero so the
sheet equilibrates tension-free. The paper's benchmark is 50x50 sites; 30x30 still
limits finite-size effects while staying real-time under interactive control.

Differences from the patch: the periodic cell means the sheet holds itself flat
with no artificial tether, and the puller is whichever bead starts nearest the box
centre.

THE BAROSTAT KEEPS RUNNING while you play, and it is not a detail of the setup --
it is what makes the planar membrane physical. Freeze the cell after the settle
and the sheet is stuck at a fixed projected area, which is only ever the right one
for the temperature it was settled at; the dial here runs to 0.5, and the
tension-free cell is 27% larger in area at T = 0.2 than at T = 0.001. A membrane
held below its relaxed area is laterally COMPRESSED, its long-wavelength
undulation modes have negative stiffness and grow, and what you end up watching is
a static ripple that never decays -- a reported artefact of exactly this
playground. See HexSheet in playground/scenario.py for the measurements and for
why this uses `press/berendsen` where the collaborator's reference deck uses
`fix nph/sphere`. Above `melt_temp` the sheet is no longer a membrane and the cell
inflates without bound; that is honest, not a fault.

Units are the paper's LJ-reduced units (sigma = eps = m = 1).
"""
from ..playground import Control, Playground, hex_sheet
from ..render_style import DEFAULT_STYLE

# A flat lattice seen at a tilt is the opposite shape of problem from the
# assembly box: almost all of its depth is the sheet RECEDING from the camera,
# not a solid volume, so the two depth effects want retuning rather than
# switching off -- pointed further into the distance, and gentler, because here
# they are describing one continuous surface rather than separating a crowd.
#
#   DEPTH OF FIELD  focus a quarter of the way in (the near half of the sheet,
#     including the pulled bead, stays sharp) with a slightly smaller bokeh, so
#     the far edge softens into a tilt-shift falloff instead of dissolving.
#   DEPTH CUE  carried further back (0.70) and lightened, so the far rows still
#     read as membrane rather than being swallowed before the box outline is.
#
# The strong outline is doing real work here: at 900 overlapping beads it is
# what keeps the individual particles legible in the middle distance.
# (`.on_light()` on the end draws the same sheet on a light background.)
STYLE = DEFAULT_STYLE.varied(
    # The cell is periodic in-plane, i.e. this sheet is a window onto an endless
    # membrane -- so draw the neighbouring windows too and fade them out with
    # distance. Asymmetric on purpose: half a cell in FRONT of the real one, so
    # the membrane runs off the bottom of the frame, and three behind it running
    # to the horizon. The real cell -- the one carrying the controlled bead, its
    # control net and the outline -- is the one you are looking into.
    # `periodic_images=(0, 0, 0)` goes back to the single cell; the fade bounds
    # are fractions of how far the copies reach, so 1.0 finishes exactly where
    # they are cut and 0.0 starts at the real cell's own edge.
    periodic_images=(2, (0.5, 3), 0),
    image_fade_start=0.5,
    image_fade_end=1.0,
    box_alpha=200,
    net_alpha=150,
    dof_focus=0.25,         # the near quarter, incl. the pulled bead, stays sharp
    dof_range=1.5,          # half the span to go fully out of focus
    dof_bokeh_px=6.0,       # (8.0 by default) -- 0 switches DoF off entirely
    cue_end=0.90,           # fade completes further back than the default 0.55
    cue_strength=0.6,       # and stops short of the full 0.75
    # Left at the defaults, listed because they are the next dials to reach for:
    ao_strength=9.83,           # contact darkening between the packed beads
    outline_strength=12.0,      # how black the line around each bead is
    outline_edge_fraction=0.90,  # how far out it starts (0.94 = hairline)
).on_light()

PLAYGROUND = Playground(
    name="MesoMem membrane sheet (3D)",
    description="Paper-scale hexagonal MesoMem sheet, held tension-free by a "
                "running barostat: pull one bead out and watch tilt/splay "
                "propagate.",
    force_field="mesomem",
    scenario=hex_sheet(n_cols=30, n_rows=30, a=0.8, z_half=4.0,
                       settle_steps=1000,
                       # The camera frames the REAL cell -- the images tile away
                       # behind it, so nothing has to be pulled back for them --
                       # but aims a little past its centre, which drops its near
                       # edge to the bottom of the frame and puts the receding
                       # copies where the eye expects the distance to be.
                       view_span=1.0, view_aim_ahead=0.55),
    mode="game",
    control=Control(
        atom="nearest_center",
        plane="xz",
        leash=(5.0, 3.5),
        # A large cohesive membrane barely tented under the patch's forces, so
        # pull authority is raised here.
        max_input_force=7.0,
        grid_step=0.8,
    ),
    observables=["nematic_S", "thickness", "area_per_particle"],
    # A large membrane buckles under high splay rather than merely stiffening, so
    # the dial is worth taking far past the patch's useful span of 3.
    param_ranges={"k_splay": (0.0, 40.0)},
    presets={
        "paper": {},
        # k_splay reaches much further on the sheet than on the 7-bead patch --
        # high splay buckles a large membrane rather than just stiffening it,
        # which is the interesting failure mode to be able to reach.
        "buckled": {"k_splay": 30.0},
        "floppy": {"k_tilt": 2.0, "k_splay": 0.1},
        "isotropic_only": {"k_tilt": 0.0, "k_splay": 0.0, "wc": 0.0},
    },
    temperature=(0.0, 0.5),
    temperature_default=0.001,
    melt_temp=0.3,
    particle_radius=0.5,
    reduced_units=True,
    # A warm sheet ripples thermally at every wavelength at once; smoothing the
    # drawn beads leaves the long-wavelength undulations and the healing of a hole
    # visible without the shimmer on top. Advanced slider, 0 (off) by default.
    trajectory_smoothing=True,
    render_style=STYLE,
)
