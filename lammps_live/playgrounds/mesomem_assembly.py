"""MesoMem self-assembly -- the paper's spontaneous-lamellae test, made live.

The paper's first validation experiment: N = 1500 directored particles dropped at
random positions and orientations into a cubic, fully periodic box of side
L = 20 sigma -- a reduced volume fraction phi = N*Vp/Vbox ~ 0.1 with
Vp = (pi/6) sigma^3. Under Langevin dynamics the disordered gas coarsens: small
patches by t ~ 500 tau, coalescing into large planar membranes by t ~ 2000 tau.

k_tilt is the critical control parameter -- below ~10 the aggregates stay compact
and isotropic, above it they flatten into membranes. Dial it live and watch the
morphology change; the `nematic_S` observable is the number that transition shows
up in, so you can see it rather than squint at it.

Runs in sim mode: Play / Pause / Reset, where Reset re-randomizes the box so a new
run starts from a fresh disordered state with whatever coefficients the sliders
currently hold. Because the mode is not baked in, `--mode game` also works here --
it will pick a particle near the box centre and let you probe whatever has
assembled.

Units are the paper's LJ-reduced units (sigma = eps = m = 1).
"""
from ..playground import Lesson, Playground, random_fill
from ._knobs import NO_ORIENTATION
from ..render_style import DEFAULT_STYLE, CameraOrbit

# A dense, roughly cubic cloud of beads seen from outside is the scene the look
# was tuned on, so this takes the defaults -- but they are written out here, at
# their default values, because these are the dials worth reaching for first.
# Every field and what it does is in lammps_live/render_style.py; the two
# playgrounds next door show what changing them buys.
#
# Add `.on_light()` to the end of this call to draw the whole scene on a light
# background instead -- background, fog, lighting, box outline, net and spokes
# all move together, and everything tuned below survives the flip.
STYLE = DEFAULT_STYLE.varied(
    # The cell is periodic in all three axes, so its images could be drawn too
    # (as the sheet does): (1, 1, 1) would tile 3x3x3 around it, and
    # (0.5, 0.5, 0.5) would put a half-cell fringe of surrounding material round
    # it for a quarter of the instances. Left at the single real cell for now.
    periodic_images=(0, 0, 0),
    image_fade_start=0.7,
    image_fade_end=1.0,
    dof_focus=0.15,
    dof_range=0.40,
    dof_bokeh_px=5.0,
    # Fade into the background: complete by 55% of the depth span, and go 75% of
    # the way to the background colour when it does.
    cue_end=0.9,
    cue_strength=0.35,
    # Contact darkening between packed beads. An exponent, so higher = deeper
    # crevices with open surfaces left at full brightness.
    ao_strength=5.83,
    # The dark line around each bead: how black, and how far out from a bead's
    # centre (as a fraction of its radius) it starts. 0.94 gives a hairline,
    # 0.85 a heavy ink outline.
    outline_strength=12.0,
    outline_edge_fraction=0.90,
).on_light()

PLAYGROUND = Playground(
    name="MesoMem self-assembly (3D)",
    description="Paper's spontaneous-assembly run: 1500 random beads in a periodic "
                "20-sigma box coarsen into membranes. Play / Pause / Reset.",
    force_field="mesomem",
    scenario=random_fill(
        n=1500, box=20.0, overlap=0.9, maxtry=200,
        # Two nudges that keep a long run watchable, neither part of the paper's
        # experiment; 0 turns either off. `k_upright` is a weak field pulling
        # each director toward the nearest of +/-z, so the membranes that
        # nucleate tend to lie flat instead of at whatever angle they happen to
        # pick (measured: 4-8 degrees off vertical with it, 35-64 without, and
        # the assembly itself proceeds the same either way). `center_accel`
        # drifts whatever has assembled back to the middle of the cell, and only
        # along axes it is actually concentrated on -- so a droplet gets centred
        # in all three, a lamella only across its own thickness, and a gas not at
        # all. See RandomFill in playground/scenario.py for both.
        k_upright=0.6,
        center_accel=0.05,
    ),
    mode="sim",
    observables=["coordination"],
    # CLUSTER FIRST, and it is the default this scene comes up in: what is
    # happening here IS aggregates finding each other, and the colouring that
    # paints them is the one that tells you so.
    bead_colors=("cluster", "director", "energy"),
    param_ranges={"k_splay": (0.0, 5.0)},
    # THE PANEL GETS SIMPLER HERE, not richer, and that is the point: this scene is
    # watched rather than driven, so there is no puller, no leash and nothing to
    # steer -- temperature, time and the turntable. Disclosure follows the lesson
    # rather than accumulating down the sequence.
    #
    # THE HERO KNOB IS THE MODEL'S CLAIM AGAIN, and this is the strongest place to
    # make it: the box has just built a membrane out of nothing, and taking the
    # orientation away un-builds it into droplets while everything else about the
    # run stays the same.
    lesson=Lesson(
        title="It builds itself",
        claim="The same beads, poured in at random. They find the sheet on their own.",
        instruction="Press Play and watch. Colours mark separate clusters as they merge.",
        hook="So the beads make a membrane. What is a membrane for?",
        hero_knobs=(NO_ORIENTATION,),
    ),
    presets={
        "paper": {},
        # Below the k_tilt ~ 10 threshold: compact isotropic droplets instead of
        # membranes. The clearest single-slider demonstration in the model.
        "compact_droplets": {"k_tilt": 4.0},
        "strongly_planar": {"k_tilt": 30.0},
        "isotropic_only": {"k_tilt": 0.0, "k_splay": 0.0, "wc": 0.0},
    },
    # Assembly needs finite T so beads can diffuse and anneal into flat membranes,
    # but below the ~eps attraction well so they stay condensed rather than boiling
    # back into a gas. The default sits in that fluid-membrane window.
    temperature=(0.0, 0.5),
    temperature_default=0.2,
    melt_temp=0.3,
    particle_radius=0.5,
    reduced_units=True,
    # The clearest case for it: coarsening is a slow collective change buried under
    # a fast thermal rattle, and the rattle is what a still frame is mostly made of.
    # Advanced slider, 0 (off) by default. See playground/smoothing.py.
    trajectory_smoothing=True,
    render_style=STYLE,
    # Nothing here is steered, and what forms is a 3D morphology -- whether the
    # aggregates are compact droplets or flat lamellae is exactly the thing one
    # fixed angle hides. So the camera turns: drag to orbit it by hand, wheel to
    # dolly, C to hand it back to the automatic turn.
    camera_orbit=CameraOrbit(autostart=True, speed=0.16),
)
