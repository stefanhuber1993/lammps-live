"""A MesoMem vesicle with ONE space-filling polymer inside it, on a cluster GPU.

The sibling of `mesomem_polymer` (which it replaced in the demo; that file is
shelved, not gone). The envelope is exactly that scene's -- the paper's beads on a
closed sphere, directors radial, the spacing a vesicle is tension-free at -- and
the lumen holds a single closed chain of 32,768 beads instead of a melt of 62
small rings. The chain is laid along a 3D Moore curve: a Hilbert curve closed
into a loop, visiting every site of a 32-site cube once, folded at every scale.
That is the crumpled ("fractal") globule picture of interphase chromatin, and
nothing about it has to be relaxed into place: every bond is one lattice step and
no two beads start closer than one (see state.moore_curve, VesicleChain).

WHAT THERE IS TO SEE. The chain starts as a dense cube in the middle of the
lumen, eight sigma clear of the membrane at its corners, and the first thing it
does is swell out to meet the envelope. The rainbow is along the chain (see
VesicleChain.render_tints): neighbours along a Moore curve are neighbours in
space, so the block begins as clean colour domains, one per stretch of the
chain, and how quickly those domains blur is the chain's own relaxation. Cut the
vesicle open with the thrust lever (see lammps_live/view_slice.py) -- a closed
monolayer is opaque, and from outside this is a sphere of beads.

THE HERO KNOB is the chain's stiffness, and it is a toggle between two extremes
rather than a nudge off the reference:

  * FLEXIBLE, k_bend = 1, is where the scene starts: persistence of five bonds
    at T = 0.2, a flexible chain that is still a chain rather than an ideal
    random walk (it was 0 until 2026-09-30; the user's call). It spreads
    through the lumen but keeps its crumpled, domain-by-domain structure for a
    long time, which is the picture the rainbow is for. It is also a gentle
    place to START a Moore curve: every one of its thousands of right-angle
    corners costs k_bend in `angle_style cosine`, 1 eps each here -- five kT,
    which the thermostat takes away in the first few tau -- where a stiff chain
    built on it begins with two hundred times that at once.
  * STIFF, k_bend = STIFF_K_BEND below, with the bath cooled to STIFF_T. The
    chain becomes a tangle of wire that wants to be far bigger than its
    container and presses the envelope out from inside: the vesicle swells and
    goes lumpy and faceted where loops of chain push on it. See the numbers at
    STIFF_K_BEND for why it is exactly this far and no further.

k_tilt and eps_poly are the other two dials, as on `mesomem_polymer`: how much
bending the envelope tolerates, and how hard the (never sticky) contact is.

SIZE. 23,120 membrane beads at a = 0.70 make a vesicle of radius 35.3 sigma. An
order-5 curve is 32^3 = 32,768 beads on a unit lattice (the bond length, which
the FENE bond's rest length is chosen to match), whose half-diagonal is
15.5 * sqrt(3) = 26.8 sigma: 8.4 sigma of clearance to the membrane however
`create_atoms ... mol` happens to rotate it. 55,888 particles in total, the size
`mesomem_remote` established a GPU can run and this end can draw. The next curve
up is eight times the beads, so `chain_order` is not the knob for "a bit more".

RUNNING IT. As `mesomem_polymer`: select it, connect through the panel, and the
cluster builds this same file at the other end. For the pipeline without the
cluster, run the server on this machine:

    python -m lammps_live.remote.server --playground mesomem_vesicle_chain \\
           --profile local --token dev --port 5723
    lammps-live --playground mesomem_vesicle_chain --remote 127.0.0.1:5723 \\
           --token dev
"""
from ..playground import HeroKnob, Lesson, Playground, vesicle_chain
from ..remote import RemoteTarget
from ..render_style import DEFAULT_STYLE, CameraOrbit

# mesomem_polymer's look, unchanged: the assembly box's, with two changes, both
# because the subject is one object at the centre of the cell rather than a cell
# full of them.
#
#   DEPTH OF FIELD  focused mid-scene, on the middle of the vesicle, and over a
#     wide range: a sphere 74 sigma across seen from 2.6 cell widths away has its
#     near and far poles at genuinely different distances, and a shallow focus
#     would leave only a band of it sharp. What the blur is doing here is
#     separating the near wall from the far one, which is exactly the confusion a
#     closed surface creates.
#   DEPTH CUE  reaching much further back than the assembly box's, and gentler.
#     Further back because a sliced vesicle's far wall is the one thing that can be
#     mistaken for its near one, and fading them apart is what separates them;
#     gentler because a slice is mostly empty space and a strong fade over it
#     leaves half the section barely there.
#   BOX OUTLINE  kept. The cell is not periodic here -- a vesicle is a finite
#     object with vacuum around it -- and the outline is what gives the sphere a
#     scale to be seen against, and what the slicing plane visibly travels through.
STYLE = DEFAULT_STYLE.varied(
    periodic_images=(0, 0, 0),
    dof_focus=0.45,
    dof_range=1.20,
    dof_bokeh_px=4.0,
    cue_start=0.35,
    cue_end=0.95,
    cue_strength=0.40,
    ao_strength=5.83,
    outline_strength=12.0,
    outline_edge_fraction=0.90,
    # THE CHAIN ON ITS OWN ENERGY SCALE, in the energy colouring: each chain bead
    # carries its BENDING energy k_bend (1 + cos theta) on theme.BEND_RAMP (blue
    # to mint), while the membrane keeps its binding energy on INFERNO. 0..1 eps
    # is 0 to ten kT at the stiff knob's T = 0.1: a thermally relaxed stiff chain
    # sits in the bottom tenth, and what lights up is the bending that has not
    # relaxed yet -- every corner of the Moore curve at once, the moment the knob
    # is pressed, then fading as the chain straightens. Flexible (k_bend 1) its
    # thermal bends sit around kT = 0.2 on this scale: dark, faintly speckled.
    tint_energy_range=(0.0, 1.0),
).on_light()

# The two sizes. The membrane's is a REQUEST (an icosphere only comes in
# 20*nu^2 beads); the chain's is exact, 8**CHAIN_ORDER.
N_MEMBRANE = 23_120
CHAIN_ORDER = 5

SCENARIO = vesicle_chain(
    n_membrane=N_MEMBRANE,
    chain_order=CHAIN_ORDER,
    # THE SPACING A CLOSED VESICLE IS TENSION-FREE AT, which is NOT the flat
    # sheet's and is the whole reason this scene used to tear itself open.
    #
    # A periodic sheet runs under `fix nph/sphere`: it is handed a spacing,
    # the barostat adjusts the cell, and the membrane reaches zero tension
    # whatever it started from (mesomem_rod measures 0.775 that way). A
    # VESICLE HAS NO BAROSTAT. Its area per bead is fixed at build time by the
    # bead count and the radius -- and the radius is itself derived from the
    # spacing -- so a spacing that is too loose is a lateral tension the
    # membrane cannot relieve by any amount of running.
    #
    # What it does instead is tear. Measured on the bare vesicle (no polymer,
    # 18,000 beads, 2000 steps), as `a` comes down:
    #
    #     a       R shrinks by   largest gap   surface open
    #     0.800      -3.9%          3.3 sigma      2.3%
    #     0.775      -3.8%          3.0            1.2%
    #     0.740      -3.6%          2.1            0.02%
    #     0.700      -2.2%          0.7            0%
    #     0.660      +0.1%          0.7            0%
    #
    # 0.7 sigma of gap is the defect-free floor (a/sqrt(3) plus thermal
    # roughness), so at 0.70 the envelope is intact and stays intact; above
    # 0.74 it sheds its excess area as ~70 small pores spread over the sphere
    # -- not at the icosahedron's twelve five-fold vertices, which are its
    # TIGHTEST-packed spots and the last to open -- and those pores do not
    # close again. The shrinkage column is the same story read as tension: the
    # envelope contracting is it trying, and failing, to reach this spacing.
    #
    # Denser also COVERS better, so nothing is traded away here: the old
    # comment worried about a monolayer of radius-sigma/2 beads ceasing to
    # cover itself above a = sqrt(3)/2 ~ 0.87, and every number that fixes the
    # tearing moves away from that bound, not toward it. The cost is bead
    # count: holding the vesicle at the ~35 sigma this demo is framed for
    # takes 23,120 beads at 0.70 where it took 18,000 at 0.80. See N_MEMBRANE.
    a=0.70,
    # ROOM FOR IT TO GO WRONG. 1.12 (mesomem_polymer's) was just enough vacuum
    # for the vesicle to bulge; held stiff long enough it tears open and the
    # chain spills out (see STIFF_K_BEND), and in a cell that tight the spill
    # met the walls at once. 1.6 leaves ~21 sigma on every side -- the chain
    # has somewhere to come out INTO, which is the thing worth watching. The
    # camera still frames the vesicle, not the cell (see VesicleChain.fit_points).
    box_factor=1.6,
    settle_steps=400,
    # 10 steps a frame. The membrane's own stability sets the ceiling on the
    # step size and the chains do not lower it -- FENE at these constants is
    # stable well past 0.005 -- so this is the sheet playgrounds' number. The
    # stiff chain does not lower it either, once ramped in: at STIFF_K_BEND no bond
    # passed 1.05 sigma over 40 tau of the full system (see tests/test_vesicle_chain
    # for the miniature). Jumped to, it breaks one -- that is energy, not the step.
    timestep=0.005,
    sim_time_per_frame=0.05,
)

# THE COUNTS THE CARD QUOTES, asked of the scenario rather than written down, as
# on mesomem_polymer: a number on a screen in front of a room has to be the one
# actually running.
_SP = SCENARIO.new_params()
N_MEMBRANE_RUN = SCENARIO.subdivision(_SP)[1]
N_POLYMER_RUN = SCENARIO.particle_count(_SP) - N_MEMBRANE_RUN
N_TOTAL_RUN = N_MEMBRANE_RUN + N_POLYMER_RUN

# THE STIFF END OF THE HERO KNOB, found by running the shipped system (56k beads,
# ramped in over 10 tau from floppy, then 40 tau held) and looking at it. k_bend 10,
# the old value, is a persistence length of 50 bonds and does nothing you can see:
# the radius moves by a tenth of a percent. The envelope only answers once the
# chain is stiff on the scale of the whole lumen, and past that the answer is not
# a bigger shape change but a torn membrane:
#
#     k_bend   T      R grows   surface       what you see
#       10    0.20     +0.1%    intact        nothing
#      100    0.20     +2.5%    intact        lumpy, faceted
#      200    0.20     +6%      one pore      a loop herniates through it
#      300    0.20     +9%      pores         chain spilling out of a dozen
#      200    0.10     +4%      intact        swollen, faceted -- THIS
#      250    0.10     +5%      pores         the first herniations
#
# HELD, IT STILL GIVES WAY EVENTUALLY. The 200/0.10 run carried on to 150 tau keeps
# swelling (+7.5%), starts to stretch out of round, and at about 125 tau -- some 80
# seconds at the demo's rate -- opens ONE large pore that the chain bulges out of.
# So it is a move to make and take back within a minute; left on, it ends in a
# rupture rather than a blow-up, which the Reset button undoes.
#
# The membrane gives out before the chain does because a vesicle's area is fixed:
# pressure from inside can only STRETCH it, and a stretched monolayer opens pores
# (the same failure the spacing comment above is about). Cooling it is what buys
# the extra stiffness: pores are thermally activated, the chain's push is not. The
# other membrane dials were tried and are worse -- k_tilt down to 5 dissolves the
# envelope, up to 25 still tears at 300, and zeta 2-3 holds it together as one
# sheet but opens holes the size of the chain's loops.
#
# IT HAS TO BE RAMPED. Switched on in one step, k_bend 200 loads every corner of
# the curve at once, heats the system fivefold and breaks a FENE bond within a few
# hundred steps; reached over RAMP_TAU the thermostat takes the heat as it
# arrives. HeroKnob.ramp_time does that in simulation time.
FLOPPY_K_BEND = 1.0
STIFF_K_BEND = 200.0
STIFF_T = 0.1
RAMP_TAU = 10.0
_T = 0.2

# NAMED FOR THE MATERIAL, not the move: "Stiff chain" said what the button did
# to a parameter, where the audience's word is "polymer", and "stiff polymer" is
# the thing they can compare with something they know (DNA is one; a cooked
# noodle is not). The caption then says what stiffness COSTS -- bending energy --
# in the numbers the energy colouring paints on the chain.
#
# AND A VERB ON THE BUTTON, like every knob now: "Stiff polymer chain" could not
# say whether the chain already was, or would be once pressed.
STIFF_CHAIN = HeroKnob(
    label="Make the polymer stiff",
    engaged_label="Make it flexible again",
    caption=f"Bending now costs energy: k_bend = {STIFF_K_BEND:.0f}, up from "
            f"{FLOPPY_K_BEND:.0f}, at T = {STIFF_T:.2f}. The chain straightens "
            f"and presses the vesicle out.",
    params={"k_bend": STIFF_K_BEND},
    temperature=STIFF_T,
    ramp_time=RAMP_TAU,
)


def _k(n):
    """55_888 -> "56k". Rounded, not truncated: 32,768 beads of chain is 33k
    and calling it 32k understates what is on the screen."""
    return f"{round(n / 1000)}k"


PLAYGROUND = Playground(
    name="MesoMem vesicle + space-filling chain, remote GPU (3D)",
    description=f"A {_k(N_MEMBRANE_RUN)}-bead closed membrane around one "
                f"{_k(N_POLYMER_RUN)}-bead polymer laid on a Moore curve, on a "
                f"cluster A100. Slice the view with the thrust lever to see in.",
    force_field="mesomem_polymer",
    scenario=SCENARIO,
    mode="sim",
    observables=["vesicle_radius", "polymer_gyration", "polymer_contact"],
    bead_colors=("director", "energy"),
    # k_bend starts at the floppy end (see the module docstring for why that is
    # the end to start at), and its slider must reach the stiff one.
    params={"k_bend": FLOPPY_K_BEND},
    param_ranges={"k_splay": (0.0, 5.0), "k_bend": (0.0, STIFF_K_BEND)},
    # THE LAST SLIDE, AND IT CLOSES THE FIRST ONE: the two-bead scene opens by
    # saying a bead is a patch of membrane and all we kept of it was the way it
    # faces, and this is where that turns out to have been enough to close a
    # bilayer around something. So the claim keeps "the same three terms"
    # (tests/test_lessons.py pins the bookend). No hook: nothing comes after it.
    #
    # THE TITLE SAYS WHAT IS ON THE SCREEN, plainly, where mesomem_polymer's was a
    # count. The counts move to the claim, interpolated from N_*_RUN, never typed.
    #
    # ONE HERO KNOB, unlike the other remote scene, and the exception is on
    # purpose: the chain's stiffness is the one move this scene is built to show,
    # it changes no structure (an angle coefficient, re-issued live), and it is
    # a toggle between two named extremes rather than a dial someone can leave
    # somewhere odd on a run a room is waiting for.
    lesson=Lesson(
        title="Closed membrane vesicle enclosing one continuous space-filling "
              "polymer chain",
        ui=("panel", "energy", "colour", "plots", "readings", "status", "snellius", "slice"),
        claim=f"{_k(N_MEMBRANE_RUN)} membrane beads around one "
              f"{_k(N_POLYMER_RUN)}-bead chain. Still the same three terms.",
        hero_knobs=(STIFF_CHAIN,),
    ),
    presets={
        # Where the scene starts: a flexible chain, k_bend 1.
        "floppy_chain": {},
        # The collaborator's deck, k_bend 2 -- the semi-flexible middle.
        "reference": {"k_bend": 2.0},
        # A stiff chain as a place to START -- a harsh one, since every corner of
        # the Moore curve begins loaded. NOT the hero knob's value: 200 from a
        # standing start breaks a bond (see STIFF_K_BEND); 10 is survivable.
        "stiff_chain": {"k_bend": 10.0},
        # A floppy envelope against the same chain: the membrane is what gives.
        "soft_envelope": {"k_tilt": 4.0},
        # The chain pushed hard against a membrane that will not bend.
        "crowded": {"eps_poly": 3.0, "k_tilt": 30.0},
    },
    # The reference deck runs the membrane at 0.2 and the chains at 1.0. One bath
    # here (see the force field's docstring), at the membrane's temperature: it is
    # the one whose physics is temperature-sensitive, and a chain at 0.2 in reduced
    # units is a flexible chain, not a frozen one.
    # The panel plots' fixed y ranges (see Playground.plot_ranges), measured on the full 56k system: P 0.005, 0.02 at T = 0.5, 0.02-0.05 stiffened; PE per bead +10.8 (positive: every chain bead holds a stretched FENE bond, about +18), to 12 at T = 0.5 and 13.6-15.7 while the stiff chain ramps in; KE 0.2-0.8 along the bottom.
    plot_ranges={"press": (-0.02, 0.06), "energy": (0.0, 16.0)},
    temperature=(0.0, 0.5),
    temperature_default=_T,
    melt_temp=0.3,
    particle_radius=0.5,
    reduced_units=True,
    trajectory_smoothing=True,
    render_style=STYLE,
    # Nothing is steered and the subject is a closed 3D object, so the camera
    # turns: drag to orbit, wheel to dolly, C to hand it back. Slower than the
    # assembly box's, so a sliced section can be followed round as it turns.
    #
    # dist_max 4 rather than the default 1.5: the framing is the vesicle, and
    # pulling back far enough to see the whole (now larger) cell, and whatever
    # has escaped into it, takes more than the default's half again.
    camera_orbit=CameraOrbit(autostart=True, speed=0.10, dist_max=4.0),
    # As mesomem_remote: the energy panels are a pass over every pair and their
    # aggregate barely moves between frames, so halving their cadence is free.
    analysis_energy_every=8,
    remote=RemoteTarget(
        host="snellius.surf.nl",
        user="stefanh",
        label="Snellius gpu_a100",
        partition="gpu_a100",
        gpus=1,
        ntasks=1,
        cpus_per_task=18,
        time="10:00:00",
        remote_dir="~/Projects/MesoMemLive/mesomem_gpu",
        env_script="_build/hpc/env.sh",
        profile="cluster-gpu",
    ),
)
