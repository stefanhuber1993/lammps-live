"""A MesoMem vesicle with chromatin inside it, on a cluster GPU, drawn here.

The collaborator's system (`polymer/` at the repo root, and his `polymer.lmp`):
a closed spherical membrane -- the paper's own beads, directors radial -- with a
melt of ring polymers sealed in the lumen. A minimal, and rather good, cartoon of
a nucleus: the DNA is a set of closed loops that cannot pass through one another,
the envelope is a fluid bilayer that can bulge and cannot leak, and NOTHING
between them attracts. Every shape the vesicle takes is therefore something the
polymer pushed it into.

WHAT THERE IS TO SEE. The melt is laid in as compact lattice rings on a coarse grid
through the lumen -- the right density, but not yet equilibrium: within the first
few hundred steps the rings swell into one another and out into the last couple of
sigma of clearance, and from there the melt and the envelope are in contact and
stay that way. Then the dials are the questions:

  * k_bend is the chains' stiffness, and it is the one to reach for first. At 0
    each ring collapses toward a globule and the melt pulls away from the wall; at
    the reference 2 it is a semi-flexible tangle that fills the lumen; well above
    that the rings straighten enough to press the envelope out of round.
  * k_tilt is the membrane's, as everywhere else in this demo: how much bending
    the envelope will tolerate before the polymer's push starts to show.
  * eps_poly is how hard the contact between them is. Never sticky -- the pair is
    cut at its minimum -- so turning it up makes the polymer press harder, not
    adhere.

YOU CANNOT SEE ANY OF IT FROM OUTSIDE, which is the other half of this file's
point. A closed monolayer is opaque, and from any angle at all the picture is a
sphere of beads with the interesting half behind it. So this is the playground the
thrust lever exists for: push it and the view cuts to a slab a few per cent of the
box thick, normal to whichever cardinal axis you are looking down, and the lever's
travel sweeps that slab through the cell -- envelope in section as a ring, the melt
inside it as a cross-section of coloured strand. Let go of it for three seconds and
the box closes back up. See lammps_live/view_slice.py; it works on every 3D
playground, this is just the one it is indispensable on.

THE COLOURS SAY WHICH IS WHICH. The membrane keeps the MesoMem banding -- yellow
equator, blue poles, tilting with the director -- because on those beads that
banding is the physics. The polymer has no orientation to show, so its beads carry
a teal-to-magenta ramp along each ring's own contour instead, which is the one
thing about them nothing else in the picture shows: a strand can be followed by eye
through the melt (see VesiclePolymer.render_tints).

SIZE, AND WHY THESE NUMBERS. 23,120 membrane beads at the spacing a closed
vesicle is tension-free at make a vesicle of radius ~35 sigma; 62 rings of 512
beads fill it at about the reference system's volume fraction. ~55,000 particles
in total, which
is the size `mesomem_remote` established a GPU can run and this end can draw (see
that file on what stopped being the limit and why). The collaborator's own run is
twice this -- 35,280 membrane beads and 125 rings -- and `n_membrane` / `n_polymer`
below are the two numbers to raise if the wire and the window turn out to have the
headroom -- but `n_membrane` is not free at a fixed radius, because the spacing it
implies is what decides whether the envelope holds together (see `a` below).

RUNNING IT. As `mesomem_remote`: select it, connect through the panel, and the
cluster builds this same file at the other end. For the pipeline without the
cluster, run the server on this machine:

    python -m lammps_live.remote.server --playground mesomem_polymer \\
           --profile local --token dev --port 5723
    lammps-live --playground mesomem_polymer --remote 127.0.0.1:5723 --token dev
"""
from ..playground import Lesson, Playground, vesicle_polymer
from ..remote import RemoteTarget
from ..render_style import DEFAULT_STYLE, CameraOrbit

# The assembly box's look, with two changes, both because the subject is one
# object at the centre of the cell rather than a cell full of them.
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
).on_light()

# The two sizes, kept here rather than inline because they are what to change and
# because the description below quotes them.
N_MEMBRANE = 23_120
N_POLYMER = 32_000

SCENARIO = vesicle_polymer(
    n_membrane=N_MEMBRANE,
    n_polymer=N_POLYMER,
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
    # Just enough vacuum around the vesicle that it can bulge, and that the
    # outline reads as a container rather than as a tight shrink-wrap.
    box_factor=1.12,
    # Laid over nine tenths of the lumen's radius, which puts the melt at
    # about the reference system's volume fraction (~0.1) and leaves a couple
    # of sigma of clearance to the wall. So the scene STARTS as a full
    # nucleus -- the picture this playground is about -- and the first thing
    # it does is swell that last stretch into contact.
    fill_fraction=0.90,
    # 512 beads a ring, as the reference system. Even, which the ring
    # construction requires (see state.lattice_ring).
    ring_side=8,
    settle_steps=400,
    # 10 steps a frame. The membrane's own stability sets the ceiling on the
    # step size and the chains do not lower it -- FENE at these constants is
    # stable well past 0.005 -- so this is the sheet playgrounds' number.
    timestep=0.005,
    sim_time_per_frame=0.05,
)

# THE COUNTS THE CARD QUOTES, asked of the scenario rather than written down.
# `n_membrane` is rounded to an icosphere size (they only come in 20*nu^2) and
# `n_polymer` down to whole rings, so the two constants above are REQUESTS and
# these are the answers. A number on a screen in front of a room has to be the
# one actually running, and the two differ by 256 beads as it happens.
_SP = SCENARIO.new_params()
N_MEMBRANE_RUN = SCENARIO.subdivision(_SP)[1]
N_POLYMER_RUN = SCENARIO.particle_count(_SP) - N_MEMBRANE_RUN
N_TOTAL_RUN = N_MEMBRANE_RUN + N_POLYMER_RUN


def _k(n):
    """55_000 -> "55k". Rounded, not truncated: 31,744 beads of polymer is 32k
    and calling it 31k understates what is on the screen."""
    return f"{round(n / 1000)}k"


PLAYGROUND = Playground(
    name="MesoMem vesicle + polymer, remote GPU (3D)",
    description=f"A {_k(N_MEMBRANE_RUN)}-bead closed membrane with "
                f"{_k(N_POLYMER_RUN)} beads of ring polymer sealed inside, on a "
                f"cluster A100. Slice the view with the thrust lever to see in.",
    force_field="mesomem_polymer",
    scenario=SCENARIO,
    mode="sim",
    observables=["vesicle_radius", "polymer_gyration", "polymer_contact"],
    bead_colors=("director", "energy"),
    param_ranges={"k_splay": (0.0, 5.0)},
    # THE LAST SLIDE, AND IT CLOSES THE FIRST ONE. The two-bead scene opens by
    # saying that a bead is a patch of membrane and all we kept of it was the way it
    # faces; this is where that turns out to have been enough to close a bilayer
    # around something. Same three terms, eight scenes apart, and the callback is
    # why the opening line is worded the way it is.
    #
    # No hook: there is nothing after this one. No hero knob either, for the remote
    # scene's reason.
    #
    # THE COUNT IS IN THE TITLE HERE, where the sibling remote scene keeps it in
    # the claim. Both are deliberate and they are not the same situation: "At
    # scale" has one number and the claim can carry it in words, while this scene
    # has TWO worth saying (the envelope and what is sealed in it) and a total
    # that is the largest thing the demo runs. So the title says the total, which
    # is the number a presenter says out loud and the one that is impressive
    # because it is happening on a machine somewhere else, and the claim breaks it
    # into the two halves the picture actually shows.
    #
    # Interpolated from N_*_RUN, never typed: raise the bead counts above and this
    # line follows, where a hardcoded "55k" would quietly start lying.
    lesson=Lesson(
        title=f"A vesicle, {_k(N_TOTAL_RUN)} beads",
        claim=f"{_k(N_MEMBRANE_RUN)} of membrane closed around "
              f"{_k(N_POLYMER_RUN)} of polymer. Still the same three terms.",
        instruction="Cut the vesicle open with the lever to see inside it.",
    ),
    presets={
        # The collaborator's deck: k_bend 2, the paper's membrane moduli.
        "reference": {},
        # Ideal flexible rings. Each collapses toward a globule and the melt
        # shrinks away from the envelope -- the comparison that shows the wall
        # contact is the chains' stiffness rather than their bulk.
        "floppy_chains": {"k_bend": 0.0},
        # Stiff enough that the rings resist being folded into the lumen and
        # press on it instead.
        "stiff_chains": {"k_bend": 12.0},
        # A floppy envelope against the same melt: the membrane is what gives.
        "soft_envelope": {"k_tilt": 4.0},
        # The polymer pushed hard against a membrane that will not bend.
        "crowded": {"eps_poly": 3.0, "k_tilt": 30.0},
    },
    # The reference deck runs the membrane at 0.2 and the chains at 1.0. One bath
    # here (see the force field's docstring), at the membrane's temperature: it is
    # the one whose physics is temperature-sensitive, and a melt at 0.2 in reduced
    # units is a melt, not a frozen chain.
    temperature=(0.0, 0.5),
    temperature_default=0.2,
    melt_temp=0.3,
    particle_radius=0.5,
    reduced_units=True,
    trajectory_smoothing=True,
    render_style=STYLE,
    # Nothing is steered and the subject is a closed 3D object, so the camera
    # turns: drag to orbit, wheel to dolly, C to hand it back. Slower than the
    # assembly box's, because the slicing plane re-aims itself when the view swings
    # far enough off it (view_slice.REAIM_DEGREES) and a brisk orbit would keep
    # doing that under you.
    camera_orbit=CameraOrbit(autostart=True, speed=0.10),
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
        time="01:00:00",
        remote_dir="~/Projects/MesoMemLive/mesomem_gpu",
        env_script="_build/hpc/env.sh",
        profile="cluster-gpu",
    ),
)
