"""MesoMem self-assembly on a cluster GPU, drawn here -- the same demo, bigger.

Physically this is `mesomem_assembly`: the paper's spontaneous-lamellae run, N
directored beads dropped at random into a fully periodic cube at reduced volume
fraction phi, coarsening under Langevin dynamics. Every coefficient, the
thermostat, the two watchability nudges and the whole look are that playground's;
what changes is where it runs and therefore how big it can be.

    N = 50,000 beads in a 109 sigma cell, against 1,500 in a 20 sigma cell.

WHAT SETS THE SIZE. It used to be the *client*, not the GPU: the Python analysis
rebuilt a full pair list over every bead, which docs/a100-plan.md section 3
measured at 1.5 us/bead/chunk and which capped N near 11,000 however fast the
simulation ran. That wall is gone. The pair list is now built over a bounded
random sample of the beads (Analysis.MAX_PAIR_BEADS) with the dilution divided
back out, so the analysis costs the same at 50k as it did at 6k -- measured 113 ms
per due frame before, 11 ms after, five times a second on a 20 fps wire. What the
sampling costs is a per cent or two of noise on the HUD observables and the energy
bars, redrawn each frame so it averages away; what it buys is that the client is
no longer the thing that decides how big this can be.

The remaining per-bead costs on this end are the ones that must touch every bead
because they are drawn: the wire payload, the drawn-state filtering, and the
renderer's instance buffers.

THE DECK IS NOT WRITTEN OUT ANYWHERE. `docs/snellius/in.mesomem_100k` exists to
benchmark with, and had to be maintained by hand against this app's own setup.
Here the server builds this playground -- this file, this force field, this
scenario -- so there is one definition and no drift. Set `n` below and both ends
follow.

RUNNING IT. Select it in the app like any other playground; it comes up
disconnected, with a Connect panel. That panel asks the cluster for a GPU
(`salloc --no-shell`), ships this package to it, starts
`lammps_live.remote.server` inside the allocation, tunnels a port back, and
cancels the job when the window closes. The one thing it cannot do for you is the
login prompt, which is why the panel has a field for it.

For the pipeline without the cluster -- what the tests use, and the way to work on
the renderer while offline -- run the server on this machine:

    python -m lammps_live.remote.server --playground mesomem_remote \\
           --profile local --token dev --port 5723
    lammps-live --playground mesomem_remote --remote 127.0.0.1:5723 --token dev
"""
import math

from ..playground import Lesson, Playground, random_fill
from ..remote import RemoteTarget
from .mesomem_assembly import STYLE
from ..render_style import CameraOrbit

# The size, and the cell that puts it at the chosen volume fraction. Written as
# the relation rather than as a number so changing N keeps the physics: with
# Vp = (pi/6) sigma^3, phi = N*Vp/L^3, so L = (N*(pi/6)/phi)^(1/3). 10k at
# phi = 0.1 is 37.41 sigma, which holds several independent membranes rather than
# the single one the 1500-bead cell manages.
N_BEADS = 50_000
PHI = 0.04
BOX = (N_BEADS * (math.pi / 6.0) / PHI) ** (1.0 / 3.0)

# The visual style is imported, not copied: this is the same scene as the local
# assembly box and should look identical. It survives the change of scale because
# every parameter in it is a fraction -- of the scene's depth span, of a bead
# radius -- rather than an absolute length (see render_style.py).

PLAYGROUND = Playground(
    name="MesoMem self-assembly, remote GPU (3D)",
    description=f"{N_BEADS:,} beads assembling on a cluster A100, drawn here. "
                f"Play / Pause / Reset, and every slider, over the wire.",
    force_field="mesomem",
    scenario=random_fill(
        n=N_BEADS, box=BOX, overlap=0.9, maxtry=200,
        # Both as in mesomem_assembly. The centring correction is if anything more
        # useful here: at this size several aggregates form at once, and without it
        # they wander off the near clipping plane one at a time.
        k_upright=0.6,
        center_accel=0.05,
    ),
    mode="sim",
    observables=["coordination"],
    bead_colors=("cluster", "director", "energy"),
    param_ranges={"k_splay": (0.0, 40.0)},
    # THE CLUSTER IS PLUMBING, NOT THE LESSON, so neither line mentions a tunnel,
    # an allocation or a queue: what the audience should hear is the bead count and
    # that it is happening now. The connect panel already says everything about the
    # machinery, to the person driving, at the moment it matters.
    #
    # No hero knob: the one thing to do here is watch it, and a button that changed
    # the physics of a run somebody is waiting on a queue for is a button pressed
    # by accident.
    lesson=Lesson(
        title="At scale",
        claim="Ten thousand beads on a cluster GPU, computed there and drawn here.",
        instruction="Drag to orbit. This is running now, not a recording.",
        hook="This sheet has no edges, but it is not closed. What if it closes?",
    ),
    presets={
        "paper": {},
        "compact_droplets": {"k_tilt": 4.0},
        "strongly_planar": {"k_tilt": 30.0},
        "isotropic_only": {"k_tilt": 0.0, "k_splay": 0.0, "wc": 0.0},
    },
    temperature=(0.0, 0.5),
    temperature_default=0.2,
    melt_temp=0.3,
    particle_radius=0.5,
    reduced_units=True,
    trajectory_smoothing=True,
    render_style=STYLE,
    camera_orbit=CameraOrbit(autostart=True, speed=0.16),
    # The frame budget, and the only concession this file makes to its size. The
    # energy panels are a pass over every pair, and while the pair list itself is
    # now bounded by sampling, halving the cadence of the one consumer whose
    # aggregate barely changes between frames is still free. The observables keep
    # their own declared cadences.
    analysis_energy_every=8,
    # Where it runs. Nothing here is a private fact: every field is overridden by
    # a `[remote]` block in your config file (lammps-live --write-config) and then
    # by LAMMPS_LIVE_REMOTE_* -- so a second person runs this on a second cluster
    # without touching this file. See userconfig.py and docs/cluster-setup.md.
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
