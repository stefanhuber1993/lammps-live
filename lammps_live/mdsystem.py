"""Abstract interface every simulated "system" (material + geometry) must
implement, plus the metadata dataclasses the UI and force-feedback layer
read to configure themselves per-system.

All systems use LAMMPS "metal" units (eV, Angstrom, ps, amu) -- real
physical scales, not reduced LJ units -- so numbers on screen mean what a
physicist expects them to mean regardless of which system is active.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .render_style import DEFAULT_STYLE, RenderStyle


@dataclass(frozen=True)
class SliderSpec:
    """UI-facing description of a live-adjustable parameter (temperature,
    puller damping, ...). Ranges are per-system: a value tuned for copper's
    force scale is meaningless for a soft LJ gas."""
    label: str
    vmin: float
    vmax: float
    default: float
    fmt: str = "{:.3f}"
    unit: str = ""
    # Optional "sweet spot" value to mark on the track (a distinct tick + "opt"
    # label), so a user who has dragged a parameter around can find the paper's
    # recommended setting again. None -> no optimum marker drawn.
    optimum: float = None
    # Stable id for extra_sliders (see SystemSpec): the app passes this back to
    # MDSystem.set_extra_param so a system knows which live parameter changed.
    # Empty for the built-in temperature/damping sliders, which have dedicated
    # setters (set_target_temp / set_puller_damping).
    key: str = ""
    # "Advanced" parameters are hidden by default behind a collapsible "Advanced"
    # toggle in the panel, so the everyday controls stay uncluttered. The user
    # clicks the toggle to reveal (and drag) them. Applies to both built-in and
    # extra sliders.
    advanced: bool = False


@dataclass(frozen=True)
class ForceFeedbackProfile:
    """Per-system tuning for translating interaction forces into
    force-feedback / on-screen-arrow signals. These live per-system (not as
    global constants) because they're scaled to each system's characteristic
    force magnitude -- EAM copper contact forces run ~0.1-6 eV/A, a soft LJ
    gas is far weaker, and a single global knee/threshold would make one of
    them feel numb or pegged at max.
    """
    ff_exaggeration: float     # amplifies small/medium contact forces before tanh soft-saturation
    ff_knee: float             # raw*exaggeration magnitude at the soft-saturation knee
    ff_max_mag: float = 120.0  # device-unit cap, comfortably inside the SDK's +-127 spring-offset range
    stiffness_threshold: float = 0.05  # eV/A -- below this, spring/arrow reads as fully limp
    stiffness_knee: float = 0.5        # eV/A -- ramp scale above threshold
    damper_min_fraction: float = 0.10  # of DAMPER_COEFFICIENT_MAX, when idle
    damper_max_fraction: float = 0.50  # of DAMPER_COEFFICIENT_MAX, in contact
    vel_damp_max_fraction: float = 0.5  # of CP_OFFSET_MAX, velocity-opposing spring-center bias


@dataclass(frozen=True)
class SystemSpec:
    """Static, UI-facing description of a system -- everything the renderer
    and control loop need without having to introspect the LAMMPS instance."""
    key: str                 # stable id, used on the CLI (--system) and for cycling
    name: str                 # short display name
    description: str          # one-liner shown in the panel header
    element_label: str        # legend text, e.g. "Cu (EAM)"
    lattice_spacing: float    # Angstrom -- equilibrium in-plane spacing, informational
    timestep: float           # ps -- used to convert "sim time per frame" into a step count
    temperature: SliderSpec
    damping: SliderSpec
    melt_temp: float          # K -- approximate dial marker (see each system's docstring)
    force_feedback: ForceFeedbackProfile
    puller_speed_cap: float   # Angstrom/ps -- nve/limit-derived velocity ceiling, for velocity-damping scale
    # The single maximum interactive force (eV/Angstrom in metal units, reduced
    # units for the MesoMem systems) applied to the puller at full deflection of
    # ANY input device -- joystick, WASD keyboard, or mouse all map their unit
    # deflection through this same scale, so the three controls share one ceiling.
    # This is the honest "how hard can I push" knob; it lives here on the system
    # (not buried in the force-feedback profile) because it's a property of the
    # interaction, independent of how forces are rendered back to the device.
    max_input_force: float
    # Per-species rendering, for systems whose atoms aren't a single
    # indistinguishable species (see get_all_positions' `species`). None ->
    # every crystal atom drawn the same flat color (Cu, Ar). Otherwise an RGB
    # tuple per species index, with an optional matching glyph per species
    # (e.g. "+"/"-" for ions; None entries draw no glyph).
    species_colors: tuple = None   # (RGB, RGB, ...) indexed by species, or None
    species_labels: tuple = None   # (str-or-None, ...) indexed by species, or None
    # Flat draw color for single-species systems (Cu, Ar), so they read as
    # distinct materials -- warm metallic copper vs. cold noble-gas argon --
    # instead of sharing one generic crystal color. None falls back to the
    # theme's CRYSTAL_COLOR. Ignored when species_colors is set.
    crystal_color: tuple = None
    # On-screen atom size, given as a PHYSICAL radius in Angstrom (converted to
    # pixels per-frame at the active box's scale, then clamped) so atoms are
    # drawn at their real relative sizes: argon's atom genuinely dwarfs copper's,
    # a Cl- anion genuinely dwarfs a Na+ cation, and the packing you see is the
    # real packing. atom_radius_A is the single-species value; species_radii_A,
    # if set, gives a per-species radius (indexed like species_colors). Both
    # None -> the theme's fixed-pixel CRYSTAL_RADIUS fallback.
    atom_radius_A: float = None
    species_radii_A: tuple = None
    # Real MD time advanced per rendered frame, in ps (overrides the global
    # config.SIM_TIME_PER_FRAME). Mesoscale systems (the coarse-grained lipid
    # membrane) evolve on a much slower intrinsic time scale, so at the shared
    # default a whole frame barely moves them and the membrane looks frozen /
    # jittery rather than a living, self-healing fluid -- they need a larger
    # per-frame time slice to come alive. None -> use the global default.
    sim_time_per_frame: float = None
    # Whether to draw the generic "faint line between atoms near their
    # equilibrium spacing" bond overlay. On for crystals; off for systems that
    # supply their own explicit bonds to draw (see get_bond_pairs), e.g. lipids.
    bond_overlay: bool = True
    # Whether this system renders as a 3D scene (perspective + depth-cued
    # spheres and directors) rather than the default top-down 2D box. A 3D
    # system additionally implements get_positions_3d / get_dipoles_3d /
    # get_bonds_3d / get_camera_params / get_control_grid, and the app routes
    # those to the renderer's 3D path. The standard 2D readouts (thermo, puller
    # energy, force feedback) are unchanged -- they act on the 2D control-plane
    # projection of the puller the system already returns.
    render_3d: bool = False
    # 3D only: draw the little per-bead director spike. On for small scenes
    # (7-bead patch) where it shows the flip; off for large sheets (hundreds of
    # beads) where hundreds of spikes are clutter and a per-bead draw cost -- the
    # banded pole/equator coloring already shows tilt there.
    director_arrows: bool = True
    # 3D only: extra live-tunable parameters beyond temperature/damping, drawn as
    # additional sliders in the panel. Each SliderSpec's `key` is handed back to
    # set_extra_param(key, value) when the user moves it. Empty -> no extra
    # sliders (every non-MesoMem system).
    extra_sliders: tuple = ()
    # 3D only: for a periodic scene, the crossfade band width as a fraction of
    # the box side. A bead within this fraction of an x/y seam dissolves toward
    # the background while a wrapped ghost fades in at the opposite edge, so it
    # slides across the periodic boundary instead of popping. 0 -> no wrap fade
    # (non-periodic scenes: the 7-bead patch).
    wrap_fade_fraction: float = 0.0
    # Whether this system runs in the paper's reduced (LJ) units (sigma = eps =
    # m = 1) rather than the default LAMMPS "metal" units. When True the panel
    # readouts and plot axes drop the Kelvin/bar/eV/m-s labels (which would be
    # physically meaningless here) and show the dimensionless reduced-unit
    # quantities instead. Only the two MesoMem 3D systems set this.
    reduced_units: bool = False
    # Whether this system is driven by Play / Pause / Reset buttons (drawn at the
    # bottom of the sim view) rather than a continuously-controlled puller atom.
    # When True the app steps the simulation only while "playing", offers a Reset
    # that rebuilds the system from a fresh state (see MDSystem.reset), and skips
    # the puller/joystick input entirely. The self-assembly system sets this;
    # every interactive (puller) system leaves it False. Space toggles play/pause
    # and R resets when set.
    playback_controls: bool = False
    # 3D only: the look of the rendered scene -- lighting, ambient occlusion,
    # contact shadows, outline, depth cue, depth of field, tonemap. See
    # lammps_live/render_style.py; the defaults are tuned on a dense bead box,
    # and a flatter or much smaller scene will want a few fields varied.
    render_style: RenderStyle = DEFAULT_STYLE
    # 3D only: a turntable camera (mouse-drag to orbit, wheel to dolly, C to
    # start/stop the automatic orbit), for a scene with nothing to steer. None ->
    # the fixed camera the scenario chose. A render_style.CameraOrbit.
    camera_orbit: object = None
    # WHAT THE INPUT DEVICE'S TWO AXES DRIVE: "force" (they push the puller) or
    # "torque" (they turn its director, and nothing pushes it). See
    # playground/spec.py's Control.drive, which is where a playground says so.
    #
    # It reaches the SystemSpec because two things above the system have to know,
    # and neither can ask the mode: the renderer draws the reaction as two straight
    # arrows (force) or as two rings in the plane each rotation happens in (torque)
    # accordingly and labels the header line to match, and the app's joystick loop
    # cancels part of the measured reaction from what it applies -- which is a
    # force-domain correction with no counterpart here (see App._tick).
    control_drive: str = "force"


class MDSystem(ABC):
    """A self-contained LAMMPS setup: a thermostatted crystal/fluid plus one
    interactively-controlled "puller" atom. Concrete systems own everything
    LAMMPS-specific (pair style, lattice, region layout); the app loop and
    renderer talk only to this interface.
    """

    spec: SystemSpec

    @abstractmethod
    def set_input_force(self, fx, fy):
        """Joystick/mouse-commanded force on the puller, in eV/Angstrom (0 at center)."""

    @abstractmethod
    def set_target_temp(self, T):
        """Thermostat target, in K."""

    @abstractmethod
    def set_puller_damping(self, gamma):
        """Puller's own velocity-proportional drag, in eV*ps/Angstrom^2."""

    @abstractmethod
    def step(self, n):
        """Advance the simulation by n timesteps."""

    def reset(self):
        """Optional: rebuild the system from a fresh initial state, keeping the
        current live parameters (temperature, force-field coefficients). Used by
        the Play/Pause/Reset playback systems (see SystemSpec.playback_controls)
        to restart a run -- e.g. re-randomizing the self-assembly box. No-op by
        default (interactive puller systems are never reset this way)."""

    @abstractmethod
    def get_puller_state(self):
        """Returns (pos, vel), each a 2-element array, or (None, None)."""

    @abstractmethod
    def get_puller_energy(self):
        """Returns (ke, pe) of the puller atom alone, in eV."""

    @abstractmethod
    def get_interaction_force(self):
        """Net force on the puller from the rest of the system, in eV/Angstrom."""

    @abstractmethod
    def get_thermo_state(self):
        """Returns (temp[K], press[bar], ke[eV], pe[eV], etotal[eV])."""

    @abstractmethod
    def get_sim_time(self):
        """Elapsed simulated (physical MD) time, in ps, since interactive
        control began -- i.e. since __init__'s silent settle run finished,
        not since the LAMMPS instance itself was created. This is real
        simulated time (nsteps * timestep), not wall-clock render time."""

    @abstractmethod
    def get_rdf(self):
        """Returns (r, g(r)) arrays or None if not yet warmed up."""

    @abstractmethod
    def get_all_positions(self):
        """Returns (ids N, positions Nx2, is_puller boolarray N, species).
        ids are LAMMPS atom ids, stable identities across frames -- needed
        (rather than array index) because LAMMPS is free to reorder its local
        atom arrays between steps (e.g. periodic spatial sorting), which array
        index alone can't survive (see ui/trail.py, which keys per-atom motion
        trails by id for exactly this reason).

        species is a per-atom int array (aligned with positions) indexing into
        the SystemSpec's species_colors/species_labels, or None for
        single-species systems (Cu, Ar) where every crystal atom is drawn the
        same. E.g. NaCl uses 0=cation, 1=anion; lipids use 0=head, 1=tail."""

    def get_bond_pairs(self):
        """Optional: explicit bonds to draw, as an (M, 2) int array of index
        pairs into the CURRENT get_all_positions ordering (so it must be called
        in the same frame, before stepping again). Used to draw molecular
        backbones, e.g. each lipid's head-tail-tail chain. None (default) means
        the system has no explicit bonds to draw."""
        return None

    def get_hbond_pairs(self):
        """Optional: hydrogen-bond-like pairs to draw in a distinct, lighter
        style than the solid molecular bonds of get_bond_pairs -- as an (M, 2)
        int array of index pairs into the CURRENT get_all_positions ordering.
        Used by the water model to show the transient hydrogen-bond network
        forming and breaking. None (default) means no such overlay."""
        return None

    def get_hud_lines(self):
        """Optional: a list of short strings drawn as a small live HUD in the
        simulation view (beneath the standard force/time readout), for
        per-system pedagogical state the fixed panel readouts don't cover --
        e.g. the water demo's phase and density-anomaly readout. Empty/None ->
        nothing drawn."""
        return None

    def get_potential_terms(self):
        """Optional: a live breakdown of the puller's interaction energy into
        the separate additive terms of the force field, for pedagogy -- returned
        as (title, [(label, value), ...], scale) or None. The renderer draws it
        as a compact signed-bar chart (each term's bar, plus their sum) so the
        additive structure of the potential is visible and updates live. Values
        are in the system's own energy units (reduced/LJ for MesoMem); scale is
        the bar half-range. None (default) -> nothing drawn."""
        return None

    def get_total_potential_terms(self):
        """Optional: like get_potential_terms, but summed over the WHOLE system
        rather than just the puller bead -- the total interaction energy in each
        additive term of the force field (e.g. the MesoMem tilt / splay /
        attraction terms across every bead pair). Returned in the same
        (title, [(label, value), ...], scale) form so the renderer can draw it as
        a second signed-bar panel beside the puller-bead breakdown. None (default)
        -> nothing drawn."""
        return None

    def get_pair_annotation(self):
        """Optional (3D): ONE pair of particles taken apart term by term, drawn as
        an annotated connector between them -- separation, director angle, and each
        additive term's energy and radial force as live numbers.

        A `playground.pair_probe.PairAnnotation`, or None (the default). This is
        the two-particle counterpart of get_potential_terms: the panels answer "how
        much of each term is in this bead's whole neighbourhood", which on a scene
        of exactly two beads is a question with a much sharper version -- what each
        term is worth at THIS separation and THIS angle, written between the two
        beads it is about.
        """
        return None

    def get_box_bounds_3d(self):
        """Optional (3D): the simulation box to outline in the scene, as
        (xlo, xhi, ylo, yhi, zlo, zhi) in world units, or None to draw no box.
        The renderer draws its twelve edges as depth-cued white lines (occluded by
        the beads) and hides the single edge nearest the camera so it doesn't
        streak across the front of the scene."""
        return None

    def set_extra_param(self, key, value):
        """Optional: update a live-tunable parameter beyond temperature/damping,
        identified by the `key` of one of spec.extra_sliders (e.g. the MesoMem
        systems' k_tilt / k_splay / eta). No-op for systems with no extra
        sliders. Called every frame with the slider's current value, so
        implementations should cheaply no-op when the value is unchanged."""

    def take_fault(self):
        """Optional: the last event that destroyed the simulation, once, then None.

        A `playground.faults.Fault` -- what happened, in a sentence and verbatim,
        and any parameter the recovery had to put back. Popped rather than read, so
        the app shows each one exactly once (App._handle_faults). None for a system
        that cannot be destroyed by its own parameters, which is the default."""
        return None

    def live_param_values(self):
        """Optional: the effective value of every live parameter, keyed by slider
        key. The app pushes sliders INTO the system every frame, so this is how a
        value the system chose for itself -- a clamp, or a rebuild falling back off
        a value LAMMPS refused -- gets back out to the slider."""
        return {}

    def get_bead_brightness(self):
        """Optional (3D): a per-bead albedo brightness multiplier aligned with
        get_positions_3d's ordering (1.0 = normal). Used to spotlight a tagged
        cluster -- e.g. the sheet's diffusion-tracer bead and its neighbours.
        None (default) -> every bead drawn at normal brightness."""
        return None

    def steer_orientation(self, rate, dt):
        """Optional: steer the puller's in-plane orientation. rate is a control
        signal in [-1, 1] (joystick yaw / twist axis, or Q/E in mouse mode); dt
        is the frame time in seconds. No-op for systems whose puller has no
        meaningful orientation (a lone atom); lipids integrate it into the
        control lipid's director angle."""

    def get_scene_fit_points(self):
        """Optional: an (N, 3) array of world-space points the 3D camera should
        frame (zoom to just fit). Used to fill the viewport at any aspect ratio
        instead of a fixed field of view. None (default) -> keep the camera's
        fixed FOV (non-3D systems, or 3D systems that don't want auto-fit)."""
        return None

    def get_torque_signals(self):
        """Optional: (applied, reaction) torques about the control-plane normal
        for the circular torque arrows, each already normalized to [-1, 1]
        (fraction of the display maximum, positive = same screen handedness as a
        positive yaw command). `applied` is the user's steering torque, `reaction`
        is the force field's restoring torque on the puller. Both are the
        component projected onto the 2D plane the scene depicts. None (default)
        -> no torque arrows (systems whose puller has no meaningful orientation)."""
        return None

    def get_torque_vectors(self):
        """Optional: (applied, reaction) torques as world 3-vectors -- the axes the
        two rotations are about, by the right-hand rule -- each normalized so 1.0 is
        its own display maximum. Drawn at the puller as two RINGS, each the circle
        of the rotation in the plane it happens in, on a system whose input is a
        TORQUE rather than a force (spec.control_drive). A force drive has real
        force vectors to draw there instead and returns None, which is also the
        default."""
        return None

    def get_box_periodic(self):
        """3D only: (x, y, z) periodicity of the cell. Default: not periodic."""
        return (False, False, False)

    def get_bead_energies(self):
        """3D only: per-particle potential energy, in the same order as
        get_positions_3d, for the energy colouring. None -> the renderer keeps the
        director banding."""
        return None

    def toggle_puller_attached(self):
        """Grab or release the puller: attached, the input device drives it and
        feels the force field push back; released, it is an ordinary particle of
        the simulation and the device holds nothing. Returns the new state.
        Default: no puller to grab, so always False."""
        return False

    def puller_attached(self):
        """Whether the input device is currently holding the puller."""
        return False

    def set_playing(self, playing):
        """Tell the system whether the run is going (Play/Pause).

        A no-op for a system this process integrates itself -- the app simply stops
        calling `step`. It matters for one whose simulation is elsewhere: that one
        has to say so, or the far end keeps integrating for a client that is not
        reading (see remote/client.py).
        """

    def puller_bead_count(self):
        """Number of LAMMPS atoms the puller is made of: 1 for a single-atom
        puller (Cu, Ar, NaCl), or the molecule's bead count for a
        molecular puller (lipid = 3, water = 4). set_input_force applies its
        force to each of these atoms, so a caller wanting a specific NET force on
        the puller (e.g. the app's joystick MD-force cancellation) divides by
        this count. Default 1."""
        return 1

    @abstractmethod
    def get_box_size(self):
        """Returns (width, height) of the simulation box, in Angstrom."""

    @abstractmethod
    def close(self):
        ...


class MDSystem3D(MDSystem):
    """A system the app renders as a 3D scene (SystemSpec.render_3d = True).

    These five methods were previously an undeclared, unenforced protocol: only a
    comment on SystemSpec.render_3d mentioned them, yet app._tick called all five
    unconditionally for any 3D system. Declaring them here means a 3D system that
    forgets one fails at construction with a clear message instead of raising
    AttributeError mid-frame.
    """

    @abstractmethod
    def get_positions_3d(self):
        """(ids, positions (N,3), is_puller (N,) bool) in a stable order."""

    @abstractmethod
    def get_dipoles_3d(self):
        """Per-particle unit orientation vectors (N,3), aligned with
        get_positions_3d's ordering. Zero rows for particles with no
        orientation."""

    @abstractmethod
    def get_bonds_3d(self):
        """Bond sticks to draw, as index pairs into get_positions_3d's ordering.
        Empty list for systems whose particles overlap into a continuous surface
        (bonds would be invisible clutter, and would not tile across a periodic
        seam anyway)."""

    def get_bead_tints(self):
        """Optional (3D): the playground's own per-bead colour, or None.

        (N, 4) aligned with get_positions_3d: display-space r, g, b in 0..255,
        and a MIX in 0..1 saying how much of it to paint over the director
        banding -- 0 leaves a bead banded, 1 makes it flat. That is what lets one
        scene hold two species that should not look alike: the polymer inside a
        vesicle is not a membrane bead and should not be painted like one, while
        the membrane keeps the banding that is the whole point of it.

        STATIC, and deliberately so. It is a fact about the composition, fixed
        when the system is built, which is what keeps it off the wire on a remote
        playground (the client builds the same scenario and knows it) and out of
        the per-frame gather. The colourings that MEASURE something -- the energy
        ramp, the cluster labelling -- are the ones that change every frame, and
        they are separate readouts for that reason.
        """
        return None

    def get_glyph_spheres(self):
        """Extra spheres to draw that are not particles, or None.

        (centers (K,3), radii (K,), directors (K,3), owners (K,)). They exist for
        a particle whose SHAPE is not the sphere LAMMPS integrates it as: a rigid
        rod is one particle with a length, and its body is drawn as a line of
        overlapping impostors with the particle itself inside them. They carry no
        state and take part in no physics; `owners` indexes the particle each
        sphere belongs to, which is all the renderer needs to paint it the same
        colour. Nothing else in the frame -- energies, brightness, the puller
        mask -- has to know they exist.
        """
        return None

    @abstractmethod
    def get_camera_params(self):
        """dict(eye=, target=, up=, fov_deg=) for the scene's perspective
        camera. The app then calls get_scene_fit_points to zoom to fit, so this
        only needs to set a good viewing angle and rough distance."""

    @abstractmethod
    def get_control_grid(self):
        """The interactive control plane to draw as a faint "net", as
        dict(origin=, u_axis=, v_axis=, u_range=, v_range=, step=), or None when
        there is nothing to steer. Its extents should be exactly the controlled
        particle's movement limits, so the net marks precisely where it can be
        dragged."""
