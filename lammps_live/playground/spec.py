"""The declarations a playground file writes.

A playground file is a Python module with one module-level `PLAYGROUND = ...`
constant -- no class to subclass, no methods to implement. That is deliberately
the same shape as the old `SPEC = SystemSpec(...)` pattern, which worked well and
which both people and language models get right first time; the difference is
that it now names a force field, a scenario and a mode instead of being the
metadata header of an 800-line class.

    PLAYGROUND = Playground(
        name="MesoMem membrane patch",
        force_field="mesomem",
        scenario=hex_patch(n_rings=1),
        mode="game",
        control=Control(atom="first", plane="xz", leash=(2.8, 2.8)),
        observables=["nematic_S", "thickness"],
        presets={"paper": {}, "floppy": {"k_tilt": 2.0, "k_splay": 0.1}},
    )
"""
from dataclasses import dataclass, field

from ..render_style import DEFAULT_STYLE, RenderStyle
from ..mdsystem import ForceFeedbackProfile


# Force-feedback tuning scaled to reduced-unit mesoscale forces, which run
# O(1-10) on a pulled particle -- with enough stick authority to tent a membrane
# and pop a particle out against tilt/splay resistance. A playground may override
# it; most should not need to.
REDUCED_UNIT_FEEDBACK = ForceFeedbackProfile(
    ff_exaggeration=1.3,
    ff_knee=4.0,
    ff_max_mag=120.0,
    stiffness_threshold=0.3,
    stiffness_knee=2.5,
    damper_min_fraction=0.10,
    damper_max_fraction=0.55,
    vel_damp_max_fraction=0.5,
)


@dataclass(frozen=True)
class Control:
    """How the user drives the controlled particle, in game mode.

    `leash` is the half-extent of the rectangle the particle can be dragged in,
    on the control plane's two axes -- and it is what the scene draws as a net, so
    it is a statement about the interaction, not a rendering detail. Keep it
    inside the force field's interaction range: a particle dragged past the cutoff
    detaches and floats free instead of being pulled back, and the snap-back on
    release is the whole point.

    `drive` is WHAT the two input axes are: a force that moves the particle, or a
    torque that turns its director. See the field below -- the two are alternatives
    on the same stick, not a mode switch on top of one another.
    """
    atom: str = "nearest_center"
    plane: str = "xz"
    # WHAT THE STICK'S TWO AXES DO.
    #
    #   "force"   they push the particle, in the control plane. The particle is
    #             normally held on that plane inside the leash (see `confine`),
    #             its director is steered separately by the twist axis, and the
    #             thing rendered back to the hand is the force field's reaction
    #             FORCE. This is the shipped default and what every playground
    #             that predates the choice does.
    #   "torque"  they turn the particle's director, about the two axes named in
    #             `torque_axes`, and no force is applied to it at all. The
    #             particle then moves only under the force field -- it is an
    #             ordinary particle of the simulation that you happen to be able
    #             to twist -- and what is rendered back to the hand is the
    #             reaction TORQUE. The twist axis is unused, because two axes
    #             already cover both of a director's degrees of freedom and a
    #             director has no third (spinning it about itself is the identity).
    #
    # Everything downstream is the same pipeline in the other domain: the input
    # ceiling (`max_input_torque`), the reaction signal, the force-feedback
    # shaping, the joystick's partial cancellation of the measured reaction. The
    # only thing that changes shape is the drawing, which swaps the two straight
    # arrows for the two circular arcs it already had (see the renderer).
    drive: str = "force"
    # Which world axes the two input axes torque about, when `drive` is "torque".
    # Signed axis names: "y" and "-x" mean +y and -x.
    #
    # THE DEFAULT IS THE TRACKBALL MAPPING for the scenes here -- an xz control
    # plane, a camera on -y, +z up on screen -- and it is worth writing out, since
    # the intuition and the arithmetic disagree about the sign. Think of the bead
    # as a ball under your palm and the director as a pin standing out of it:
    #
    #   stick right  the top of the ball goes right, so the pin tips toward +x.
    #                d(n)/dt = w x n, and (+y) x (+z) = +x, so w is +y.
    #   stick fwd    the top goes away from you, toward +y (into the screen), so
    #                the pin tips that way. (-x) x (+z) = +y, so w is -x.
    #
    # A scenario framed from a different angle sets its own pair. The rule for
    # picking one: name the two axes PERPENDICULAR to where the directors start,
    # never the one they lie along -- a torque about the director itself does
    # nothing, so that pair would leave one input axis dead at rest.
    torque_axes: tuple = ("y", "-x")
    # The peak torque at full deflection of any input device, for `drive` of
    # "torque" -- the rotational counterpart of `max_input_force`, and the one
    # honest "how hard can I twist" number. Both are here rather than one shared
    # field because they are in different units and a playground that switched
    # between them would silently reuse the wrong magnitude.
    #
    # It shares `yaw_torque`'s default because it is the same kind of number: the
    # angular-momentum kick a full deflection is worth, per frame. A twist that
    # flips a director over the tilt term's barrier at 1.0 on the twist axis flips
    # it at 1.0 here too.
    max_input_torque: float = 1.0
    leash: tuple = (3.0, 3.0)
    # WHERE THAT RECTANGLE IS CENTRED, on the same two axes. The default is the
    # world origin, which is right for every scenario built symmetrically about it
    # -- the patch, the sheet, the rod.
    #
    # It exists for a scene whose SUBJECT is off-centre. The two-bead pair spans
    # x = 0 to x = partner_x, so a leash centred on the origin reaches two sigma
    # further left than right of what there is to look at, and the net drawn at its
    # limits is visibly off to one side of the pair it belongs to. Centring both on
    # the pair's midpoint fixes the picture and the travel together -- which is why
    # this is one field and not a rendering offset: the net IS the leash made
    # visible (see u_range / v_range below, which both read), and a net drawn
    # somewhere the particle cannot go, or stopping short of somewhere it can, is
    # the one thing it must never be.
    leash_center: tuple = (0.0, 0.0)
    # How much of each leash half-extent the REPORTED force fades out over as the
    # particle approaches that boundary, as a fraction. The leash is the app's
    # constraint, not the model's: nothing in the force field says there is a wall
    # there, so holding the particle against one and rendering the resulting load
    # on the stick makes the user push against a wall that does not exist -- and
    # it is a sustained push, since the particle cannot move away from it. Fading
    # the force out over the last 20% instead means the resistance melts away as
    # you reach the limit, the particle simply stops, and nothing pushes back.
    # The interesting physics is untouched: on the patch the membrane's pull peaks
    # around a third of the way out and is well past its peak by here.
    leash_release: float = 0.20
    speed_cap: float = 6.0
    # Whether to hold the particle on the control plane inside the leash at all.
    # False gives a genuinely free particle -- the deposition setups, where the
    # atom is meant to fly in, stick, and be knocked loose again, and where the
    # simulation is 2D so there is no out-of-plane axis to pin. A free particle
    # draws no net, because there is no boundary to draw.
    confine: bool = True
    # Maximum displacement per timestep for the controlled particle, via
    # `nve/limit`. This is how an unconfined particle is kept stable through a
    # hard contact impact instead of tunnelling through the lattice. None -> the
    # global integrator handles it.
    displacement_cap: float = None
    max_input_force: float = 9.0
    damping_default: float = 4.0
    damping_range: tuple = (0.0, 8.0)
    # Yaw steering: an angular-momentum kick about the plane normal per unit yaw
    # per frame, plus the per-frame rotational-velocity retention. Strong enough
    # that a firm twist drives a director over the tilt term's barrier at 45 deg
    # and flips it to the opposite normal -- the term is bistable, both +n and -n
    # are minima, and that flip is the pedagogical point. A gentle twist just
    # deflects and springs back.
    yaw_torque: float = 1.0
    rot_damp: float = 0.88
    # Reaction torque that fills the display arc. The tilt term reaches
    # O(k_tilt/2) at large deflection; this is picked so a firm twist against it
    # fills the arc.
    reaction_torque_max: float = 6.0
    grid_step: float = 0.5

    @property
    def drives_torque(self):
        return self.drive == "torque"

    @property
    def input_scale(self):
        """The ceiling a unit of input deflection is multiplied by, whichever
        domain this control drives in. One number, so the app, the HUD and the
        arrows all read the same one and nothing has to ask which drive it is."""
        return self.max_input_torque if self.drives_torque else self.max_input_force

    @property
    def u_range(self):
        c, h = self.leash_center[0], self.leash[0]
        return (c - h, c + h)

    @property
    def v_range(self):
        c, h = self.leash_center[1], self.leash[1]
        return (c - h, c + h)


@dataclass(frozen=True)
class Thesis:
    """A one-click A/B on the force field's own central claim.

    THE CLAIM this whole demo makes is that a membrane's shape comes out of beads
    that care which way they point -- and the fastest way to teach a claim is to
    take it away. Zeroing the two orientational moduli leaves the SAME beads with
    the SAME isotropic attraction, and what was a membrane collapses into a
    droplet. One button, one second, and the thesis is proven rather than asserted.

    It is a toggle rather than a preset because the point is the COMPARISON. A
    preset (`isotropic_only`, which every membrane playground declares and which
    this zeroes the same parameters as) is a way to start somewhere; this is a way
    to go there and come back while the audience watches the same beads, which is
    a different pedagogical act. Nothing is hidden while it is engaged: the app
    drives the real sliders to zero and puts them back, so the panel always says
    what the physics is (see App._toggle_thesis).

    `params` are zeroed in order and restored to whatever they were. The default
    set is MesoMem's two orientational moduli plus their cutoff -- exactly the
    `isotropic_only` preset, because "what isotropic-only means" should have one
    definition and not two.
    """
    # On the button, released and engaged. The engaged label is the way OUT, which
    # is what a lit button should say.
    label: str = "Remove orientation"
    engaged_label: str = "Restore orientation"
    # Live force-field parameters driven to zero while engaged.
    params: tuple = ("k_tilt", "k_splay", "wc")
    # One line, drawn under the button while it is engaged, saying what is now on
    # screen. Without it the audience sees a collapse and has to be told what was
    # taken away; with it the screen says so, which is the difference between a
    # demo and a lesson.
    caption: str = ("no orientation: the same beads, the same attraction, "
                    "and no membrane")


@dataclass(frozen=True)
class Lesson:
    """What a playground TEACHES -- as opposed to what it simulates.

    Every field here is audience-facing text or a decision about how much of the
    instrument to show, and it is declared next to the physics for the same reason
    the render style is: a scene and the thing it is trying to say are one design,
    and splitting them across two files is how they drift apart.

    THE THREE LINES are a fixed template, drawn in the same place on every
    playground so the audience learns where to look once (see
    Renderer._draw_lesson_card):

        title        the LESSON, not the geometry. "Twist", not "6+1 beads,
                     torque-driven". The geometry is already in `Playground.name`,
                     which the panel still shows; this is the word the talk uses.
        claim        one sentence of physics, and the one thing to remember. Keep
                     it under about twelve words: it is read from across a room,
                     off a projector, in the two seconds before the presenter
                     starts talking over it.
        instruction  what to do with your hands, in the imperative. A scene nobody
                     knows how to touch teaches nothing, and this is the only line
                     that is about the app rather than the physics.

    THE HOOK is the question this stage leaves open, and it is what turns eight
    scenes into one argument: the sheet ends by asking whether every bead had to be
    placed on a lattice, and the assembly box answers it. It is drawn in the PANEL
    rather than in the scene, because it is a note to whoever is driving about
    where to go next -- the audience gets it spoken, not written.

    THE ACT is which third of the argument this belongs to (see registry._ACTS).
    It is what makes the position indicator say "where am I in the story" rather
    than only "how much is left".
    """
    title: str
    claim: str
    instruction: str
    hook: str = ""
    # WHICH LIVE PARAMETERS ARE EVERYDAY HERE, by name -- the rest drop behind the
    # panel's collapsible "Advanced" group. None leaves every playground's dials
    # exactly as the force field declared them.
    #
    # This is progressive disclosure, and the rule it follows is: a control appears
    # at the stage where the thing it changes becomes VISIBLE, and not before. A
    # k_splay slider on a two-bead scene is a dial whose effect cannot be seen,
    # which teaches that the model is arbitrary. An empty tuple means no everyday
    # dials at all -- the two-bead scene, where the subject is one interaction and
    # every dial is a distraction from it.
    #
    # HIDDEN, NOT REMOVED, and that distinction is the whole reason this is the
    # `advanced` flag rather than a filter: the presenter who wants k_tilt on the
    # opening slide is one click from it, and nothing about the force field has
    # been quietly redefined.
    everyday_params: tuple = None
    # Whether the panel's four stacked plots (temperature, pressure, energy, g(r))
    # are drawn at all.
    #
    # OFF EARLY, and not to reduce clutter -- because they are not true yet. A
    # time series is a statement about an ensemble, and on two beads or seven the
    # plots show thermostat noise on a sample too small to have a temperature;
    # g(r) of a single pair is one spike. They arrive at the sheet, which is the
    # first scene big enough for a statistic to mean anything, and that arrival is
    # itself worth noticing.
    plots: bool = True
    # The one-click A/B on this scene's central claim (a Thesis), or None.
    thesis: object = None


@dataclass(frozen=True)
class Playground:
    """One explorable setup: a force field on a scenario, driven in a mode."""

    name: str
    force_field: str
    scenario: object
    description: str = ""
    # Constructor keyword arguments for the force field -- non-parameter choices
    # like the sphere diameter that sets the rotational inertia. Distinct from
    # `params`, which are the declared tunables.
    force_field_options: dict = field(default_factory=dict)
    key: str = ""                      # CLI id; defaults to the module basename
    mode: str = "game"                 # "game" or "sim"
    control: Control = None
    # Live force-field parameter overrides applied on top of its declared
    # defaults. For a different STRUCTURAL value, configure the scenario instead
    # -- that is the file/GUI boundary.
    params: dict = field(default_factory=dict)
    # Per-playground slider spans, {name: (vmin, vmax)}. Use when this setup wants
    # to explore a wider or narrower range than the force field's default span.
    param_ranges: dict = field(default_factory=dict)
    # Named parameter sets, selectable with --preset. A preset is the unit of
    # "the setting I found interesting", and is what makes a demo reproducible.
    presets: dict = field(default_factory=dict)
    observables: tuple = ()
    # WHAT THIS PLAYGROUND TEACHES: a Lesson (above) -- the title, claim and
    # instruction drawn over the scene, the hook to the next stage, and how much of
    # the instrument to expose here. None means a playground that is not part of
    # the taught sequence: the shelved atomistic classics, and anyone's own file.
    # Everything downstream treats it as optional, so a scene without one simply
    # draws no card and shows every dial (see registry.lesson_position).
    lesson: object = None
    # Draw the force field's additive terms as an annotated connector BETWEEN the
    # first two particles: the separation, the director angle, and each term's
    # energy and radial force as live numbers, with the force field's landmark
    # radii as rings around the second one. See playground/pair_probe.py.
    #
    # The first two particles, on the same argument as the `pair_separation` /
    # `pair_director_angle` observables: it is meaningless on a membrane, where a
    # bead has a dozen neighbours and no distinguished one, and it is the whole
    # readout on the two-bead scene, where the pair IS the system. Declaring it
    # anywhere else annotates an arbitrary pair, which is why it is off by default.
    pair_annotation: bool = False
    # Temperature dial, in the force field's own units.
    temperature: tuple = (0.0, 0.5)    # (min, max)
    temperature_default: float = 0.001
    melt_temp: float = 0.3
    force_feedback: ForceFeedbackProfile = REDUCED_UNIT_FEEDBACK
    # Rendered particle radius, in world units.
    particle_radius: float = 0.5
    reduced_units: bool = True
    seed: int = None                   # None -> a fresh random seed each build

    # --- rendering ------------------------------------------------------------
    # Whether to draw as a 3D scene (perspective spheres and directors) or the
    # top-down 2D box. The membrane playgrounds are 3D; a 2D crystal slab is not.
    render_3d: bool = True
    # How the 3D scene is lit and post-processed -- a RenderStyle (see
    # lammps_live/render_style.py). The default is the showreel look, tuned on a
    # dense bead box; `DEFAULT_STYLE.varied(dof_bokeh_px=4.0, ...)` adjusts it
    # for a scene of a different shape.
    render_style: RenderStyle = DEFAULT_STYLE
    # A turntable camera instead of the scenario's fixed one: mouse-drag to
    # orbit, wheel to dolly, C to start/stop the automatic orbit. Worth it for a
    # scene that is watched rather than driven -- the self-assembly box, where
    # what is forming is a 3D morphology and one fixed angle hides it. A
    # render_style.CameraOrbit, or None.
    camera_orbit: object = None
    # Offer the "Smoothing" slider (an advanced control): a visual-only temporal
    # low-pass on the drawn bead positions and directors, so thermal jitter
    # averages out of the picture and the slow rearrangements underneath it are
    # what the eye follows. See playground/smoothing.py. Worth it on a scene whose
    # interest IS a slow collective change (assembly, a healing membrane);
    # pointless on a handful of beads whose individual motion is the subject.
    trajectory_smoothing: bool = False
    # Bar half-range for the CONTROLLED particle's energy panel. None -> twice the
    # force field's per-particle scale, which is right when the controlled
    # particle is a typical one: it shares each pair with one neighbour, so its
    # own share of the energy is of a bead's order. Set it where the controlled
    # particle is NOT typical -- the rod, which is one particle in contact with a
    # hundred beads at once and whose adhesion energy is two orders above any
    # bead's. Without it that panel is simply pinned at full deflection.
    pulled_energy_scale: float = None
    # Cadence of the energy-panel evaluation, in analysis frames (default 4, see
    # observables.Analysis). The one dial that matters on a big system: the panels
    # are a pass over every pair, measured at 2.5 us/bead, so a playground running
    # 10k beads instead of 1500 buys its frame budget back here.
    analysis_energy_every: int = None
    # Set to run the simulation somewhere else -- a cluster GPU -- and only draw it
    # here. A remote playground builds no local LAMMPS: `registry.build` hands back
    # a remote.client.RemoteSystem, which takes its frames off a socket and runs
    # this same file's analysis on them. Everything else on this declaration
    # (sliders, observables, render style, camera) is used exactly as it is, by
    # both ends -- the server builds the very same Playground to decide what to
    # integrate. See remote/session.py for the connection it describes.
    remote: object = None

    element_label: str = ""            # legend text, e.g. "Ar (LJ)"
    lattice_spacing: float = 1.0       # informational, and the bond-overlay optimum
    # Flat draw colour for a single-species 2D system, so different materials read
    # as different materials rather than sharing one generic crystal colour.
    crystal_color: tuple = None
    species_colors: tuple = None
    species_labels: tuple = None
    species_radii: tuple = None
    # The generic "faint line between atoms near their equilibrium spacing"
    # overlay. Useful for a crystal; meaningless for overlapping membrane beads.
    bond_overlay: bool = False
    # Force-feedback speed scale, in the scenario's length units per time unit.
    # None -> derived from the timestep.
    puller_speed_cap: float = None

    def resolved_params(self, preset=None):
        """Live parameter overrides: the playground's own, then the named preset
        on top."""
        merged = dict(self.params)
        if preset:
            if preset not in self.presets:
                known = ", ".join(sorted(self.presets)) or "(none defined)"
                raise KeyError(
                    f"{self.key or self.name}: unknown preset {preset!r}. "
                    f"Defined: {known}"
                )
            merged.update(self.presets[preset])
        return merged

    def effective_control(self):
        return self.control or Control()

    def is_everyday(self, name, declared_advanced):
        """Whether a live parameter belongs on the panel's everyday list here.

        The force field declares a default (`Param.advanced` -- rc and wc are
        advanced everywhere, because they are cutoffs rather than physics dials),
        and a lesson may narrow it further for a scene where a dial has nothing
        visible to do yet. Narrow only: a lesson can move a dial into the Advanced
        group, never drag one out of it, so `everyday_params` cannot accidentally
        promote a cutoff onto the opening slide of a talk.
        """
        if declared_advanced:
            return False
        if self.lesson is None or self.lesson.everyday_params is None:
            return True
        return name in self.lesson.everyday_params
