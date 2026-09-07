"""The ForceField interface: one object per force field, reused by every
scenario that runs it.

A ForceField owns three things, and nothing else:

  1. Its parameter declarations (see params.Param) -- the single source of the
     sliders the GUI shows.
  2. The LAMMPS commands that install it (`pair_style`, `pair_coeff`, masses,
     per-particle attributes) and re-install it when a live parameter moves.
  3. Optionally, a vectorized Python expression of its energy, decomposed into
     the additive terms it is built from.

That third item is the part that makes this more than a refactor. It exists once,
so it can (a) drive the live energy-breakdown panels and (b) be checked against
what LAMMPS actually computed (see verify.py). Previously the MesoMem energy
expression was hand-written five times across three system modules -- with
docstrings warning that the copies had to be kept in sync with the C++ by hand.

It is a DIAGNOSTIC, not the force loop. The forces that move particles always
come from the compiled pair style inside LAMMPS.
"""
from abc import ABC, abstractmethod

from .params import ParamSet

_REGISTRY = {}


def register(cls):
    """Class decorator adding a ForceField to the by-name registry."""
    _REGISTRY[cls.name] = cls
    return cls


def get(name):
    """Look up a force field class by name, importing the bundled ones lazily.

    Lazy so `--list-playgrounds` does not import scipy and the LAMMPS bindings
    just to print a table (the old eager systems registry imported all eight
    system modules, and therefore `lammps` + `scipy`, to answer
    `--list-systems`).
    """
    if name not in _REGISTRY:
        from .. import forcefields  # noqa: F401  -- import registers the bundled fields
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(f"Unknown force field {name!r}. Available: {known}") from None


def available():
    from .. import forcefields  # noqa: F401
    return dict(_REGISTRY)


class ForceField(ABC):
    """Base class for a force field.

    Subclasses set the class attributes and implement `pair_commands`. Everything
    else is optional.
    """

    name = ""                    # registry key, used in playground files
    units = "lj"                 # LAMMPS units command
    dimension = 3
    atom_style = "atomic"
    n_types = 1
    # Bonded topology, for a force field whose particles are chained rather than
    # loose. Both counts and the per-atom allowances below become `create_box`
    # keywords (see box_keywords) -- they have to be declared THERE, before any
    # atom exists, because LAMMPS sizes its per-atom bond and special-neighbour
    # arrays once and cannot grow them later.
    n_bond_types = 0
    n_angle_types = 0
    # Extra `create_box` keywords, verbatim -- the `extra/*/per/atom` allowances a
    # bonded force field needs. Left as strings rather than derived, because how
    # many bonds an atom can carry is a fact about the topology the SCENARIO will
    # build, and a force field that declares bonds at all is declaring the shape
    # of that topology with them.
    box_extras = ()
    # Set when the style is a custom C++ pair style needing compilation (see
    # plugin.PluginSpec); None for stock LAMMPS styles.
    plugin = None
    # Declared parameters, in the order their sliders should appear.
    params = ()
    # Display labels for the additive energy terms, in the order energy_terms
    # returns them. Empty -> this force field offers no decomposition.
    energy_terms_labels = ()
    # WHAT EACH OF THOSE TERMS IS A FUNCTION OF, as short symbolic text in the
    # same order -- "r", "n.rhat", and so on. Drawn beside each term's name in the
    # two-bead callout (see Renderer._draw_pair_callout), which is the one place
    # the additive structure is the subject rather than the instrument.
    #
    # IT IS PHYSICS, NOT DECORATION, and that is why it is declared here next to
    # the labels rather than typed into the renderer. The single most useful thing
    # to know about this force field is that one of its terms depends on nothing
    # but the separation while the others depend on the orientations -- which is
    # exactly why the van der Waals torque column sits at zero while the tilt row
    # swings, and a reader who can see "(r)" written next to that zero does not
    # have to be told. Empty -> nothing is drawn, and the names stand alone.
    energy_terms_arguments = ()
    # Whether particles carry an orientation (LAMMPS `mu`). Drives whether
    # FrameState.directors is populated and whether director-based observables
    # are offered.
    has_directors = False
    # Whether the pair style implements single(), i.e. whether LAMMPS can report
    # the force between two groups directly with `compute group/group`. False
    # forces the caller to recover it as "total force minus the forces we applied
    # ourselves", which is exact but fragile -- see modes.GameMode.interaction_force.
    # MesoMem's pair style has no single(); lj/cut and eam do.
    supports_single = False
    # Bar half-range for the energy panels, per particle involved. Scaled by the
    # relevant particle count so the same force field reads sensibly on a 7-bead
    # patch and a 1500-bead box.
    energy_scale_per_particle = 1.0

    def new_params(self, overrides=None):
        return ParamSet.build(self.params, overrides)

    def ensure_available(self, lmp):
        """Load any custom pair-style plugin into this LAMMPS instance."""
        if self.plugin is not None:
            from .plugin import ensure_loaded
            ensure_loaded(self.plugin, lmp)

    def setup_commands(self, params):
        """Per-particle setup issued after the atoms exist: masses, diameters,
        anything the pair style needs on each particle. Returns a list of LAMMPS
        command strings."""
        return []

    def integrator_command(self):
        """The time integrator. Orientation-carrying particles need their
        rotational degrees of freedom integrated too, which is a property of the
        force field (it is what makes the directors move), not of the scenario."""
        if self.has_directors:
            return "fix integrate all nve/sphere update dipole"
        return "fix integrate all nve"

    def box_keywords(self):
        """Extra keywords for `create_box`, as one string starting with a space
        (or empty). See n_bond_types."""
        parts = []
        if self.n_bond_types:
            parts.append(f"bond/types {self.n_bond_types}")
        if self.n_angle_types:
            parts.append(f"angle/types {self.n_angle_types}")
        parts += list(self.box_extras)
        return (" " + " ".join(parts)) if parts else ""

    def bonded_commands(self, params):
        """The bond and angle styles and their coefficients, as command strings.

        Separate from `pair_commands` because they are a separate installation
        with a separate lifetime: the pair style is re-issued whenever a cutoff
        moves, while a bond style is declared once and only its coefficients ever
        change. Empty for a force field with no topology, which is most of them.
        """
        return []

    @abstractmethod
    def pair_commands(self, params):
        """The full pair-style installation, as a list of command strings --
        `pair_style ...` followed by its `pair_coeff` line(s)."""

    def live_commands(self, params, changed_name):
        """Commands to re-apply the force field after the live parameter
        `changed_name` moved.

        The default handles the two live tiers generically: a HOT parameter
        re-issues the coefficients, a HOT_RESTYLE parameter re-issues the whole
        pair style first (because it moved the global cutoff, and thus the
        neighbour list). LAMMPS overwrites the stored per-type coefficients in
        place and re-inits the pair style on the next run, so this is safe
        between steps.
        """
        from .params import Tier
        tier = params.tier_of(changed_name)
        if tier is Tier.HOT_RESTYLE:
            return self.pair_commands(params)
        return self.coeff_commands(params)

    def coeff_commands(self, params):
        """Just the coefficient line(s) -- by default the pair_commands minus the
        leading `pair_style`."""
        return [c for c in self.pair_commands(params)
                if not c.startswith("pair_style")]

    def interaction_cutoff(self, params):
        """The largest separation at which this force field does anything, used
        to build the pair list for the energy decomposition and observables."""
        return 0.0

    def pair_landmarks(self, params):
        """The separations where this force field's behaviour CHANGES -- a term
        switching on, a branch of one taking over -- as
        ((label, radius, term_index), ...) in increasing radius.

        Drawn by the two-bead scene as rings around the fixed partner, coloured by
        the term each landmark belongs to (`term_index` indexes
        energy_terms_labels), so the shells the driven bead crosses are visible
        before it crosses them. They are read off the LIVE parameters, so dragging
        the rc or wc slider moves the ring it names.

        Empty (the default) -> a force field with nothing to point at, or one
        nobody has asked to be explained this way. See pair_probe.py.
        """
        return ()

    def extended_pairs(self, state, pairs, params):
        """Extra pairs to append to the analysis list, or None.

        For a force field where ONE species reaches much further than the rest.
        Widening `interaction_cutoff` to cover it would be correct and very
        expensive: it is a global cutoff, so every ordinary pair in the system
        gets found at the long range too. On the rod playground that is the
        difference between 55k pairs and 335k -- a 37 ms lump on the frame the
        energy panels land on, to find the hundred pairs one rod is having.

        So the long-ranged species names its own pairs instead, which for a single
        particle is one small query rather than a whole-system tree. Indices are
        into `state`, which is the same (possibly subsampled) state the rest of
        `pairs` was built from.
        """
        return None

    @property
    def thermo_is_per_atom(self):
        """Whether LAMMPS normalizes this unit style's thermodynamic output per
        atom. True for `lj` (`thermo_modify norm yes` is the default there),
        false for `metal` and the other physical unit styles -- which the verifier
        must undo, or a correct force field looks off by a factor of N."""
        return self.units == "lj"

    def glyph_spheres(self, state, params):
        """Extra spheres to DRAW, for a force field whose particles are not the
        shape LAMMPS integrates them as.

        Returns (centers (K, 3), radii (K,), directors (K, 3), owners (K,)) or
        None; `owners` names the particle each sphere belongs to, so the renderer
        can paint it that particle's colour. They carry no state and take part in
        no physics -- they exist because a rigid rod is one particle with a
        length, and drawing it as a single sphere hides the only thing the user is
        steering. The renderer appends them to the bead instances, so the real
        particle is still there, inside its own body.
        """
        return None

    def energy_terms(self, state, pairs, params):
        """Per-pair energy in each additive term of the potential.

        Returns a dict {label: (M,) array} aligned with `pairs`, or None if this
        force field offers no decomposition. Per-PAIR (rather than pre-summed) so
        one evaluation serves both energy panels: the whole-system total sums
        everything, and a single particle's share sums the pairs touching it.
        """
        return None
