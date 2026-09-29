"""Live-tunable parameter declarations.

The one rule that decides what belongs in a playground file and what belongs on
a GUI slider: **does changing this require rebuilding the LAMMPS instance?**

    STRUCTURAL   bead count, box size, boundary, atom style, which pair style.
                 File-time only. A slider here would tear down and rebuild the
                 simulation on every drag frame, throwing away the state the
                 user was watching.

    HOT          pair_coeff arguments, thermostat target. Re-applied in place
                 between steps; a slider is generated automatically from the
                 declared range/default/optimum.

    HOT_RESTYLE  a HOT parameter that also moves the pair style's GLOBAL cutoff
                 (MesoMem's rc), so `pair_style` must be re-declared before the
                 coefficients. Declared, not special-cased -- the old code had
                 `if key == "rc"` copy-pasted into three systems.

A Param is declared ONCE and the slider follows from it. Previously every
tunable was written down twice: as a SliderSpec in the SystemSpec and again as a
string key in a hand-written `set_extra_param` dispatch dict, in each of three
files.
"""
from dataclasses import dataclass, field
from enum import Enum


class Tier(Enum):
    STRUCTURAL = "structural"
    HOT = "hot"
    HOT_RESTYLE = "hot_restyle"

    @property
    def is_live(self):
        return self is not Tier.STRUCTURAL


@dataclass(frozen=True)
class Param:
    """One named quantity of a force field or scenario.

    `clamp` expresses a dependency on other parameters: it is called as
    clamp(value, values_dict) and returns the value actually used. MesoMem's
    orientational cutoff declares `clamp=lambda v, p: min(v, p["rc"])`, which
    replaces the `_effective_wc()` helper that was hand-written identically in
    three system modules -- and, more importantly, means the clamp is applied
    everywhere the parameter is read (the pair_coeff line AND the Python energy
    decomposition) instead of only where someone remembered to call it.
    """
    name: str
    default: float
    label: str = ""
    vmin: float = 0.0
    vmax: float = 1.0
    tier: Tier = Tier.HOT
    fmt: str = "{:.3f}"
    unit: str = ""
    optimum: float = None
    advanced: bool = False
    clamp: object = None       # callable(value, values) -> value, or None
    doc: str = ""

    def slider_spec(self):
        """The UI-facing SliderSpec for this parameter. Only meaningful for live
        tiers; structural parameters never reach the panel.

        Imported here rather than at module scope so `playground` stays free of
        the systems package (and therefore of LAMMPS and scipy) until something
        actually needs a UI object -- which is what lets `--list-playgrounds`
        stay a cheap import.
        """
        from ..mdsystem import SliderSpec
        return SliderSpec(
            label=self.label or self.name,
            vmin=self.vmin, vmax=self.vmax, default=self.default,
            fmt=self.fmt, unit=self.unit, optimum=self.optimum,
            key=self.name, advanced=self.advanced,
        )


def structural(name, default, doc=""):
    """A build-time-only parameter: geometry, counts, box size, seeds."""
    return Param(name=name, default=default, tier=Tier.STRUCTURAL, doc=doc)


@dataclass
class ParamSet:
    """Live values for a set of Params, with change detection and clamping.

    The app pushes every slider's value every frame (by design -- see
    MDSystem.set_extra_param's contract), so `set` returning False for an
    unchanged value is what keeps that cheap.
    """
    params: tuple
    values: dict = field(default_factory=dict)

    def __post_init__(self):
        self._by_name = {p.name: p for p in self.params}
        for p in self.params:
            self.values.setdefault(p.name, p.default)

    @classmethod
    def build(cls, params, overrides=None):
        """Fresh value set, with `overrides` (a playground's preset or explicit
        parameter overrides) applied on top of the declared defaults. Unknown
        names raise rather than being silently ignored -- a typo in a preset
        would otherwise look like the preset simply having no effect."""
        ps = cls(tuple(params))
        for name, value in (overrides or {}).items():
            if name not in ps._by_name:
                known = ", ".join(sorted(ps._by_name))
                raise KeyError(
                    f"unknown parameter {name!r}. This force field / scenario "
                    f"declares: {known}"
                )
            ps.values[name] = value
        return ps

    def spec(self, name):
        return self._by_name[name]

    def has(self, name):
        return name in self._by_name

    def set(self, name, value):
        """Record a new raw value. Returns True if it actually changed (so the
        caller knows whether to re-issue any LAMMPS commands)."""
        if name not in self._by_name:
            return False
        if self.values.get(name) == value:
            return False
        self.values[name] = value
        return True

    def raw(self, name):
        return self.values[name]

    def __getitem__(self, name):
        """The EFFECTIVE value: the raw value with its declared clamp applied.

        Clamps see the raw values of their dependencies, which keeps resolution
        order irrelevant (no clamp may depend on another clamp's output).
        """
        p = self._by_name[name]
        v = self.values[name]
        return p.clamp(v, self.values) if p.clamp is not None else v

    def as_dict(self):
        """All effective values -- what a force field's command emitters and its
        energy decomposition both read, so they cannot disagree."""
        return {name: self[name] for name in self._by_name}

    def live_params(self):
        return tuple(p for p in self.params if p.tier.is_live)

    def slider_specs(self, range_overrides=None):
        """SliderSpecs for every live parameter, in declaration order.

        `range_overrides` maps a parameter name to (vmin, vmax). A slider's span
        is a property of what a particular playground wants to explore, not of the
        force field: a splay modulus useful up to 3 on a 7-bead patch is worth
        dialling to 40 on a large sheet, where it buckles the membrane rather than
        just stiffening it. That was the one constant the three old MesoMem modules
        genuinely disagreed on.
        """
        import dataclasses
        specs = []
        for p in self.live_params():
            lo, hi = (range_overrides or {}).get(p.name, (p.vmin, p.vmax))
            # THE SLIDER STARTS AT THIS SET'S VALUE, not the declaration's: a
            # playground's own override or a preset is what the system is built
            # with, and the app pushes every slider into the system every frame
            # -- so a slider left at the declared default silently undid the
            # override on the first frame.
            specs.append(dataclasses.replace(
                p, vmin=lo, vmax=hi, default=self.values[p.name]).slider_spec())
        return tuple(specs)

    def tier_of(self, name):
        return self._by_name[name].tier
