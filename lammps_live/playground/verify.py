"""Cross-check a force field's Python energy expression against LAMMPS.

This is the payoff of having ONE vectorized `energy_terms` instead of five
hand-maintained copies: the sum of its additive terms must equal the potential
energy LAMMPS computed from the compiled pair style. If your C++ and your paper
disagree, this is where it shows up -- as a number, at startup, instead of as a
wrong-looking simulation weeks later.

Run it with `--verify`. It is off by default and costs nothing when off.

A note on units, which is the one real trap here: for `units lj` LAMMPS
normalizes thermodynamic output PER ATOM by default (`thermo_modify norm yes`),
while `units metal` reports extensive totals. The comparison has to undo that, or
it looks like a factor-of-N bug in the force field.
"""
import numpy as np


class VerificationResult:
    def __init__(self, lammps_pe, python_pe, terms, natoms, tolerance):
        self.lammps_pe = lammps_pe
        self.python_pe = python_pe
        self.terms = terms
        self.natoms = natoms
        self.tolerance = tolerance

    @property
    def absolute_error(self):
        return abs(self.python_pe - self.lammps_pe)

    @property
    def relative_error(self):
        scale = max(abs(self.lammps_pe), 1e-12)
        return self.absolute_error / scale

    @property
    def ok(self):
        return self.relative_error <= self.tolerance

    def report(self, label=""):
        head = "OK  " if self.ok else "FAIL"
        lines = [
            f"[verify] {head} {label}",
            f"    LAMMPS  potential energy : {self.lammps_pe:+.9g}",
            f"    Python  sum of terms     : {self.python_pe:+.9g}",
            f"    relative error           : {self.relative_error:.3e} "
            f"(tolerance {self.tolerance:.0e})",
            f"    N particles              : {self.natoms}",
        ]
        for label_, value in self.terms:
            share = value / self.python_pe if self.python_pe else float("nan")
            lines.append(f"      {label_:<38s} {value:+.6g}  ({share:+.1%})")
        return "\n".join(lines)


def verify_system(system, tolerance=1e-6):
    """Compare a running PlaygroundSystem's Python energy terms to LAMMPS' pe.

    Forces a fresh evaluation rather than reading the throttled cache, so the two
    numbers describe the same configuration.
    """
    force_field = system.force_field
    if not force_field.energy_terms_labels:
        return None

    # Re-evaluate at the CURRENT configuration before reading pe. In game mode the
    # per-frame constraint adjusts the controlled particle's director after the
    # step, so LAMMPS' stored energy describes the pre-constraint state; comparing
    # against it charges the force field for a discrepancy that is really just two
    # different configurations. `run 0` recomputes forces and energy without
    # integrating.
    system.lmp.command("run 0")
    state = system.frame_state()
    # The same list the live panels use -- including any pairs a long-ranged
    # species names for itself. See observables.analysis_pairs.
    from .observables import analysis_pairs
    pairs = analysis_pairs(force_field, state, system.params)
    terms = force_field.energy_terms(state, pairs, system.params)
    if terms is None:
        return None

    summed = [(label, float(np.sum(terms[label])))
              for label in force_field.energy_terms_labels if label in terms]
    python_pe = sum(v for _, v in summed)

    natoms = len(state.positions)
    lammps_pe = system.lmp.get_thermo("pe")
    if force_field.thermo_is_per_atom:
        lammps_pe *= natoms

    return VerificationResult(lammps_pe, python_pe, summed, natoms, tolerance)


def verify_playground(ref, mode=None, preset=None, params=None, tolerance=1e-6,
                      steps=0):
    """Build a playground, optionally step it, and verify. Returns the result.

    Stepping first is worth doing: a pristine lattice can accidentally satisfy an
    expression that is wrong in general (every director exactly parallel makes the
    splay term vanish, for instance), so a thermalized, disordered configuration is
    the stronger test.
    """
    from .registry import build, load
    # A remote playground has no local LAMMPS to check the expression against --
    # its simulation is on a cluster. The force field it uses is verified through
    # whichever local playground uses it, which is the same expression.
    if load(ref).remote is not None:
        return None
    system = build(ref, mode=mode, preset=preset)
    try:
        for name, value in (params or {}).items():
            system.set_extra_param(name, value)
        if steps:
            system.step(steps)
        return verify_system(system, tolerance)
    finally:
        system.close()


def verify_all(refs, tolerance=1e-6, steps=40, param_sets=None):
    """Verify several playgrounds, each over several parameter sets. Returns
    (all_ok, [(label, result), ...])."""
    # Vary the parameters so a term that happens to vanish at the defaults cannot
    # hide a mistake: switch the orientational terms off, push them hard, and take
    # the splay symmetry to both extremes.
    param_sets = param_sets or [
        ("defaults", {}),
        ("no orientation", {"k_tilt": 0.0, "k_splay": 0.0}),
        ("stiff", {"k_tilt": 40.0, "k_splay": 2.5}),
        ("abs splay", {"splay_symmetry": 1.0}),
        ("short cutoffs", {"rc": 1.6, "wc": 1.2}),
        # The spontaneous-curvature offset enters both orientational terms, and
        # at c0 = 0 it vanishes from both.
        ("curved", {"c0": 0.2}),
    ]
    results = []
    all_ok = True
    for ref in refs:
        for name, params in param_sets:
            try:
                res = verify_playground(ref, params=params, tolerance=tolerance,
                                        steps=steps)
            except Exception as exc:
                results.append((f"{ref} [{name}]", exc))
                all_ok = False
                continue
            if res is None:
                continue
            results.append((f"{ref} [{name}]", res))
            all_ok = all_ok and res.ok
    return all_ok, results
