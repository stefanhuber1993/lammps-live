"""Regression guard for the one vectorized MesoMem energy decomposition.

The old code hand-wrote the paper's energy expression five times across three
system modules (a scalar loop in mesomem_hex.get_potential_terms and again in
mesomem_hex._pair_terms and mesomem_sheet.get_potential_terms, plus a vectorized
copy in mesomem_sheet.get_total_potential_terms and mesomem_assembly). Those
copies are now replaced by MesoMem.energy_terms.

`old_pair_terms` below is the scalar version copied VERBATIM from the deleted
mesomem_hex._pair_terms (git history: lammps_live/systems/mesomem_hex.py:624-652
on main). It is the reference: if the vectorized rewrite ever drifts from what
shipped, this fails.
"""
import math

import numpy as np
import pytest

from lammps_live.forcefields.mesomem import ISO, SIGMA, SPLAY, TILT, MesoMem
from lammps_live.playground.state import Box, FrameState, build_pairs, normalize_rows

EPS = 1.0


def old_pair_terms(d, ni, nj, rc, wc, zeta, ktilt, ksplay, splay_sym):
    """Verbatim transcription of the pre-refactor scalar formulas."""
    r = float(np.linalg.norm(d))
    if r >= rc or r < 1e-9:
        return 0.0, 0.0, 0.0
    rhat = d / r
    if r < SIGMA:
        t2 = (SIGMA / r) ** 2
        u_iso = EPS * (t2 * t2 - 2.0 * t2)
    else:
        g = math.pi * 0.5 * (r - SIGMA) / (rc - SIGMA)
        u_iso = -EPS * math.cos(g) ** (2.0 * zeta)
    u_tilt = u_splay = 0.0
    if r < wc:
        rga = 0.5 * wc
        denom = (r / wc) ** 4 - 1.0
        if denom < -1e-14:
            w = math.exp((r * r) / (rga * rga * denom))
            nir = float(ni @ rhat)
            njr = float(nj @ rhat)
            ninj = float(ni @ nj)
            ninj_eff = (1.0 - splay_sym) * ninj + splay_sym * abs(ninj)
            u_tilt = 0.5 * ktilt * (nir * nir + njr * njr) * w
            u_splay = 0.5 * ksplay * (ninj_eff - 1.0) ** 2 * w
    return u_iso, u_tilt, u_splay


def reference_totals(positions, directors, box, rc, wc, **kw):
    """Whole-system totals from the old scalar formulas over every unique pair,
    with the same minimum-image convention the new pair list uses."""
    n = len(positions)
    u_iso = u_tilt = u_splay = 0.0
    for a in range(n):
        for b in range(a + 1, n):
            d = box.minimum_image(positions[a] - positions[b]) if box is not None \
                else positions[a] - positions[b]
            ui, ut, us = old_pair_terms(d, directors[a], directors[b],
                                        rc=rc, wc=wc, **kw)
            u_iso += ui
            u_tilt += ut
            u_splay += us
    return u_iso, u_tilt, u_splay


def make_config(n=40, box=None, seed=7):
    rng = np.random.default_rng(seed)
    if box is None:
        pos = rng.uniform(-3.0, 3.0, size=(n, 3))
    else:
        lo, hi = np.array(box.lo), np.array(box.hi)
        pos = rng.uniform(lo, hi, size=(n, 3))
    dirs = normalize_rows(rng.normal(size=(n, 3)))
    return pos, dirs


# A spread of parameter sets including the paper's standard conditions, the
# splay-symmetry extremes, wc > rc (which must clamp), and a degenerate rc.
PARAM_SETS = [
    dict(k_tilt=12.0, k_splay=1.0, zeta=5.0, rc=2.5, wc=2.0, splay_symmetry=0.0),
    dict(k_tilt=12.0, k_splay=1.0, zeta=5.0, rc=2.5, wc=2.0, splay_symmetry=1.0),
    dict(k_tilt=3.0, k_splay=0.1, zeta=1.0, rc=2.5, wc=2.0, splay_symmetry=0.5),
    dict(k_tilt=40.0, k_splay=30.0, zeta=11.0, rc=3.0, wc=3.0, splay_symmetry=0.0),
    dict(k_tilt=12.0, k_splay=1.0, zeta=5.0, rc=1.5, wc=2.8, splay_symmetry=0.0),
    dict(k_tilt=12.0, k_splay=1.0, zeta=5.0, rc=2.5, wc=0.0, splay_symmetry=0.0),
]

BOXES = [
    None,                                                  # non-periodic patch
    Box.centered(8.0, 8.0, 8.0, periodic=(True, True, False)),   # periodic sheet
    Box.cube(8.0, periodic=(True, True, True)),                  # periodic box
]


@pytest.mark.parametrize("params_in", PARAM_SETS)
@pytest.mark.parametrize("box", BOXES, ids=["open", "periodic_xy", "periodic_xyz"])
def test_vectorized_matches_old_scalar(params_in, box):
    ff = MesoMem()
    params = ff.new_params(params_in)
    pos, dirs = make_config(box=box)
    state = FrameState(positions=pos, directors=dirs, box=box)
    pairs = build_pairs(pos, ff.interaction_cutoff(params), box)

    terms = ff.energy_terms(state, pairs, params)
    got = (terms[ISO].sum(), terms[TILT].sum(), terms[SPLAY].sum())

    # The clamp is part of the declaration, so the reference must be fed the
    # EFFECTIVE wc -- which is exactly the behaviour _effective_wc() provided.
    want = reference_totals(
        pos, dirs, box,
        rc=params["rc"], wc=params["wc"], zeta=params["zeta"],
        ktilt=params["k_tilt"], ksplay=params["k_splay"],
        splay_sym=params["splay_symmetry"],
    )
    assert got == pytest.approx(want, rel=1e-10, abs=1e-10)


def test_wc_clamped_to_rc():
    """wc > rc must be capped, the behaviour the old _effective_wc() gave. The
    clamp now lives on the Param, so it applies to both the pair_coeff line and
    the energy decomposition."""
    ff = MesoMem()
    params = ff.new_params(dict(rc=1.8, wc=2.9))
    assert params["wc"] == pytest.approx(1.8)
    assert "1.8" in ff.coeff_commands(params)[0]


def test_per_particle_share_sums_to_total():
    """Per-pair energies let one evaluation serve both panels: summing the pairs
    that touch each particle double-counts every pair exactly once per endpoint,
    so it must come to twice the whole-system total."""
    ff = MesoMem()
    params = ff.new_params()
    pos, dirs = make_config(n=30)
    state = FrameState(positions=pos, directors=dirs)
    pairs = build_pairs(pos, ff.interaction_cutoff(params), None)
    terms = ff.energy_terms(state, pairs, params)

    total = terms[ISO].sum()
    per_particle = sum(terms[ISO][pairs.touching(i)].sum() for i in range(len(pos)))
    assert per_particle == pytest.approx(2.0 * total, rel=1e-12)


def test_empty_pair_list_is_harmless():
    """rc = 0 switches the interaction off entirely; the panels must still get
    well-formed (empty) arrays rather than raising."""
    ff = MesoMem()
    params = ff.new_params(dict(rc=0.0))
    pos, dirs = make_config(n=5)
    state = FrameState(positions=pos, directors=dirs)
    pairs = build_pairs(pos, ff.interaction_cutoff(params), None)
    assert len(pairs) == 0
    terms = ff.energy_terms(state, pairs, params)
    assert set(terms) == {ISO, TILT, SPLAY}
    assert all(len(v) == 0 for v in terms.values())


def test_spontaneous_curvature_is_the_sphere_it_asks_for():
    """c0 (the pair style's own spontaneous-curvature offset, a live dial since
    the 50k box's "Curved membrane" knob): beads on a sphere of radius R with
    their directors pointing IN cost no tilt and no splay at c0 = 1/R -- the
    sphere is what c0 asks for -- and cost both at c0 = 0, where flat is.
    (Pointing out, the sign flips: the pair style's rhat runs j -> i.)"""
    from lammps_live.playground.state import icosphere_faces

    ff = MesoMem()
    R = 5.0
    _, normals = icosphere_faces(3)
    pos = normals * R
    state = FrameState(positions=pos, directors=-normals, box=None)

    def orientational(c0):
        params = ff.new_params({"c0": c0})
        pairs = build_pairs(pos, ff.interaction_cutoff(params), None)
        terms = ff.energy_terms(state, pairs, params)
        return terms[TILT].sum() + terms[SPLAY].sum()

    assert orientational(1.0 / R) == pytest.approx(0.0, abs=1e-9)
    assert orientational(0.0) > 1.0
