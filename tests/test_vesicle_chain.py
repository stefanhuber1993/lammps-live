"""The vesicle with one space-filling chain inside it (mesomem_vesicle_chain).

As tests/test_vesicle_polymer.py, and for the same reason split in two: first the
pure geometry the no-push-off construction rests on -- the Moore curve is a
closed, self-avoiding, unit-step walk, and the cube it fills fits in the lumen --
then a miniature built in LAMMPS, run, and stiffened mid-run.
"""
import colorsys
import dataclasses

import numpy as np
import pytest

from lammps_live.playground import registry
from lammps_live.playground.scenario import VesicleChain, vesicle_chain
from lammps_live.playground.state import hilbert_curve, moore_curve

# Small enough to build in a second: an order-3 chain (512 beads, an 8-site cube)
# in the smallest vesicle whose lumen holds it at the shipped spacing.
SMALL = dict(n_membrane=1280, a=0.70, chain_order=3, settle_steps=20)


# --- the curves ---------------------------------------------------------------

def _steps(walk, closed):
    pts = np.vstack([walk, walk[:1]]) if closed else walk
    return np.abs(np.diff(pts, axis=0)).sum(axis=1)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_the_moore_curve_is_a_closed_self_avoiding_unit_step_walk(order):
    curve = moore_curve(order)
    side = 2 ** order
    assert curve.shape == (side ** 3, 3)
    assert curve.min() == 0 and curve.max() == side - 1
    assert len(np.unique(curve, axis=0)) == side ** 3, "a site is visited twice"
    # Closed: the step from the last site back to the first counts.
    assert np.all(_steps(curve, closed=True) == 1), "a bond is not one step"


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_the_hilbert_curve_is_open_and_ends_one_edge_from_its_start(order):
    curve = hilbert_curve(order)
    side = 2 ** order
    assert len(curve) == side ** 3
    assert len(np.unique(curve, axis=0)) == side ** 3
    assert np.all(_steps(curve, closed=False) == 1)
    # The property the Moore construction needs of it.
    assert tuple(curve[0]) == (0, 0, 0)
    assert tuple(curve[-1]) == (side - 1, 0, 0)


def test_the_moore_curve_is_space_filling_not_a_serpentine():
    """What it was chosen for over one big `lattice_ring`: any stretch of the
    chain is compact. Every aligned 64-bead stretch of an order-4 curve fills a
    4x4x4 block exactly."""
    curve = moore_curve(4)
    for k in range(0, len(curve), 64):
        piece = curve[k:k + 64]
        assert np.all(np.ptp(piece, axis=0) == 3), k


def test_the_curves_reject_orders_they_cannot_have():
    with pytest.raises(ValueError):
        moore_curve(0)
    with pytest.raises(ValueError):
        hilbert_curve(-1)


# --- the scenario's build -----------------------------------------------------

def _built(**overrides):
    scenario = VesicleChain(**{**SMALL, **overrides})
    params = scenario.new_params()
    return scenario, params, scenario.build(params, np.random.default_rng(7))


def test_build_is_the_membrane_plus_one_chain():
    scenario, params, build = _built()
    _, n_mem = scenario.subdivision(params)
    assert build.n_uploaded == n_mem
    assert len(build.positions) == scenario.particle_count(params) == n_mem + 512
    assert np.count_nonzero(build.types == 2) == 512
    assert scenario.loop_lengths(params) == [512]


def test_the_chain_is_one_closed_loop_in_the_template():
    """Read the topology back out of the molecule file LAMMPS is handed: one
    molecule, and the bonds form a single cycle through every bead."""
    scenario, params, build = _built()
    polymer = build.positions[build.n_uploaded:]
    path = scenario._write_template(params, polymer)
    sections, current = {}, None
    for line in open(path):
        word = line.strip()
        if word in ("Coords", "Types", "Diameters", "Masses", "Molecules",
                    "Bonds", "Angles"):
            current = sections.setdefault(word, [])
        elif word and current is not None:
            current.append([int(float(v)) for v in word.split()])
    n = len(polymer)
    assert {row[1] for row in sections["Molecules"]} == {1}
    bonds = np.array(sections["Bonds"])[:, 2:] - 1
    assert len(bonds) == n and len(sections["Angles"]) == n
    # Every bead in exactly two bonds, and walking them visits all n once.
    assert np.all(np.bincount(bonds.ravel(), minlength=n) == 2)
    nbrs = {i: [] for i in range(n)}
    for a, b in bonds:
        nbrs[a].append(b)
        nbrs[b].append(a)
    seen, prev, here = {0}, None, 0
    while True:
        nxt = [j for j in nbrs[here] if j != prev][0]
        if nxt == 0:
            break
        assert nxt not in seen
        seen.add(nxt)
        prev, here = here, nxt
    assert len(seen) == n, "the bonds make more than one loop"


def test_every_bond_is_intact_and_no_two_beads_overlap():
    cKDTree = pytest.importorskip("scipy.spatial").cKDTree
    _, _, build = _built()
    chain = build.positions[build.n_uploaded:]
    bonds = np.linalg.norm(np.diff(np.vstack([chain, chain[:1]]), axis=0), axis=1)
    assert bonds.max() < 1.4, "a bond starts beyond FENE's maximum extension"
    d, _ = cKDTree(chain).query(chain, k=2)
    assert d[:, 1].min() > 0.55, "two beads start on top of each other"


def test_the_chain_starts_inside_the_vesicle_whatever_its_rotation():
    scenario, params, build = _built()
    radius = scenario.radius(params)
    r = np.linalg.norm(build.positions[build.n_uploaded:], axis=1)
    # Centred, and its corners (the furthest a rotation can put anything) clear.
    assert np.allclose(build.positions[build.n_uploaded:].mean(axis=0), 0.0,
                       atol=0.1)
    assert r.max() < radius - 1.0
    assert scenario.chain_half_diagonal(params) < radius - 1.0


def test_a_chain_the_lumen_cannot_hold_is_refused():
    with pytest.raises(ValueError, match="lumen"):
        _built(chain_order=4)


def test_the_build_is_reproducible_from_the_seed():
    _, _, a = _built()
    _, _, b = _built()
    assert np.allclose(a.positions, b.positions)


def test_the_tints_are_one_rainbow_along_the_chain():
    scenario, params, build = _built()
    tints = scenario.render_tints(params)
    n_mem = build.n_uploaded
    assert tints.shape == (len(build.positions), 4)
    assert np.all(tints[:n_mem, 3] == 0.0), "the membrane must keep its banding"
    assert np.all(tints[n_mem:, 3] == 1.0)
    rgb = tints[n_mem:, :3] / 255.0
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0
    hue = np.array([colorsys.rgb_to_hsv(*c)[0] for c in rgb])
    # The hue climbs with the chain index, once round the whole wheel...
    assert np.all(np.diff(hue) > 0)
    assert hue[0] < 0.01 and hue[-1] > 0.99
    # ...so the closed loop shows no seam where its two ends meet.
    assert np.abs(rgb[0] - rgb[-1]).max() < 0.02


# --- the shipped playground ---------------------------------------------------

@pytest.fixture(scope="module")
def shipped():
    return registry.load("mesomem_vesicle_chain")


def test_the_shipped_chain_is_the_order_5_curve_with_room_to_spare(shipped):
    scenario = shipped.scenario
    params = scenario.new_params()
    _, n_mem = scenario.subdivision(params)
    assert scenario.chain_length(params) == 32768
    assert n_mem == 23120
    clearance = scenario.radius(params) - scenario.chain_half_diagonal(params)
    assert clearance > 5.0, clearance


def test_the_card_quotes_the_counts_that_are_running(shipped):
    params = shipped.scenario.new_params()
    n_chain = shipped.scenario.chain_length(params)
    n_mem = shipped.scenario.subdivision(params)[1]
    claim = shipped.lesson.claim
    assert f"{round(n_mem / 1000)}k" in claim
    assert f"{round(n_chain / 1000)}k" in claim
    assert "same three terms" in claim.lower()
    assert not shipped.lesson.hook


def test_the_hero_knob_toggles_k_bend_between_its_two_extremes(shipped):
    from lammps_live.playground import forcefield
    from lammps_live.playground.system import make_spec

    (knob,) = shipped.lesson.hero_knobs
    assert set(knob.params) == {"k_bend"}
    stiff = knob.params["k_bend"]
    # It starts floppy, and the knob is the way to the other end.
    ff = forcefield.get(shipped.force_field)()
    assert ff.new_params(shipped.resolved_params(None))["k_bend"] == 1.0
    # Stiff enough to press the envelope out (10 did nothing visible), cooled so
    # the membrane survives it, and ramped because a jump breaks a bond.
    assert stiff >= 100.0
    assert f"{stiff:.0f}" in knob.caption
    assert knob.temperature is not None and knob.temperature < shipped.temperature_default
    assert f"{knob.temperature:.2f}" in knob.caption
    assert knob.ramp_time >= 5.0
    # A live dial, and its slider reaches the knob's value.
    spec = make_spec(shipped, shipped.mode)
    sliders = {s.key: s for s in spec.extra_sliders}
    assert "k_bend" in sliders
    lo, hi = shipped.param_ranges["k_bend"]
    assert lo <= 0.0 and stiff <= hi


def test_it_still_goes_to_the_cluster(shipped):
    assert shipped.remote is not None
    assert shipped.remote.time == "10:00:00"
    assert shipped.remote.profile == "cluster-gpu"


# --- the deck -----------------------------------------------------------------

pytest.importorskip("lammps")


@pytest.fixture(scope="module")
def small_system(shipped):
    """A miniature of the shipped playground, actually built in LAMMPS."""
    from lammps_live.playground.system import PlaygroundSystem

    playground = dataclasses.replace(
        shipped, scenario=vesicle_chain(**SMALL, timestep=0.005,
                                        sim_time_per_frame=0.05),
        remote=None, seed=99)
    system = PlaygroundSystem(playground, mode_name="sim")
    yield system
    system.close()


def _chain_bonds(system):
    state = system.frame_state()
    chain = state.positions[np.asarray(state.types) == 2]
    return np.linalg.norm(np.diff(np.vstack([chain, chain[:1]]), axis=0), axis=1)


def test_the_deck_creates_one_closed_chain(small_system):
    params = small_system.scenario_params
    n_chain = small_system.scenario.chain_length(params)
    assert small_system.natoms == small_system.scenario.particle_count(params)
    # A closed chain of L beads has L bonds and L angles: no free end.
    assert small_system.lmp.extract_global("nbonds") == n_chain
    assert small_system.lmp.extract_global("nangles") == n_chain
    assert small_system.params["k_bend"] == 1.0


def test_it_runs_floppy_then_stiff_without_blowing_up(small_system):
    """The hero knob's move, done to a running system the way the app does it:
    cooled, and k_bend ramped from 0 to the stiff value over the knob's ramp_time
    with every right-angle corner of the curve loaded, then held."""
    from lammps_live.playgrounds.mesomem_vesicle_chain import STIFF_CHAIN

    small_system.step(1000)
    assert _chain_bonds(small_system).max() < 1.5
    small_system.set_target_temp(STIFF_CHAIN.temperature)
    stiff = STIFF_CHAIN.params["k_bend"]
    n_ramp = round(STIFF_CHAIN.ramp_time / 0.005)
    for i in range(1, 41):
        small_system.set_extra_param("k_bend", 1.0 + (stiff - 1.0) * i / 40)
        small_system.step(n_ramp // 40)
        assert _chain_bonds(small_system).max() < 1.5, i
    small_system.step(1000)
    state = small_system.frame_state()
    assert np.isfinite(state.positions).all()
    assert _chain_bonds(small_system).max() < 1.5
    assert 0.0 < small_system.get_thermo_state()[0] < 1.5
    small_system.set_extra_param("k_bend", 1.0)
    small_system.set_target_temp(0.2)


def test_the_observables_make_sense_for_one_chain(small_system):
    values = dict(zip(("radius", "gyration", "contact"),
                      (float(line.split("=")[1]) for line in
                       small_system.get_hud_lines())))
    scenario, params = small_system.scenario, small_system.scenario_params
    assert values["radius"] == pytest.approx(scenario.radius(params), rel=0.15)
    assert 0.2 * values["radius"] < values["gyration"] < 0.9 * values["radius"]
    assert values["contact"] >= 0.0


# --- the chain on its own energy scale -----------------------------------------

def test_the_bending_energy_of_a_moore_curve_is_its_corners():
    """k_bend (1 + cos theta) at every bead: 0 on a straight run, k_bend at a
    right-angle corner -- and a lattice curve has nothing else."""
    from lammps_live.forcefields.mesomem_polymer import MesoMemPolymer

    ff = MesoMemPolymer()
    walk = moore_curve(2).astype(float)
    e = ff.bonded_bead_energies(walk, [len(walk)], {"k_bend": 7.0})
    assert set(np.round(e, 9)) <= {0.0, 7.0}
    assert np.isclose(e, 7.0).sum() > len(walk) // 3, "a Moore curve folds"
    assert ff.bonded_bead_energies(walk, [len(walk)], {"k_bend": 0.0}).max() == 0.0


def test_the_style_puts_the_chain_on_its_own_scale(shipped):
    """The energy colouring paints the chain's bending, not its binding, and says
    where: a second range on the style, and the chain beads are the tinted ones."""
    lo, hi = shipped.render_style.tint_energy_range
    assert lo == 0.0 < hi
    scenario, params = _built()[:2]
    tints = scenario.render_tints(params)
    start, loops = scenario.bonded_loops(params)
    assert (tints[:start, 3] <= 0.5).all()
    assert (tints[start:start + sum(loops), 3] > 0.5).all()


def test_the_energy_colouring_carries_the_chains_bending(small_system):
    """The membrane keeps its pair energy; each chain bead carries the bending
    energy of the angle centred on it, recomputed from positions -- so it is the
    same number the remote client derives from the frame it was sent."""
    from lammps_live.playground.forcefield import with_bonded_energies

    small_system.set_extra_param("k_bend", 5.0)
    small_system.step(10)
    state = small_system.frame_state()
    chain = np.asarray(state.types) == 2
    got = small_system.get_bead_energies()
    params = small_system.scenario_params
    start, loops = small_system.scenario.bonded_loops(params)
    want = small_system.force_field.bonded_bead_energies(
        state.positions[start:], loops, small_system.params)
    if small_system._smoothing_tau <= 0.0:
        assert got[chain] == pytest.approx(want)
    assert 0.0 <= got[chain].min() and got[chain].max() <= 2 * 5.0 + 1e-9
    assert got[~chain].max() < 1.0, "the membrane is still on its binding scale"
    # The helper leaves a system without chains alone.
    plain = np.arange(4.0)
    assert with_bonded_energies(small_system.force_field,
                                registry.load("mesomem_sheet").scenario, None,
                                small_system.params, np.zeros((4, 3)), plain) is plain
    small_system.set_extra_param("k_bend", 1.0)
