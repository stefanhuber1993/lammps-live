"""A simulation destroyed by its own sliders is an event, not a crash.

The failure this file exists for: `zeta` below 1 is fine to the CPU pair style and
rejected by the Kokkos one, so a remote run streamed happily -- the per-chunk
`run ... pre no` never re-validates coefficients -- until Reset rebuilt, `run 0`
validated them, and the exception took out the server AND its A100 allocation, via
the server's own `scancel`.

So: a rebuild falls back to values that are known to build, the event is reported
once with a sentence anyone can read and the raw error underneath, and the sliders
follow whatever the simulation settled on.
"""
import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from lammps_live.playground.faults import Fault, first_line, summarise
from lammps_live.ui.alert import Alert, wrap_text


# ---- the summaries ---------------------------------------------------------

@pytest.mark.parametrize("error, expect", [
    ("ERROR: Lost atoms: original 900 current 41 (src/thermo.cpp:483)",
     "flew out of the box"),
    ("ERROR on proc 0: Non-numeric atom coords - simulation unstable", "blew up"),
    ("ERROR: mesomem/kk requires zeta >= 1 (src/KOKKOS/pair_mesomem_kokkos.cpp:585)",
     "will not accept that parameter value"),
    ("ERROR: Neighbor list overflow, boost neigh_modify one", "too many neighbours"),
    ("ERROR: Unknown pair style mesomem", "does not have a style"),
    ("ERROR: Incorrect args for pair coefficients", "was not valid for this build"),
])
def test_a_raw_lammps_error_gets_a_sentence(error, expect):
    assert expect in summarise(error)


def test_an_unrecognised_error_says_so_rather_than_guessing():
    assert summarise("ERROR: something nobody has met yet") == \
        "LAMMPS stopped the simulation."
    assert summarise("") == "LAMMPS stopped the simulation."


def test_the_detail_keeps_the_input_line_and_drops_the_traceback():
    raw = ("ERROR: mesomem/kk requires zeta >= 1 (src/pair.cpp:585)\n"
           "Last input line: run 0\n"
           "  File \"server.py\", line 274, in serve_client\n")
    detail = first_line(raw)
    assert "requires zeta >= 1" in detail
    assert "Last input line: run 0" in detail
    assert "serve_client" not in detail


def test_a_fault_survives_the_wire():
    fault = Fault.from_error("ERROR: Lost atoms: original 900 current 41",
                            reverted={"zeta": 5.0}, fatal=False)
    back = Fault.from_message(fault.as_message())
    assert back.summary == fault.summary
    assert back.detail == fault.detail
    assert back.reverted == {"zeta": 5.0}
    assert back.fatal is False
    assert "reverted zeta to 5" in fault.line()
    assert Fault.from_message(None) is None


# ---- the card --------------------------------------------------------------

def _fonts():
    pygame.display.init()
    pygame.font.init()

    class FakeRenderer:
        screen = pygame.Surface((1200, 800))
        header_font = pygame.font.Font(None, 26)
        small_font = pygame.font.Font(None, 14)
        font = pygame.font.Font(None, 18)
        sim_width = 900
        window_size = (1200, 800)

    return FakeRenderer()


def test_the_card_shows_for_three_seconds_then_stops():
    alert = Alert(show_seconds=0.15)
    assert not alert.visible
    alert.show("The simulation blew up.", "ERROR: Lost atoms")
    assert alert.visible
    renderer = _fonts()
    alert.draw(renderer)                      # must not raise, headless
    import time
    time.sleep(0.2)
    assert not alert.visible
    alert.draw(renderer)                      # nor when it is gone


def test_a_second_fault_replaces_the_first_rather_than_queueing():
    alert = Alert(show_seconds=5.0)
    alert.show_fault(Fault.from_error("ERROR: Lost atoms"))
    alert.show_fault(Fault.from_error("ERROR: Neighbor list overflow"))
    assert "neighbours" in alert.summary
    assert len(alert.shown) == 2               # both recorded, one on screen


def test_wrapping_never_loses_a_long_path():
    pygame.font.init()
    font = pygame.font.Font(None, 14)
    path = "/gpfs/home3/stefanh/Projects/MesoMemLive/mesomem_gpu/pair_mesomem.cpp:585"
    lines = wrap_text(f"ERROR: at {path}", font, 120)
    assert any(path in line for line in lines), lines


# ---- the rebuild ladder ----------------------------------------------------

def _ladder_playground():
    from lammps_live.playground import Playground, random_fill
    return Playground(
        name="fault ladder", force_field="mesomem",
        scenario=random_fill(n=120, box=8.0), mode="sim", seed=11,
    )


def test_reset_puts_a_refused_value_back_before_it_can_reach_the_rebuild():
    """The zeta failure, and why it is now unreachable through Reset.

    A value the local style accepts and the cluster's rejects (stood in for here by
    a coefficient that only appears when zeta < 1) used to stream happily and then
    kill the rebuild that Reset triggered -- which on the cluster took the server
    and its allocation with it. The recovery was the fallback ladder below.

    Reset now restores every parameter to what the playground declares BEFORE it
    rebuilds, so the rebuild is never handed the value in the first place. No fault,
    because nothing failed: the sliders are back where they started and so is the
    run.
    """
    pytest.importorskip("lammps")
    from lammps_live.playground.system import PlaygroundSystem

    system = PlaygroundSystem(_ladder_playground(), mode_name="sim", analysis=False)
    try:
        real = system.force_field.pair_commands
        seen = []

        def sabotaged(params):
            if float(params["zeta"]) < 1.0:
                seen.append(float(params["zeta"]))
                return real(params) + ["pair_coeff 1 1 nonsense"]
            return real(params)

        system.force_field.pair_commands = sabotaged
        good = system.params["zeta"]

        system.set_extra_param("zeta", 0.4)
        assert system.params["zeta"] == pytest.approx(0.4)
        seen.clear()
        system.reset()                          # must NOT raise, and must not fail

        assert not seen, f"the rebuild was handed zeta={seen}"
        assert system.take_fault() is None, "nothing failed, so nothing to report"
        assert system.params["zeta"] == good
        assert system.live_param_values()["zeta"] == good
        system.step(5)
        assert system.unstable is None
    finally:
        system.close()


def test_a_rebuild_that_fails_anyway_falls_back_instead_of_raising():
    """The ladder itself, for the failures Reset's parameter restore cannot
    pre-empt.

    Injected as a ONE-SHOT poison -- the first `pair_commands` of the rebuild is
    broken and the next is not -- because that is the shape of what a ladder can
    recover from at all. A failure that depends on a VALUE cannot be, now that every
    rung is holding the same declared values, and it is right that such a thing is
    fatal: nothing this playground declares builds, which is not a slider problem.
    """
    pytest.importorskip("lammps")
    from lammps_live.playground.system import PlaygroundSystem

    system = PlaygroundSystem(_ladder_playground(), mode_name="sim", analysis=False)
    try:
        real = system.force_field.pair_commands
        remaining = [1]

        def poison_once(params):
            if remaining and remaining.pop():
                return real(params) + ["pair_coeff 1 1 nonsense"]
            return real(params)

        system.force_field.pair_commands = poison_once
        system.reset()                          # must NOT raise

        fault = system.take_fault()
        assert fault is not None
        assert not fault.fatal, "there is a running simulation, so not fatal"
        assert "Restarted with" in fault.summary
        assert "pair_coeff 1 1 nonsense" in fault.detail
        # And it is really running.
        assert system.lmp is not None
        system.step(5)
        assert system.unstable is None
        # Popped, not latched: the card is shown once.
        assert system.take_fault() is None
    finally:
        system.close()


# ---- a rebuild must not move the controlled particle ------------------------

def test_a_rebuild_does_not_carry_the_old_atom_ordering_into_the_new_one():
    """Reset with a live bead used to blow the simulation up on the first frame.

    The mechanism, because it is not obvious from either end: the id-order
    permutation (`_order`) is cached per FRAME, and a rebuild does not advance the
    frame counter -- so after Reset the cache still described the LAMMPS instance
    that had just been closed. `controlled_local` then named a different particle,
    `GameMode.on_built` read ITS coordinate as the control plane, and the first
    `constrain()` teleported the real controlled bead most of a sigma onto that
    plane -- on top of a neighbour, which is a blow-up. Measured before the fix:
    the 7-bead patch went from T = 0.001 to T = 2.8 in two frames.

    So this pins the invariant that closes it: after a rebuild, the controlled
    particle is where the fresh build put it, the plane it is held in is ITS OWN
    coordinate, and holding it there moves nothing.
    """
    pytest.importorskip("lammps")
    import numpy as np

    from lammps_live.playground import registry

    system = registry.build("mesomem_patch")
    try:
        # Steps first: LAMMPS reorders its local arrays as it runs, so the stale
        # permutation has to actually differ from the fresh one for this to bite.
        for _ in range(20):
            system.step(10)
        system.reset()

        ic = system.controlled_local()
        assert ic is not None
        ids = system.lmp.numpy.extract_atom("id")[:system.natoms]
        assert int(ids[ic]) == system.controlled_id, (
            "controlled_local named some other particle -- the stale permutation")

        mode = system.mode
        x = np.array(system.lmp.numpy.extract_atom("x"))
        assert mode._pin_value == pytest.approx(float(x[ic][mode.pin_axis])), (
            "the control plane is not the controlled particle's own coordinate")

        # ...and holding it on that plane is therefore a no-op, not a teleport.
        before = np.array(x[ic][:3])
        mode.constrain()
        after = np.array(system.lmp.numpy.extract_atom("x")[ic][:3])
        assert float(np.linalg.norm(after - before)) < 1e-6

        # End to end: the fresh state stays cold instead of exploding.
        for _ in range(30):
            system.step(10)
        assert system.unstable is None
        assert system.take_fault() is None
        temp = system.get_thermo_state()[0]
        assert temp < 0.1, f"the rebuilt patch ran away to T = {temp}"
    finally:
        system.close()


def test_the_first_build_settles_before_it_picks_the_control_plane():
    """The same bug without any Reset at all, on a scenario LAMMPS fills itself.

    There, `_pick_controlled` has to read the positions back out of LAMMPS to
    choose a particle -- which builds an id-order permutation BEFORE the settle
    runs. The settle then reorders, and the permutation is stale by the time
    `GameMode.on_built` uses it to ask where its particle is. Same wrong-particle
    coordinate, same teleport, on the very first frame of the playground.
    """
    pytest.importorskip("lammps")
    import numpy as np

    from lammps_live.playground import Playground, random_fill
    from lammps_live.playground.system import PlaygroundSystem

    playground = Playground(
        name="filled box", force_field="mesomem",
        # Dense enough that LAMMPS really does rebin during the build's own
        # settle: a permutation that never changes would not test anything.
        scenario=random_fill(n=200, box=9.0), mode="game", seed=7,
    )
    system = PlaygroundSystem(playground, mode_name="game", analysis=False)
    try:
        ic = system.controlled_local()
        ids = system.lmp.numpy.extract_atom("id")[:system.natoms]
        assert int(ids[ic]) == system.controlled_id
        mode = system.mode
        x = np.array(system.lmp.numpy.extract_atom("x"))
        assert mode._pin_value == pytest.approx(float(x[ic][mode.pin_axis]))
    finally:
        system.close()


def test_a_rebuild_of_a_lammps_filled_box_picks_its_bead_from_the_new_atoms():
    """The third reader, and the earliest: `_pick_controlled` itself.

    On a scenario LAMMPS fills, choosing the controlled particle means reading the
    positions back -- through the same id-order permutation. On a REBUILD that
    happens before anything else has had a chance to drop the dead instance's
    permutation, and gathering through it either names the wrong particle for the
    whole session or, when the two builds differ in size, indexes off the end.
    """
    pytest.importorskip("lammps")
    import numpy as np

    from lammps_live.playground import Playground, random_fill
    from lammps_live.playground.system import PlaygroundSystem

    playground = Playground(
        name="filled box", force_field="mesomem",
        scenario=random_fill(n=200, box=9.0), mode="game", seed=3,
    )
    system = PlaygroundSystem(playground, mode_name="game", analysis=False)
    try:
        for _ in range(15):
            system.step(10)
        system.reset()                      # must NOT raise, and must not misname
        ic = system.controlled_local()
        ids = system.lmp.numpy.extract_atom("id")[:system.natoms]
        assert int(ids[ic]) == system.controlled_id
        # The choice is "nearest the box centre", so it has to actually be that --
        # gathered through a stale permutation it would be some other particle.
        x = np.array(system.lmp.numpy.extract_atom("x")[:system.natoms])
        by_id = x[np.argsort(ids, kind="stable")]
        centre = np.asarray(system.box.center)
        want = int(np.argmin(np.linalg.norm(by_id[:, :2] - centre[:2], axis=1)))
        assert system.controlled_id == want + 1
    finally:
        system.close()


def test_a_released_bead_is_the_case_that_broke():
    """Same rebuild, with the bead let go of first -- how the bug was reported.

    Worth its own test because `attached` survives a rebuild (a bead you let go of
    stays let go), so the released path reaches `constrain()` down a different
    branch: no leash, and nothing else holding the particle to stop a bad plane
    value from being the only thing that moves it.
    """
    pytest.importorskip("lammps")
    from lammps_live.playground import registry

    system = registry.build("mesomem_patch")
    try:
        system.toggle_puller_attached()
        assert not system.puller_attached()
        for _ in range(20):
            system.step(10)
        system.reset()
        for _ in range(30):
            system.step(10)
        assert system.unstable is None
        assert system.get_thermo_state()[0] < 0.1
    finally:
        system.close()
