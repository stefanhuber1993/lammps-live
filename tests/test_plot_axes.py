"""The panel's plots keep a y axis that holds still (RollingHistory.axis_range).

Re-fitting the axis to what is on screen every frame made the plots jump: the
trace sat still and the ruler under it moved."""
from lammps_live.ui.plotting import RollingHistory


def _history():
    return RollingHistory(10.0, ["press"])


def test_the_axis_starts_at_its_floor_and_does_not_follow_the_noise():
    h = _history()
    assert h.axis_range("press", [0.08, 0.09], (-0.5, 0.5)) == (-0.5, 0.5)
    assert h.axis_range("press", [0.01, 0.2], (-0.5, 0.5)) == (-0.5, 0.5)


def test_it_widens_in_round_steps_and_never_shrinks_back():
    h = _history()
    lo, hi = h.axis_range("press", [0.7], (-0.5, 0.5))
    assert lo <= -0.5 and hi >= 0.7
    assert (lo, hi) == (round(lo, 6), round(hi, 6))
    assert h.axis_range("press", [0.0], (-0.5, 0.5)) == (lo, hi)


def test_a_reset_gives_the_next_run_a_new_scale():
    h = _history()
    h.axis_range("press", [40.0], (-0.5, 0.5))
    h.reset()
    assert h.axis_range("press", [0.1], (-0.5, 0.5)) == (-0.5, 0.5)


def test_every_scene_with_plots_declares_its_ranges():
    """Fixed, measured axes (Playground.plot_ranges) wherever the plots are shown:
    the widening fallback is for a scene nobody has measured."""
    from lammps_live.playground import registry
    from lammps_live.ui import disclosure
    for key in registry.bundled_keys():
        pg = registry.load(key)
        if pg.lesson is None or not pg.lesson.plots:
            continue
        if "plots" not in disclosure.shown(pg.lesson):
            continue
        for plot in ("press", "energy"):
            lo, hi = pg.plot_ranges[plot]
            assert lo < hi, (key, plot)
