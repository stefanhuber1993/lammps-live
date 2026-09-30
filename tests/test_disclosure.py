"""The interface appears a piece at a time along the sequence (ui/disclosure.py),
and the demo goes back to its first scene when nobody is using it."""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest

from lammps_live.app import App
from lammps_live.playground import registry
from lammps_live.ui import disclosure

FRAME = 1 / 60


def _lesson(key):
    return registry.load(key).lesson


def test_the_first_scene_shows_none_of_it():
    assert disclosure.shown(_lesson("mesomem_bead")) == frozenset()


def test_each_element_is_introduced_where_it_first_appears():
    # The hero row comes with the patch: the first scene has no knob, and the
    # arrow on its first appearance is what tells anyone it is there.
    assert disclosure.introduced(_lesson("mesomem_patch"),
                                 _lesson("mesomem_bead")) == ("panel", "energy",
                                                              "hero")
    assert disclosure.introduced(_lesson("mesomem_patch_torque"),
                                 _lesson("mesomem_patch")) == ()
    assert set(disclosure.introduced(_lesson("mesomem_sheet"),
                                     _lesson("mesomem_patch_torque"))) == {
        "colour", "plots"}
    # The rod has no knob, so the row comes back on the 50k box -- with its arrow.
    assert disclosure.introduced(_lesson("mesomem_remote"),
                                 _lesson("mesomem_rod")) == ("hero", "snellius")
    # The view cut arrives on the assembly box, and not before.
    assert "slice" in disclosure.introduced(_lesson("mesomem_assembly"),
                                            _lesson("mesomem_sheet"))
    for key in ("mesomem_bead", "mesomem_patch", "mesomem_patch_torque",
                "mesomem_sheet"):
        assert "slice" not in disclosure.shown(_lesson(key)), key


def test_the_hero_row_is_shown_exactly_where_there_are_knobs():
    """Derived from Lesson.hero_knobs, never listed in Lesson.ui."""
    for key, pg in registry.all_playgrounds():
        if pg.lesson is None or pg.lesson.ui is None:
            continue
        assert "hero" not in pg.lesson.ui, key
        assert ("hero" in disclosure.shown(pg.lesson)) == bool(
            pg.lesson.hero_knobs), key


def test_a_scene_without_a_lesson_shows_everything():
    assert disclosure.shown(None) == disclosure.ALL


@pytest.fixture
def app():
    a = App(input_mode="mouse", initial_system_key="mesomem_patch")
    a._tick(FRAME)
    yield a
    a.system.close()


def test_the_panel_goes_away_on_the_first_scene_and_comes_back(app):
    assert app.renderer.panel_visible
    app._build_system("mesomem_bead")
    assert not app.renderer.panel_visible
    assert app.renderer.sim_width == app.renderer.window_size[0]
    app._tick(FRAME)
    app._build_system("mesomem_patch")
    assert app.renderer.panel_visible
    assert app._callouts == ("panel", "energy", "hero")


def test_a_minute_untouched_goes_back_to_the_start(app):
    app._note_input()
    app._last_input -= App.IDLE_SECONDS - 5.0
    assert 4.0 < app._idle_seconds_left() <= 5.0     # counting down
    app._check_idle()
    assert app.system_key == "mesomem_patch", "not yet"
    app._last_input -= 6.0
    app._check_idle()
    assert app.system_key == "mesomem_bead"
    # And it does not keep firing for nobody.
    assert app._idle_seconds_left() is None
    app._last_input -= 2 * App.IDLE_SECONDS
    app._check_idle()
    assert app.system_key == "mesomem_bead"


def test_any_input_cancels_the_countdown(app):
    app._note_input()
    app._last_input -= App.IDLE_SECONDS - 3.0
    app._note_input()
    assert app._idle_seconds_left() > App.IDLE_SECONDS - 1.0


def test_the_lever_cuts_nothing_where_slicing_is_not_offered(app):
    """Hidden means not offered: on the patch the lever is live and cuts nothing,
    and there is no gauge for it."""
    app.input_mode = "joystick"
    for value in (0.0, 0.1, 0.5, 0.5):
        app.source.poll_throttle = (lambda v=value: v)
        app._tick(FRAME)
    assert app.view_slice.plane is None
    assert "slice" not in app.renderer._ui_rects


def test_the_assembly_box_offers_the_cut_and_draws_its_gauge(app):
    app._build_system("mesomem_assembly")
    app.input_mode = "joystick"
    for value in (0.0, 0.1, 0.5, 0.5, 0.5):
        app.source.poll_throttle = (lambda v=value: v)
        app._tick(FRAME)
    assert app.view_slice.progress > 0.0
    assert "slice" in app.renderer._ui_rects
    assert "slice" in app._callouts
