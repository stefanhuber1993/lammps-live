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
    assert disclosure.introduced(_lesson("mesomem_patch"),
                                 _lesson("mesomem_bead")) == ("panel", "energy")
    assert disclosure.introduced(_lesson("mesomem_patch_torque"),
                                 _lesson("mesomem_patch")) == ()
    assert set(disclosure.introduced(_lesson("mesomem_sheet"),
                                     _lesson("mesomem_patch_torque"))) == {
        "colour", "plots"}
    assert disclosure.introduced(_lesson("mesomem_remote"),
                                 _lesson("mesomem_rod")) == ("snellius",)


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
    assert app._callouts == ("panel", "energy")


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
