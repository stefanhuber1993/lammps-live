"""Walking the playground picker from the keyboard.

Tab steps to the next playground, shift-Tab to the previous one, and the two
have to agree on the same order -- a step forward followed by a step back is
where you started.
"""
import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pytest.importorskip("lammps")

from lammps_live.app import App

FRAME = 1.0 / 60


@pytest.fixture
def app():
    a = App(input_mode="mouse", initial_system_key="mesomem_assembly")
    yield a
    a.system.close()
    pygame.event.clear()


def _tab(shift=False):
    return pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB,
                              mod=pygame.KMOD_LSHIFT if shift else 0,
                              unicode="\t")


def test_shift_tab_steps_back_to_where_tab_came_from(app, monkeypatch):
    steps = []
    monkeypatch.setattr(app, "_cycle_system", steps.append)

    pygame.event.post(_tab())
    app._handle_events(FRAME)
    pygame.event.post(_tab(shift=True))
    app._handle_events(FRAME)

    assert steps == [1, -1]


def test_a_tab_and_a_shift_tab_land_on_the_playground_they_started_from(app):
    start = app.system_key

    pygame.event.post(_tab())
    app._handle_events(FRAME)
    assert app.system_key != start, "Tab actually moved"

    pygame.event.post(_tab(shift=True))
    app._handle_events(FRAME)
    assert app.system_key == start


# ---- Back and Next on screen ---------------------------------------------------

def _click(pos):
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos)


def _deck(app):
    """Draw one frame and return the bottom row's buttons by name."""
    app._tick(FRAME)
    return {b.name: b for b in app.renderer.playback_buttons}


def test_back_and_next_are_drawn_only_where_there_is_somewhere_to_go(app):
    """No rollover (App._cycle_system), so no button that would do nothing: the
    first scene has no Back, the last no Next, the ones between have both."""
    keys = [key for key, _ in app.systems]
    assert {"prev", "next"} <= set(_deck(app))
    app._build_system(keys[0])
    assert "prev" not in _deck(app) and "next" in _deck(app)
    app._build_system(keys[-1])
    assert "next" not in _deck(app) and "prev" in _deck(app)


def test_clicking_next_and_back_walks_the_sequence(app, monkeypatch):
    steps = []
    monkeypatch.setattr(app, "_cycle_system", steps.append)
    deck = _deck(app)
    pygame.event.post(_click(deck["next"].rect.center))
    app._handle_events(FRAME)
    pygame.event.post(_click(deck["prev"].rect.center))
    app._handle_events(FRAME)
    assert steps == [1, -1]


def test_a_click_on_a_hero_knob_is_not_a_camera_grab(app):
    """The deck sits inside the sim view, over the turntable: a press on a knob
    has to fire the knob, not start orbiting the camera."""
    _deck(app)
    rect = app.renderer._hero_rects[0]
    pygame.event.post(_click(rect.center))
    app._handle_events(FRAME)
    assert not app._orbit_dragging
    assert app.hero_engaged == {0}
