"""The kiosk lock: nothing closes, minimises or leaves fullscreen without the
password, and a wrong one is refused (see lammps_live/kiosk.py)."""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

from lammps_live import kiosk
from lammps_live.app import App

FRAME = 1 / 60


def test_the_password_check_and_the_lockout(monkeypatch):
    lock = kiosk.KioskLock("booth")
    assert lock.check("booth")
    for _ in range(kiosk.MAX_TRIES):
        assert not lock.check("nope")
    assert lock.locked_out_for() > 0
    assert not lock.check("booth"), "refused during the lockout, even if right"


@pytest.fixture
def app():
    a = App(input_mode="mouse", initial_system_key="mesomem_bead",
            lock=kiosk.KioskLock("booth"))
    a._tick(FRAME)
    yield a
    if a._watchdog is not None:
        a._watchdog.stop()
    a.system.close()


def test_no_lockdown_or_watchdog_under_a_headless_driver(app):
    assert not app.lock.platform_locked
    assert app._watchdog is None


def _type(text):
    for ch in text:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=0, mod=0,
                                             unicode=ch))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN,
                                         mod=0, unicode="\r"))


def test_closing_the_window_asks_for_the_password(app):
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    assert app._handle_events(FRAME) is True, "did not close"
    assert app._unlock is not None and app._unlock["action"] == "quit"
    _type("wrong")
    assert app._handle_events(FRAME) is True
    assert app._unlock["error"]
    _type("booth")
    assert app._handle_events(FRAME) is False, "closes with the password"


def test_escape_and_cancel(app):
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE,
                                         mod=0, unicode=""))
    assert app._handle_events(FRAME) is True
    assert app._unlock is not None
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE,
                                         mod=0, unicode=""))
    app._handle_events(FRAME)
    assert app._unlock is None and app.locked


def test_the_idle_return_locks_it_again(app):
    app.locked = False
    app._note_input()
    app._last_input -= App.IDLE_SECONDS + 1
    app._check_idle()
    assert app.locked


def test_unlocked_is_an_ordinary_app():
    a = App(input_mode="mouse", initial_system_key="mesomem_bead")
    try:
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        assert a._handle_events(FRAME) is False
    finally:
        a.system.close()
