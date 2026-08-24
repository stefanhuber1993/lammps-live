"""Driving the remote connect card from the joystick, wired up through the app.

The card is tested on its own in test_remote_panel.py. What is tested here is the
part that only exists in app.py and that is easy to get subtly wrong: WHICH
buttons and axes reach the card while it is up, and which keep their usual
meaning. The card is modal, so a mapping that leaks -- the trigger still starting
a run that does not exist, or the hat still walking a slider panel nobody can see
-- is a control that does nothing in front of an audience.
"""
import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

pytest.importorskip("lammps")

from lammps_live import config
from lammps_live.app import App
from lammps_live.remote import session as session_mod
from lammps_live.ui.remote_panel import RemotePanel

FRAME = 1.0 / 60


class FakeDevice:
    """What the app asks an input source for, with nothing plugged in."""

    def __init__(self):
        self.buttons = set()
        self.hat = (0, 0)

    def poll_buttons(self):
        return set(self.buttons)

    def poll_hat(self):
        return self.hat

    def read(self):
        return 0.0, 0.0, 0.0

    def send_force(self, fx, fy, stiffness):
        pass

    def set_damper_coefficient(self, coefficient):
        pass

    def close(self):
        pass


class FakePanel(RemotePanel):
    """A visible card with a fixed set of buttons and no session behind it.

    A real session means SSH; what the app's mapping needs is only "the card is up
    and these are its buttons", so the two methods the app calls are recorded and
    the rest of the panel is left exactly as it is.
    """

    def __init__(self, names=("copy", "connect", "close")):
        super().__init__()
        self.visible = True
        # Through the real method, so the focus starts where a drawn card would put
        # it rather than on whatever happens to be first.
        self.set_shown(names, "connect")
        self.pressed = []

    def activate_focus(self):
        name = self.focused_button
        if name is None:
            return False
        self.pressed.append(name)
        return True


@pytest.fixture
def app():
    # Built in mouse mode and then told it is a joystick: constructing it as one
    # opens the real Sidewinder, and there is not one plugged into CI. Everything
    # under test reads `input_mode` and `source`, both of which are replaced here.
    a = App(input_mode="mouse", initial_system_key="mesomem_assembly")
    a.source.close()
    a.input_mode = "joystick"
    a.source = FakeDevice()
    a.remote_panel = FakePanel()
    yield a
    a.system.close()


def _press(app, button):
    app.source.buttons = {button}
    app._poll_device_buttons()
    app.source.buttons = set()
    app._poll_device_buttons()


def _hat(app, direction):
    app.source.hat = direction
    app._poll_device_buttons()
    app.source.hat = (0, 0)
    app._poll_device_buttons()


def test_the_hat_walks_the_cards_buttons(app):
    panel = app.remote_panel
    assert panel.focused_button == "connect"
    _hat(app, (1, 0))
    assert panel.focused_button == "close"
    _hat(app, (-1, 0))
    assert panel.focused_button == "connect"
    _hat(app, (-1, 0))
    assert panel.focused_button == "copy"


def test_the_trigger_presses_the_focused_button_and_does_not_start_a_run(app):
    panel = app.remote_panel
    app.sim_playing = False
    _press(app, config.JOYSTICK_PLAY_PAUSE_BUTTON)
    assert panel.pressed == ["connect"]
    assert app.sim_playing is False, \
        "there is no run behind a card that is up because there is no run"


def test_reset_does_not_fire_behind_the_card(app):
    """It belongs to a scene that is not running. And it is the one button here
    whose side effect would be expensive -- a rebuild -- for nothing."""
    resets = []
    app._reset_simulation = lambda *a, **k: resets.append(1)
    _press(app, config.JOYSTICK_RESET_BUTTON)
    assert resets == []


def test_switching_playground_still_works_behind_the_card(app):
    """Switching away is how you leave the card, and it costs the session nothing."""
    cycled = []
    app._cycle_system = lambda step=1: cycled.append(step)
    _press(app, config.JOYSTICK_NEXT_PLAYGROUND_BUTTON)
    _press(app, config.JOYSTICK_PREV_PLAYGROUND_BUTTON)
    assert cycled == [1, -1]


def test_the_stick_walks_the_buttons_and_nothing_else_gets_it(app):
    panel = app.remote_panel
    start = panel.focused_button
    out = app._route_stick(0.9, 0.8, 0.7, FRAME)
    assert out == (0.0, 0.0, 0.0), "nothing behind the card may be driven"
    assert app._stick_target == "panel"
    assert panel.focused_button != start
    # A held stick steps once, not every frame (see RemotePanel.push_axis).
    moved = panel.focused_button
    for _ in range(10):
        app._route_stick(0.9, 0.0, 0.0, FRAME)
    assert panel.focused_button == moved


def test_a_hand_that_was_setting_a_slider_keeps_setting_it(app):
    """The card must not silently change what the hat and the stick mean under a
    hand that is mid-gesture somewhere else. Only the viewport hands them over."""
    panel = app.remote_panel
    app.focus.set_stops([s for s in app._sliders() if not s.advanced])
    app.focus.enter_stops()
    assert not app.focus.on_viewport
    focused = panel.focused_button

    _hat(app, (1, 0))
    assert panel.focused_button == focused, "the hat still belongs to the panel rows"
    _press(app, config.JOYSTICK_PLAY_PAUSE_BUTTON)
    assert panel.pressed == []
    out = app._route_stick(0.9, 0.0, 0.0, FRAME)
    assert out == (0.0, 0.0, 0.0)
    assert app._stick_target == "slider"


def test_the_card_being_down_gives_everything_back(app):
    app.remote_panel.visible = False
    app.sim_playing = False
    _press(app, config.JOYSTICK_PLAY_PAUSE_BUTTON)
    assert app.sim_playing is True
    assert app.remote_panel.pressed == []


def test_the_stub_panel_agrees_with_the_real_one():
    """FakePanel overrides one method; if the real card's contract moves, this test
    file is checking a shape that no longer exists."""
    real = RemotePanel()
    for name in ("focused_button", "step_focus", "push_axis", "activate_focus",
                 "set_shown"):
        assert hasattr(real, name)
    assert real.focused_button is None, "a card that is down focuses nothing"
    assert session_mod.READY  # the module the panel's states come from is importable
