"""Taking the force field's central claim away, and putting it back.

The button is one line of UI over a piece of state that has to be exactly right,
because what it is claiming is strong: THESE beads, with THIS attraction, and no
orientation. Every failure mode here is one where the screen would be lying --
sliders that say 12 over a membrane that has stopped being one, a round trip that
comes back to the declared defaults instead of to where the demo actually was, an
engagement that survives into a playground it was never made about.
"""
import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

pytest.importorskip("lammps")

from lammps_live.app import App

FRAME = 1.0 / 60


@pytest.fixture
def app():
    a = App(input_mode="mouse", initial_system_key="mesomem_sheet")
    a._tick(FRAME)
    yield a
    a.system.close()


def _slider(app, key):
    return dict(zip(app.extra_slider_keys, app.extra_sliders))[key]


def test_engaging_it_drives_the_real_sliders_to_zero(app):
    """THROUGH the sliders, not around them. The app pushes every slider into the
    system once a frame, so this is both how the physics changes and how the panel
    stays honest about it -- an override held anywhere else would show k_tilt = 12
    on a screen where the membrane has visibly stopped being one.
    """
    thesis = app.system.spec.lesson.thesis
    assert thesis is not None
    before = {key: _slider(app, key).value for key in thesis.params}
    assert any(v != 0.0 for v in before.values()), "nothing to remove"

    app._toggle_thesis()

    assert app.thesis_engaged
    for key in thesis.params:
        assert _slider(app, key).value == 0.0, key
    # And the system agrees, once a frame has pushed them.
    app._tick(FRAME)
    live = app.system.live_param_values()
    for key in thesis.params:
        assert live[key] == pytest.approx(0.0), key


def test_releasing_it_goes_back_to_where_the_demo_was_not_to_the_defaults(app):
    """The whole value of the button is that it is a ROUND TRIP from wherever the
    demo happens to be. A presenter who has spent a minute finding an interesting
    k_tilt and then shows what removing orientation does must get that k_tilt
    back, not the paper's.
    """
    _slider(app, "k_tilt").value = 4.5
    _slider(app, "k_splay").value = 2.25
    app._tick(FRAME)

    app._toggle_thesis()
    app._tick(FRAME)
    app._toggle_thesis()

    assert not app.thesis_engaged
    assert _slider(app, "k_tilt").value == pytest.approx(4.5)
    assert _slider(app, "k_splay").value == pytest.approx(2.25)
    app._tick(FRAME)
    assert app.system.live_param_values()["k_tilt"] == pytest.approx(4.5)


def test_it_leaves_every_other_dial_alone(app):
    """A Thesis names the orientational terms. Anything else on the panel is
    somebody's setting and none of its business."""
    app.temp_slider.value = 0.12
    zeta = _slider(app, "zeta").value
    app._toggle_thesis()
    assert _slider(app, "zeta").value == pytest.approx(zeta)
    assert app.temp_slider.value == pytest.approx(0.12)


def test_reset_puts_the_orientation_back(app):
    """R means "the beginning", both halves of it (see App._reset_simulation). A
    reset that left the orientation switched off would rebuild a fresh membrane
    straight back into a droplet with nothing on screen explaining why -- and the
    saved values it would later restore belong to sliders reset has just
    overwritten.
    """
    app._toggle_thesis()
    app._reset_simulation()

    assert not app.thesis_engaged
    assert app._thesis_saved == {}
    assert _slider(app, "k_tilt").value == pytest.approx(12.0)


def test_switching_playground_puts_the_orientation_back(app):
    """It is a statement about the membrane you are looking at. Carried across a
    switch it would leave the next scene silently crippled with its button in
    whatever state the last one left."""
    app._toggle_thesis()
    app._build_system("mesomem_patch")

    assert not app.thesis_engaged
    assert app._thesis_saved == {}
    assert _slider(app, "k_tilt").value == pytest.approx(12.0)


def test_the_keyboard_shortcut_and_the_button_are_one_state(app):
    """O and the click both go through `_toggle_thesis`, so they cannot disagree
    about whether it is engaged -- the bug where a click after a keypress steps
    from whatever the keypress left behind. Same reason the bead-colour toggle
    routes its click through the Choice rather than stepping its own copy.
    """
    import pygame

    app.renderer.draw_thesis_button(app.system.spec.lesson.thesis, False)
    assert app.renderer.thesis_hit(app.renderer.thesis_button.rect.center)

    app._toggle_thesis()                                   # what the click calls
    assert app.thesis_engaged
    app._handle_events(FRAME)                              # no input queued
    assert app.thesis_engaged, "an empty event pump must not change it"
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_o,
                                         mod=0, unicode="o", scancode=0))
    app._handle_events(FRAME)                              # what O calls
    assert not app.thesis_engaged


def test_a_playground_with_no_thesis_ignores_the_toggle(app):
    """Every key that does not apply to a playground does nothing on it, and this
    is one. Not an error and not a crash -- the two-bead scene simply has no
    membrane to take apart."""
    app._build_system("mesomem_bead")
    assert app.system.spec.lesson.thesis is None
    app._toggle_thesis()
    assert not app.thesis_engaged
    # And the button is not drawn, so a stale rect from the sheet cannot be hit.
    app.renderer.draw_thesis_button(None, False)
    assert not app.renderer.thesis_hit((0, 0))
    assert not app.renderer.thesis_hit(
        app.renderer.thesis_button.rect.center)
