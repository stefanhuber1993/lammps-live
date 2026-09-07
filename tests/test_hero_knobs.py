"""Applying a scene's big move, and taking it back off.

Each hero knob is one line of UI over a piece of state that has to be exactly
right, because what the screen then claims is strong: THESE beads, with THIS
attraction, and no orientation; or this membrane, at this temperature. Every
failure mode here is one where the screen would be lying -- sliders that say 12
over a membrane that has stopped being one, a round trip that comes back to the
declared defaults instead of to where the demo actually was, an engagement that
survives into a playground it was never made about.
"""
import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

pytest.importorskip("lammps")

from lammps_live import config
from lammps_live.app import App

FRAME = 1.0 / 60


@pytest.fixture
def sheet():
    """The sheet, whose one knob is Heat (the temperature dial)."""
    a = App(input_mode="mouse", initial_system_key="mesomem_sheet")
    a._tick(FRAME)
    yield a
    a.system.close()


@pytest.fixture
def patch():
    """The seven-bead patch, whose one knob is Remove orientation (parameters).
    Seven beads, so it builds in a moment."""
    a = App(input_mode="mouse", initial_system_key="mesomem_patch")
    a._tick(FRAME)
    yield a
    a.system.close()


def _slider(app, key):
    return dict(zip(app.extra_slider_keys, app.extra_sliders))[key]


# ---- a knob that drives force-field parameters ----------------------------

def test_removing_orientation_drives_the_real_sliders_to_zero(patch):
    """THROUGH the sliders, not around them. The app pushes every slider into the
    system once a frame, so this is both how the physics changes and how the panel
    stays honest about it -- an override held anywhere else would show k_tilt = 12
    on a screen where the patch has visibly stopped being flat.
    """
    knob = patch.system.spec.lesson.hero_knobs[0]
    assert knob.label == "Remove orientation"
    assert any(_slider(patch, k).value != 0.0 for k in knob.params)

    patch._toggle_hero(0)

    assert patch.hero_engaged == {0}
    for key in knob.params:
        assert _slider(patch, key).value == 0.0, key
    patch._tick(FRAME)
    live = patch.system.live_param_values()
    for key in knob.params:
        assert live[key] == pytest.approx(0.0), key


def test_releasing_it_goes_back_to_where_the_demo_was_not_to_the_defaults(patch):
    """The whole value of a hero knob is that it is a ROUND TRIP from wherever the
    demo happens to be. A presenter who has spent a minute finding an interesting
    k_tilt and then shows what removing orientation does must get that k_tilt
    back, not the paper's.
    """
    _slider(patch, "k_tilt").value = 4.5
    _slider(patch, "k_splay").value = 2.25
    patch._tick(FRAME)

    patch._toggle_hero(0)
    patch._tick(FRAME)
    patch._toggle_hero(0)

    assert patch.hero_engaged == set()
    assert _slider(patch, "k_tilt").value == pytest.approx(4.5)
    assert _slider(patch, "k_splay").value == pytest.approx(2.25)
    patch._tick(FRAME)
    assert patch.system.live_param_values()["k_tilt"] == pytest.approx(4.5)


def test_it_leaves_every_other_control_alone(patch):
    """A knob names what it changes. Anything else on the panel is somebody's
    setting and none of its business."""
    patch.temp_slider.value = 0.12
    zeta = _slider(patch, "zeta").value
    patch._toggle_hero(0)
    assert _slider(patch, "zeta").value == pytest.approx(zeta)
    assert patch.temp_slider.value == pytest.approx(0.12)


# ---- a knob that drives the temperature dial ------------------------------

def test_heat_moves_the_temperature_dial_and_cool_puts_it_back(sheet):
    """The dial, not a hidden target: the panel reads the temperature the sheet is
    actually being held at, and the melt marker on that same track is what says
    how close to melting the knob has taken it."""
    knob = sheet.system.spec.lesson.hero_knobs[0]
    assert knob.label == "Heat" and knob.temperature is not None
    started = sheet.temp_slider.value

    sheet._toggle_hero(0)
    assert sheet.hero_engaged == {0}
    assert sheet.temp_slider.value == pytest.approx(knob.temperature)
    sheet._tick(FRAME)
    assert sheet.system._target_temp == pytest.approx(knob.temperature)

    sheet._toggle_hero(0)
    assert sheet.temp_slider.value == pytest.approx(started)


def test_heat_stays_inside_the_dial_it_drives(sheet):
    """A knob asking for a temperature off the end of the slider would leave the
    handle and the physics disagreeing."""
    knob = sheet.system.spec.lesson.hero_knobs[0]
    sheet._toggle_hero(0)
    assert sheet.temp_slider.vmin <= sheet.temp_slider.value <= sheet.temp_slider.vmax
    assert sheet.temp_slider.value < sheet.system.spec.melt_temp
    assert knob.temperature < sheet.system.spec.melt_temp


# ---- more than one, and the rest of the app -------------------------------

def test_two_knobs_do_not_forget_each_other(sheet):
    """Each knob restores what IT found, so engaging a second one and releasing it
    must not undo the first. Driven here with a second knob grafted onto the sheet,
    since no shipped scene declares two yet -- the mechanism is what is being
    tested, and this is the failure it exists to avoid.
    """
    import dataclasses
    from lammps_live.playgrounds._knobs import NO_ORIENTATION

    lesson = sheet.system.spec.lesson
    both = dataclasses.replace(
        lesson, hero_knobs=lesson.hero_knobs + (NO_ORIENTATION,))
    sheet.system.spec = dataclasses.replace(sheet.system.spec, lesson=both)

    sheet._toggle_hero(0)                       # Heat
    warm = sheet.temp_slider.value
    sheet._toggle_hero(1)                       # and remove orientation
    assert sheet.hero_engaged == {0, 1}
    assert _slider(sheet, "k_tilt").value == 0.0

    sheet._toggle_hero(1)                       # put the orientation back
    assert sheet.hero_engaged == {0}
    assert _slider(sheet, "k_tilt").value == pytest.approx(12.0)
    assert sheet.temp_slider.value == pytest.approx(warm), "it stayed warm"


def test_reset_lets_go_of_every_knob(sheet):
    """R means "the beginning", both halves of it (see App._reset_simulation). A
    reset that left a knob engaged would rebuild a fresh scene straight back into
    the altered state with nothing on screen explaining why, and the values it
    would later restore belong to sliders reset has just overwritten.
    """
    sheet._toggle_hero(0)
    sheet._reset_simulation()

    assert sheet.hero_engaged == set()
    assert sheet._hero_saved == {}
    assert sheet.temp_slider.value == pytest.approx(
        sheet.system.spec.temperature.default)


def test_switching_playground_lets_go_of_every_knob(sheet):
    """A knob is a statement about the scene you are looking at, and its index
    points into THAT scene's declared tuple. Carried across a switch it would
    leave the next scene altered with a button in whatever state the last one
    left."""
    sheet._toggle_hero(0)
    sheet._build_system("mesomem_patch")

    assert sheet.hero_engaged == set()
    assert sheet._hero_saved == {}


def test_the_click_the_key_and_the_device_button_are_one_state(patch):
    """All three go through `_toggle_hero`, so they cannot disagree about whether a
    knob is engaged -- the bug where a click after a keypress steps from whatever
    the keypress left behind. Same reason the bead-colour toggle routes its click
    through the Choice rather than stepping its own copy.
    """
    import pygame

    patch.renderer.draw_hero_knobs(patch.system.spec.lesson.hero_knobs, frozenset())
    assert patch.renderer.hero_hit(patch.renderer._hero_rects[0].center) == 0

    patch._toggle_hero(0)                                  # what the click calls
    assert patch.hero_engaged == {0}
    patch._handle_events(FRAME)
    assert patch.hero_engaged == {0}, "an empty event pump must not change it"

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F1,
                                         mod=0, unicode="", scancode=0))
    patch._handle_events(FRAME)                            # what F1 calls
    assert patch.hero_engaged == set()

    # And the device button, which is the number drawn on the button itself.
    patch.source.poll_buttons = lambda: {config.JOYSTICK_HERO_FIRST_BUTTON}
    patch.source.poll_hat = lambda: (0, 0)
    patch._poll_device_buttons()
    assert patch.hero_engaged == {0}


def test_a_scene_with_no_knobs_ignores_the_toggle(patch):
    """Every key that does not apply to a playground does nothing on it, and these
    are three of them. Not an error and not a crash."""
    patch._build_system("mesomem_bead")
    assert patch.system.spec.lesson.hero_knobs == ()
    patch._toggle_hero(0)
    patch._toggle_hero(3)
    assert patch.hero_engaged == set()
    # And nothing is drawn, so a stale rect from the patch cannot be hit.
    patch.renderer.draw_hero_knobs((), frozenset())
    assert patch.renderer.hero_hit((0, 0)) is None
    assert patch.renderer._hero_rects == []


# ---- the bead colouring across a switch -----------------------------------

def test_the_assembly_box_opens_in_cluster_colouring(patch):
    """Its first declared colouring is its default, and arriving on it applies
    that -- what is happening in that box IS aggregates finding each other, and
    the colouring that paints them is the one that says so."""
    assert patch.renderer.bead_color_mode == "director"     # the patch's own first
    patch._build_system("mesomem_assembly")
    assert patch.renderer.bead_color_mode == "cluster"


def test_a_colouring_the_viewer_picked_follows_them(patch):
    """Once somebody has chosen, it is their preference: a scene's default must not
    silently undo it on every Tab."""
    patch.color_choice.step(1)                              # director -> energy
    assert patch.renderer.bead_color_mode == "energy"
    assert patch._color_user_chosen

    patch._build_system("mesomem_assembly")
    assert patch.renderer.bead_color_mode == "energy", "their choice survived"

    # ...except onto a scene that does not offer it, where there is no preference
    # to honour and the scene's own first is the only answer.
    patch.color_choice.step(1)                              # -> cluster
    assert patch.renderer.bead_color_mode == "cluster"
    patch._build_system("mesomem_sheet")
    assert patch.renderer.bead_color_mode == "director"


def test_the_two_bead_scene_is_not_a_colour_stop_at_all(patch):
    """It offers no colouring, so the joystick's focus cycle must not stop on a
    toggle that is not drawn."""
    patch._build_system("mesomem_bead")
    assert patch.system.spec.bead_colors == ()
    assert all(stop is not patch.color_choice
               for stop in patch.focus._stops), "a stop with nothing to show"
