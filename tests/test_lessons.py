"""The taught sequence: what each playground says, and how much of the instrument
it shows while saying it.

The demo is eight scenes that make one argument (see registry._ACTS), and almost
everything that holds that argument together is a DECLARATION rather than code --
a title, a claim, a hook, a list of which dials are everyday here. Declarations
drift silently: nothing breaks when a new playground arrives with no lesson, or
when a claim grows into a paragraph nobody at the back of the room can read, or
when the two patch scenes stop showing the same panel and quietly ruin the one
comparison they exist for. So the rules the design actually rests on are asserted
here, next to the reasons for them.
"""
import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from lammps_live.playground import registry
from lammps_live.playground.spec import HeroKnob, Lesson

# The narrowest window the card is expected to hold up in. The claim is drawn from
# the left margin of the sim view in `claim_font`, and the sim view is the window
# minus the fixed-width panel -- so "does the claim fit" is a question with a real
# answer in pixels, which is what this checks instead of counting words. A word
# count was the first version of this rule and it was measuring the wrong thing:
# it failed a line for containing a dash.
NARROWEST_WINDOW = 1100


def _offered():
    return registry.all_playgrounds()


# ---- every scene in the demo is a scene in the story -----------------------

def test_every_offered_playground_teaches_something():
    """A playground in the demo with no lesson would draw no card and no position:
    an unexplained scene in the middle of an explanation."""
    missing = [key for key, pg in _offered() if pg.lesson is None]
    assert not missing, f"offered but not taught: {missing}"


def test_every_offered_playground_is_in_exactly_one_act():
    """The position indicator says which third of the argument you are in, so a
    scene in no act (or in two) is a scene it cannot place."""
    counts = {key: 0 for key, _ in _offered()}
    for _, keys in registry.acts():
        for key in keys:
            assert key in counts, f"{key} is in an act but not offered"
            counts[key] += 1
    assert all(n == 1 for n in counts.values()), counts


def test_the_acts_are_the_offered_order_regrouped():
    """Not a second ordering. The rail draws the acts and the number keys walk
    `bundled_keys`, so if the two disagree the marks are in a different order from
    the scenes they stand for."""
    flat = [key for _, keys in registry.acts() for key in keys]
    assert flat == registry.bundled_keys()


def test_lesson_position_places_each_scene_and_declines_to_place_the_others():
    keys = registry.bundled_keys()
    assert registry.lesson_position(keys[0]) == (1, len(keys), "Rules")
    assert registry.lesson_position("mesomem_sheet") == (4, len(keys), "Material")
    assert registry.lesson_position(keys[-1])[0] == len(keys)
    # A shelved playground is legitimately outside the story: index 0, no act, and
    # the renderer draws no rail rather than inventing a position for it.
    assert registry.lesson_position("lj_argon") == (0, len(keys), "")
    assert registry.lesson_position("not_a_playground")[0] == 0


# ---- the three lines ------------------------------------------------------

def test_every_card_line_fits_the_narrowest_window():
    """The card is drawn straight onto the scene with no wrapping, so a line too
    long for the sim view runs under the position rail or off the edge. Measured
    against the real fonts at the real layout."""
    import pygame
    from lammps_live.ui.scale import UI
    from lammps_live.ui.theme import LESSON_CLAIM_SIZE, LESSON_TITLE_SIZE, PANEL_WIDTH

    # FONTS ONLY, no display. `pygame.display.set_mode` is process-global: calling
    # it here resized the display surface that test_pair_annotation's module-scoped
    # Renderer draws onto, and every annotation it drew afterwards was clipped to
    # 10x10 px. Font metrics need `font.init()` and nothing else.
    pygame.font.init()
    sim_w = NARROWEST_WINDOW - UI(PANEL_WIDTH)
    fonts = {"title": UI.font(LESSON_TITLE_SIZE, bold=True),
             "claim": UI.font(LESSON_CLAIM_SIZE),
             "instruction": UI.font(18)}
    over = {}
    for key, pg in _offered():
        index = registry.lesson_position(key)[0]
        for slot, text in (("title", f"{index}. {pg.lesson.title}"),
                           ("claim", pg.lesson.claim),
                           ("instruction", pg.lesson.instruction)):
            width = UI(10) + fonts[slot].size(text)[0]
            if width > sim_w - UI(16):
                over[f"{key}.{slot}"] = width
    assert not over, f"card lines wider than a {NARROWEST_WINDOW}px window: {over}"


def test_no_card_line_is_written_in_dashes():
    """Audience-facing text is plain sentences. The code around it writes a dash as
    a bare "--" (this codebase's convention, since it stays ASCII), and that
    convention leaking onto a projector is how a line stops reading like something
    a person said out loud."""
    bad = {}
    for key, pg in _offered():
        lesson = pg.lesson
        for slot in ("claim", "instruction", "hook"):
            text = getattr(lesson, slot)
            if "--" in text or "\u2014" in text or "\u2013" in text:
                bad[f"{key}.{slot}"] = text
        for knob in lesson.hero_knobs:
            if "--" in knob.caption:
                bad[f"{key}.{knob.label}"] = knob.caption
    assert not bad, f"dashes in audience text: {bad}"


def test_the_instruction_tells_a_hand_what_to_do():
    """Imperative, and about the controls. A scene nobody knows how to touch
    teaches nothing, and this is the only one of the three lines that is about the
    app rather than about the physics."""
    for key, pg in _offered():
        instruction = pg.lesson.instruction
        assert instruction and instruction[0].isupper(), key
        assert not instruction.endswith("?"), f"{key}: an instruction, not a question"


def test_every_scene_but_the_last_asks_the_next_one_s_question():
    """The hook is what makes eight scenes an argument rather than a menu. The last
    one has nothing after it, so it is the only one allowed to be silent."""
    keys = registry.bundled_keys()
    for key, pg in _offered():
        if key == keys[-1]:
            assert not pg.lesson.hook, "the last scene has nothing to hook to"
        else:
            assert pg.lesson.hook.endswith("?"), f"{key}: the hook is a question"


def test_the_first_and_last_scenes_are_the_same_sentence():
    """The bookend, and it is deliberate: the first scene says what a bead is, and
    the last says that nothing was added to it to get a vesicle. Break either half
    and the callback the talk is built around is gone.
    """
    keys = registry.bundled_keys()
    first = dict(_offered())[keys[0]].lesson
    last = dict(_offered())[keys[-1]].lesson
    assert "patch of lipid membrane" in first.claim
    assert "same three terms" in last.claim.lower()


# ---- progressive disclosure ------------------------------------------------

def test_the_two_bead_scene_offers_no_dials_and_no_plots():
    """The opening scene's subject is ONE interaction. A dial whose effect cannot
    be seen teaches that the model is arbitrary, and a time series of two beads is
    thermostat noise on a sample too small to have a temperature."""
    spec = dict(registry.list_playgrounds())["mesomem_bead"]
    assert [s.key for s in spec.extra_sliders if not s.advanced] == []
    assert spec.lesson.plots is False
    # Hidden, not removed: every dial is still there, behind the Advanced toggle.
    assert len(spec.extra_sliders) >= 6


def test_the_two_patch_scenes_show_exactly_the_same_panel():
    """These are the same seven beads and the same force field, and Tab between
    them IS the experiment (see mesomem_patch_torque). A panel that rearranged
    itself between the two would hide the one thing that changed."""
    specs = dict(registry.list_playgrounds())
    force = specs["mesomem_patch"]
    torque = specs["mesomem_patch_torque"]
    everyday = lambda s: [x.key for x in s.extra_sliders if not x.advanced]
    assert everyday(force) == everyday(torque) == ["k_tilt", "k_splay"]
    assert force.lesson.plots == torque.lesson.plots is False


def test_the_plots_arrive_with_the_first_scene_big_enough_to_have_a_statistic():
    """Off for the three Rules scenes, on from the sheet -- because on two beads or
    seven they are not true, not merely cluttered."""
    lessons = {key: pg.lesson for key, pg in _offered()}
    for _, keys in registry.acts():
        for key in keys:
            expected = registry.lesson_position(key)[2] != "Rules"
            assert lessons[key].plots is expected, key


def test_a_lesson_can_hide_a_dial_but_never_promote_a_cutoff():
    """`everyday_params` narrows only. Otherwise a playground could drag rc or wc
    -- cutoffs, advanced everywhere by the force field's own declaration -- onto
    the opening slide of a talk."""
    playground = dict(_offered())["mesomem_sheet"]
    # The force field says rc is advanced; a lesson naming it cannot change that.
    assert playground.is_everyday("rc", True) is False
    assert playground.is_everyday("k_tilt", False) is True


def test_a_playground_with_no_lesson_shows_everything():
    """The shelved atomistic classics, and anyone's own file: no lesson means no
    opinion about disclosure, not an empty one."""
    from lammps_live.playground.system import make_spec
    pg = registry.load("lj_argon")
    assert pg.lesson is None
    spec = make_spec(pg, pg.mode)
    assert spec.lesson is None
    declared = {p.name for p in pg.scenario.params if p.tier.is_live}
    everyday = {s.key for s in spec.extra_sliders if not s.advanced}
    assert everyday, "nothing was hidden, so something should be everyday"


# ---- the hero knobs -------------------------------------------------------

def test_the_hero_knobs_are_where_the_move_is_worth_making():
    """One or none per scene, and each on the scene where its move is the obvious
    thing to do next.

    "Remove orientation" needs a scene whose beads visibly ARE a membrane a second
    earlier, and the two it is on are the two where that is the whole claim: the
    seven-bead patch (which flattens because of its arrows) and the assembly box
    (which just built a sheet out of nothing). "Heat" needs a membrane big enough
    to flow, which is the sheet and the rod.

    The torque patch has none deliberately: it is meant to read as identical to the
    force patch next door, and a button on only one of them is a difference the eye
    has to rule out.
    """
    knobs = {key: [k.label for k in pg.lesson.hero_knobs] for key, pg in _offered()}
    assert knobs == {
        "mesomem_bead": [],
        "mesomem_patch": ["Remove orientation"],
        "mesomem_patch_torque": [],
        "mesomem_sheet": ["Heat"],
        "mesomem_assembly": ["Remove orientation"],
        "mesomem_rod": ["Heat"],
        "mesomem_remote": [],
        "mesomem_polymer": [],
    }, knobs


def test_a_hero_knob_drives_something_the_playground_actually_has():
    """A knob naming a parameter no playground declares would be a button that
    lights up and changes nothing."""
    for key, spec in registry.list_playgrounds():
        keys = {s.key for s in spec.extra_sliders}
        for knob in spec.lesson.hero_knobs:
            assert set(knob.params) <= keys, (key, knob.label)
            assert knob.params or knob.temperature is not None, (key, knob.label)


def test_the_temperature_a_heat_knob_asks_for_is_reachable_and_below_melting():
    """"Heat" has to leave a membrane that still IS one. Above `melt_temp` the
    sheet stops being a membrane and its barostat inflates the cell without
    bound (see HexSheet), so a knob that landed there would be labelled wrong.
    """
    for key, spec in registry.list_playgrounds():
        for knob in spec.lesson.hero_knobs:
            if knob.temperature is None:
                continue
            assert spec.temperature.vmin <= knob.temperature <= spec.temperature.vmax
            assert knob.temperature < spec.melt_temp, key
            assert knob.temperature > spec.temperature.default * 10, (
                f"{key}: heating to {knob.temperature} is not a change from "
                f"{spec.temperature.default}")


def test_removing_orientation_is_the_isotropic_only_preset_by_another_route():
    """One definition of what "isotropic only" means, not two: the preset is a
    place to start and the knob is a place to visit and come back from, and they
    have to agree about what is being removed."""
    for key, pg in _offered():
        for knob in pg.lesson.hero_knobs:
            if not knob.params or "isotropic_only" not in pg.presets:
                continue
            preset = pg.presets["isotropic_only"]
            assert knob.params == preset, key


def test_every_hero_knob_says_in_numbers_what_it_did():
    """A room that walked in halfway through sees a change and no explanation
    unless the screen carries one, and "warm" is a feeling until it is a number."""
    import re
    for key, pg in _offered():
        for knob in pg.lesson.hero_knobs:
            assert knob.caption, key
            assert knob.engaged_label != knob.label, key
            assert re.search(r"\d", knob.caption), (
                f"{key}: {knob.label}'s caption states no number")


def test_the_hero_row_fits_the_narrowest_window():
    """Laid out centred at a fixed width per knob, so a scene with too many of them
    would run off both edges rather than wrap."""
    from lammps_live.ui.scale import UI
    from lammps_live.ui.theme import HERO_GAP, HERO_W, PANEL_WIDTH
    sim_w = NARROWEST_WINDOW - UI(PANEL_WIDTH)
    for key, spec in registry.list_playgrounds():
        n = len(spec.lesson.hero_knobs)
        if not n:
            continue
        assert n * UI(HERO_W) + (n - 1) * UI(HERO_GAP) <= sim_w, key


# ---- what the force field declares about its own terms --------------------

def test_every_mesomem_term_says_what_it_is_a_function_of():
    """And the first one's argument list contains no director, which is the reason
    its torque column sits at zero -- the fact the callout is trying to make
    self-evident."""
    from lammps_live.playground import forcefield as ff_registry
    ff = ff_registry.get("mesomem")()
    assert len(ff.energy_terms_arguments) == len(ff.energy_terms_labels)
    iso, tilt, splay = ff.energy_terms_arguments
    assert iso == "r"
    assert "n" in tilt and "n" in splay


def test_a_term_doing_nothing_is_a_term_the_callout_fades():
    """The fade turns over at exactly the digit the display rounds away (see
    Renderer._pair_number), so a row can never be dimmed while showing a number.

    Measured on the real force field at the paper's coefficients, at the separation
    the two-bead scene STARTS at: the pair is inside rc, the van der Waals term is
    already pushing, and both orientational terms are silent because the directors
    are broadside -- which is the thing that scene opens by pointing at.
    """
    import numpy as np
    from lammps_live.playground import forcefield as ff_registry
    from lammps_live.playground.pair_probe import probe_pair
    from lammps_live.playground.params import ParamSet
    from lammps_live.playground.state import FrameState
    from lammps_live.ui.renderer import Renderer

    ff = ff_registry.get("mesomem")(bead_diameter=2.0)
    values = ParamSet.build(ff.params).values

    def rows(r, tilt_deg):
        a = np.radians(tilt_deg)
        state = FrameState(
            positions=np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]]),
            directors=np.array([[np.sin(a), 0.0, np.cos(a)], [0.0, 0.0, 1.0]]))
        ann = probe_pair(ff, state, values, plane_normal=(0.0, 1.0, 0.0))
        return [all(Renderer._pair_number(v) == "+0.00"
                    for v in (t.energy, t.radial_force, t.twist))
                for t in ann.terms]

    # Broadside, driving in: the isotropic term is awake and the two orientational
    # ones are asleep, all the way from the starting separation into the well.
    for r in (2.0, 1.6, 1.4, 1.0):
        assert rows(r, 0) == [False, True, True], r
    # Twisted: they wake up, and nothing is asleep any more.
    assert rows(1.2, 45) == [False, False, False]


# ---- the shapes the UI is handed ------------------------------------------

def test_the_spec_carries_the_lesson_to_the_renderer():
    """The renderer is handed a spec, not a playground, so anything it draws about
    the lesson has to have made the crossing."""
    for key, spec in registry.list_playgrounds():
        assert isinstance(spec.lesson, Lesson), key
        for knob in spec.lesson.hero_knobs:
            assert isinstance(knob, HeroKnob), key


# ---- what each scene colours its beads by ---------------------------------

def test_the_colourings_are_only_offered_where_they_mean_something():
    """Cluster colouring paints connected aggregates. On the two assembly boxes
    that IS the story and it is what they open in; on a single connected membrane
    it paints everything one colour, and on two beads or seven it is a joke."""
    from lammps_live.ui import bead_color_modes
    offered = {key: bead_color_modes(spec)
               for key, spec in registry.list_playgrounds()}
    assert offered["mesomem_bead"] == (), "one pair: nothing to choose between"
    for key in ("mesomem_patch", "mesomem_patch_torque", "mesomem_sheet",
                "mesomem_rod", "mesomem_polymer"):
        assert "cluster" not in offered[key], key
    for key in ("mesomem_assembly", "mesomem_remote"):
        assert offered[key][0] == "cluster", f"{key} should open in cluster"


def test_no_scene_offers_a_colouring_nothing_can_draw():
    """The offered list is filtered against the real one, so a typo in a
    playground cannot put a mode on the cycle that the renderer has no branch
    for."""
    from lammps_live.ui import BEAD_COLOR_MODES, bead_color_modes
    for key, spec in registry.list_playgrounds():
        assert set(bead_color_modes(spec)) <= set(BEAD_COLOR_MODES), key


def test_the_whole_system_energy_panel_is_off_where_it_says_the_same_thing():
    """On seven beads the pulled one takes part in half the bonds, so the box's
    breakdown is its own times roughly two: two panels, one fact."""
    lessons = {key: spec.lesson for key, spec in registry.list_playgrounds()}
    assert lessons["mesomem_patch"].system_energy is False
    assert lessons["mesomem_patch_torque"].system_energy is False
    assert lessons["mesomem_sheet"].system_energy is True


def test_the_over_informative_observables_are_gone():
    """A nematic order parameter, a membrane thickness and an area per particle are
    numbers nothing in the demo explains, so no scene puts one on the HUD. They
    still EXIST as observables -- tests and sweeps use them, and
    test_runtime.py's tension check reads nematic order directly."""
    from lammps_live.playground import observables
    banned = {"nematic_S", "thickness", "area_per_particle"}
    for key, pg in _offered():
        assert not banned & set(pg.observables), key
    assert all(observables.get(name) is not None for name in banned)
