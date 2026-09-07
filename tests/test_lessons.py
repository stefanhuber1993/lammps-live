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
from lammps_live.playground.spec import Lesson, Thesis

# The claim is read off a projector, from the back of a room, in the two seconds
# before the presenter starts talking over it. That is the whole specification, and
# this is it in a number: past about twelve words it is a sentence being read
# rather than a line being taken in.
CLAIM_MAX_WORDS = 13


def _words(text):
    """Words, not tokens: this codebase writes a dash as a bare "--", and counting
    punctuation as a word would fail a line for having a dash in it."""
    return [w for w in text.split() if any(c.isalnum() for c in w)]


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

def test_the_claim_stays_short_enough_to_read_across_a_room():
    long = {key: len(_words(pg.lesson.claim))
            for key, pg in _offered()
            if len(_words(pg.lesson.claim)) > CLAIM_MAX_WORDS}
    assert not long, f"claims too long to take in at a glance: {long}"


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
    """The bookend, and it is deliberate: the opening claim is that all we kept of
    a hundred lipids was one arrow, and the closing one is that the arrow was
    enough. Break either half and the callback the talk is built around is gone.
    """
    keys = registry.bundled_keys()
    first = dict(_offered())[keys[0]].lesson
    last = dict(_offered())[keys[-1]].lesson
    assert "points" in first.claim and "which way" in first.claim
    assert "arrow" in last.claim and "enough" in last.claim


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


# ---- the thesis -----------------------------------------------------------

def test_the_thesis_appears_wherever_there_is_a_membrane_to_take_apart():
    """Not before. The button's whole force is that what collapses was visibly a
    membrane a second ago, which is not true of two beads or of seven."""
    lessons = {key: pg.lesson for key, pg in _offered()}
    has = {key for key, lesson in lessons.items() if lesson.thesis is not None}
    assert has == {"mesomem_sheet", "mesomem_assembly", "mesomem_rod",
                   "mesomem_remote", "mesomem_polymer"}


def test_the_thesis_zeroes_parameters_the_playground_actually_has():
    """A Thesis names live parameters, and a name that no playground declares would
    be a button that silently did nothing."""
    for key, pg in _offered():
        if pg.lesson.thesis is None:
            continue
        spec = dict(registry.list_playgrounds())[key]
        keys = {s.key for s in spec.extra_sliders}
        assert set(pg.lesson.thesis.params) <= keys, key


def test_the_thesis_is_the_isotropic_only_preset_by_another_route():
    """One definition of what "isotropic only" means, not two: the preset is a
    place to start and the button is a place to visit and come back from, and they
    have to agree about what is being removed."""
    for key, pg in _offered():
        thesis = pg.lesson.thesis
        if thesis is None or "isotropic_only" not in pg.presets:
            continue
        preset = pg.presets["isotropic_only"]
        assert set(thesis.params) == set(preset), key
        assert all(v == 0.0 for v in preset.values()), key


def test_the_thesis_says_what_is_missing_while_it_is_engaged():
    """A room that walked in halfway through sees a collapse and no explanation
    unless the screen carries one."""
    for key, pg in _offered():
        if pg.lesson.thesis is not None:
            assert pg.lesson.thesis.caption, key
            assert pg.lesson.thesis.engaged_label != pg.lesson.thesis.label, key


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
        if spec.lesson.thesis is not None:
            assert isinstance(spec.lesson.thesis, Thesis), key
