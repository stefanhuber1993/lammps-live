"""Progressive disclosure: which pieces of the interface a scene shows, and the
arrow that introduces a piece the first time the sequence reaches it.

The opening scene is two beads and the reading between them -- no side panel, no
status line, no plots. Every later scene adds what it needs and nothing more,
declared per playground as `Lesson.ui` (a tuple of the names below). When a scene
shows an element that the scene BEFORE it in the offered order did not, the
renderer draws a callout on it -- a highlight, an arrow and one line saying what
it is -- for the first few seconds. "Before it in the order", not "the scene you
came from": the explanation belongs to the scene, so it reads the same however
the demo got there.

`Lesson.ui = None` means everything, which is what a scene without a lesson (the
shelved atomistic classics, anyone's own file) has always had.

TWO NAMES ARE NOT DRAWN AS PANEL PIECES:

  hero   the row of big-move buttons under the scene. Not listed in `Lesson.ui`:
         a scene shows it exactly when it declares hero knobs, so it is derived
         from `Lesson.hero_knobs` -- two places saying the same thing would drift.
  slice  the thrust lever's view cut, and the small gauge in the corner of the
         scene that says it is there. Listed in `Lesson.ui` like the rest, and
         HIDDEN MEANS NOT OFFERED: on a scene without it the lever cuts nothing.
         A slab through two beads or a flat sheet shows less, not more.
"""

from .. import config

# name -> the callout's one line, or (mouse line, joystick line) where the two
# devices reach it differently -- telling someone with a stick to "drag", or
# someone with a mouse to use "the hat", is a line they cannot act on. Ordered as
# the sequence introduces them. Read through `text`.
ELEMENTS = {
    "panel": ("Controls: drag a slider to change the physics live",
              "Controls: hat right to get here, then the stick changes the "
              "physics live"),
    "energy": "The energy of the bead you hold, split into its three terms",
    "colour": "Change what the colours show: orientation, energy or cluster",
    "plots": "Live measurements of the whole system",
    "readings": "",                  # the observables HUD -- no callout of its own
    "hero": ("The big move on this scene: click it, and click again to undo",
             f"The big move on this scene: button "
             f"{config.JOYSTICK_HERO_FIRST_BUTTON} on the back of the stick. "
             f"Press again to undo"),
    "status": "Simulated time since the start",
    "slice": "Thrust lever: slice the box open (either end = whole box)",
    "snellius": "This one runs live on Snellius, the Dutch national "
                "supercomputer at SURF",
}

ALL = frozenset(ELEMENTS)


def text(name, joystick=False):
    """The callout line for `name`, worded for the device in hand."""
    line = ELEMENTS.get(name, "")
    if isinstance(line, tuple):
        return line[1] if joystick else line[0]
    return line

# How long a callout stays up, and how much of that is its fade-out, in seconds.
CALLOUT_SECONDS = 9.0
CALLOUT_FADE = 1.0


def shown(lesson):
    """The set of element names `lesson` shows."""
    if lesson is None or getattr(lesson, "ui", None) is None:
        return ALL
    names = frozenset(lesson.ui) - {"hero"}
    return names | {"hero"} if getattr(lesson, "hero_knobs", ()) else names


def introduced(lesson, previous_lesson):
    """The elements `lesson` shows that `previous_lesson` did not, in ELEMENTS
    order, skipping the ones with no callout text. `previous_lesson` None (the
    first scene, or one off the sequence) introduces nothing: the first scene's
    elements are the baseline, not news."""
    if lesson is None or previous_lesson is None:
        return ()
    new = shown(lesson) - shown(previous_lesson)
    return tuple(name for name in ELEMENTS if name in new and ELEMENTS[name])
