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
"""

# name -> the callout's one line. Ordered as the sequence introduces them.
ELEMENTS = {
    "panel": "Controls: drag a slider to change the physics live "
             "(joystick: hat right, then the stick)",
    "energy": "The energy of the bead you hold, split into its three terms",
    "colour": "Change what the colours show: orientation, energy or cluster",
    "plots": "Live measurements of the whole system",
    "readings": "",                  # the observables HUD -- no callout of its own
    "status": "Simulated time since the start",
    "snellius": "This one runs live on Snellius, the Dutch national "
                "supercomputer at SURF",
}

ALL = frozenset(ELEMENTS)

# How long a callout stays up, and how much of that is its fade-out, in seconds.
CALLOUT_SECONDS = 9.0
CALLOUT_FADE = 1.0


def shown(lesson):
    """The set of element names `lesson` shows."""
    if lesson is None or getattr(lesson, "ui", None) is None:
        return ALL
    return frozenset(lesson.ui)


def introduced(lesson, previous_lesson):
    """The elements `lesson` shows that `previous_lesson` did not, in ELEMENTS
    order, skipping the ones with no callout text. `previous_lesson` None (the
    first scene, or one off the sequence) introduces nothing: the first scene's
    elements are the baseline, not news."""
    if lesson is None or previous_lesson is None:
        return ()
    new = shown(lesson) - shown(previous_lesson)
    return tuple(name for name in ELEMENTS if name in new and ELEMENTS[name])
