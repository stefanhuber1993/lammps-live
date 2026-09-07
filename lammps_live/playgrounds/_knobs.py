"""The hero knobs the MesoMem playgrounds share.

A hero knob is the one big move worth making on a scene, written down as a button
(see playground/spec.py's HeroKnob). Two of them come up on more than one scene,
so they live here rather than being typed out again in each file with the numbers
slowly drifting apart.

The leading underscore keeps this file out of playground discovery: `registry`
skips module names starting with one, so this is importable by the playgrounds
next to it without ever being offered as a scene or walked by `--verify`.
"""
from ..playground import HeroKnob

# What the membrane loses when it stops caring which way its beads face.
#
# THE SAME PARAMETERS THE `isotropic_only` PRESET SETS, and it has to stay that
# way: the preset is how you start a run without orientation and the knob is how
# you take it away mid-run, and two definitions of "isotropic only" that disagree
# would be worse than either. tests/test_lessons.py pins them together.
#
# wc goes with them. With both moduli at zero the two orientational terms are
# already exactly zero whatever the cutoff is, so this changes no physics -- but it
# collapses the wc ring in the two-bead scene and reads honestly in the panel,
# which is where the audience is looking for the answer to "what did you take
# away".
NO_ORIENTATION = HeroKnob(
    label="Remove orientation",
    engaged_label="Restore orientation",
    caption="k_tilt = 0, k_splay = 0. Only the plain attraction is left.",
    params={"k_tilt": 0.0, "k_splay": 0.0, "wc": 0.0},
)

# Where the membrane stops being a solid and starts being a liquid, which is what
# a real bilayer is and the one thing the default temperature hides.
#
# 0.2 IS PICKED, NOT ROUNDED. Both scenes that offer this run the dial from 0 to
# 0.5 with `melt_temp` at 0.3, and above 0.3 the sheet is no longer a membrane at
# all (the barostat then inflates the cell without bound -- see HexSheet). At 0.2
# beads exchange neighbours and long-wavelength undulations are visible while the
# sheet still holds together, and the tension-free cell there is about 11% wider in
# area than the one it was built at, which the running barostat reaches in a few
# hundred frames. Warm, and a long way from broken.
#
# The caption carries both numbers on purpose. "Heat" on its own is a feeling; the
# number and the melting point it is below make it a measurement, which is what
# somebody in the audience needs in order to ask a sensible question about it.
HEAT_TEMPERATURE = 0.2

HEAT = HeroKnob(
    label="Heat",
    engaged_label="Cool",
    caption=f"T = {HEAT_TEMPERATURE:.2f}, up from 0.001. Melting is at 0.30, "
            f"so it flows and still holds together.",
    temperature=HEAT_TEMPERATURE,
)
