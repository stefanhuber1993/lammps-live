"""How big the 2D UI is drawn: one global multiplier.

Every pixel size in the 2D layer -- font sizes, paddings, row heights, panel
width, stroke widths -- is written as the number that looked right on an ordinary
~100 dpi display, and passed through `UI` here on its way to pygame.

WHY THIS EXISTS. The window is an OpenGL surface and pygame 2 does not ask SDL
for a Retina-aware one (SDL_WINDOW_ALLOW_HIGHDPI is neither set nor exposed), so
one pixel we draw is one *point*, not one physical pixel. On a display running a
HiDPI mode (a 2x backing scale) the OS therefore magnifies everything we draw by
2 and the whole GUI -- text worst of all, since a size-18 font is only 12 px tall
before magnification -- goes soft. Run the same screen at its native resolution
instead and nothing is resampled, but then the UI is drawn at half the physical
size it was designed for: sharp and unreadably small.

Scaling here is the fix for the second half of that: sizes are multiplied BEFORE
anything is rasterized, so the fonts are rendered at their final size and every
line lands on a whole pixel. Sharp, and the right size.

`UI.factor == 1.0` reproduces the original layout exactly (`UI(n) == n` for
integer n), so the default costs nothing.
"""
import os

import pygame

# SURF's heading typeface (Nunito, SIL Open Font License), bundled so the titles
# and the Snellius badge look the same on any machine the demo is run from.
BRAND_FONT_PATH = os.path.join(os.path.dirname(__file__), "assets",
                               "Nunito-VariableFont_wght.ttf")


class _UIScale:
    """The multiplier, plus the three ways a pixel size gets used."""

    def __init__(self, factor=1.0):
        self.factor = float(factor)

    def __call__(self, n):
        """A hardcoded pixel size -> physical pixels, snapped to a whole one."""
        return int(round(n * self.factor))

    def f(self, n):
        """...unsnapped, for lengths that are compared or interpolated."""
        return n * self.factor

    def w(self, n):
        """...as a stroke width, which must stay at least one pixel visible."""
        return max(1, int(round(n * self.factor)))

    def font(self, size, bold=False):
        """A UI font of a size quoted at scale 1 -- rasterized at final size."""
        return pygame.font.SysFont(None, self(size), bold=bold)

    def brand_font(self, size, bold=False):
        """The same, in the bundled brand face -- falling back to the UI font if
        the file is missing, so a stripped install still runs."""
        try:
            font = pygame.font.Font(BRAND_FONT_PATH, self(size))
        except (OSError, FileNotFoundError):
            return self.font(size, bold=bold)
        font.set_bold(bold)
        return font


UI = _UIScale()


def set_ui_scale(factor):
    """Set the global scale (clamped to a sane range) and return what was used."""
    UI.factor = max(0.5, min(4.0, float(factor)))
    return UI.factor


def auto_ui_scale(desktop_size):
    """The scale a screen of this size wants, when the user names none.

    `desktop_size` is what SDL reports, which is already in the units we draw in:
    a HiDPI-scaled display reports its *point* size (2560x1440 on this monitor at
    "looks like 1440p"), where the classic sizes are right and scaling up would
    double an already-magnified UI. A screen reporting a full 4K height is one
    with no scaling of its own, where every size we quote lands half as big as it
    was meant to -- so those get 2.
    """
    return 2.0 if desktop_size[1] >= 1800 else 1.0
