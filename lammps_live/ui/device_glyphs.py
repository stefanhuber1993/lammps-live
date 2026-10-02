"""The joystick's buttons, drawn as themselves.

Every on-screen button that a device button also fires carries a picture of that
device button -- not a number in a box, but its SHAPE, so the eye can go from the
screen to the hand without reading anything:

  1  the trigger      a curved blade, the number sitting in its hollow where the
                      finger goes
  2  the thumb lens   a flat lens-shaped button on top of the stick
  3  4                small round dark-red buttons (8 mm), previous / next scene
  5 6 7 8             big round grey flat buttons (15 mm), this scene's moves

The SIZES ARE TO SCALE with each other where it reads: the red pair is about half
the width of the grey block, as it is on the device. The colours are the device's
too, with a light rim so a black trigger still reads on the dark button plate.

Shapes are drawn 4x oversized and smoothscaled down, because pygame's own circles
and polygons have no antialiasing and a jagged lens is the one thing that would
make these look cheap. Each (button, scale) is drawn once and cached.
"""
import pygame

from .. import config
from .scale import UI

# Shape size of each kind at UI scale 1, in px (width, height).
GLYPH_SIZE = {
    "trigger": (15, 30),     # the blade alone; the number sits to its right
    "lens": (32, 19),
    "round_small": (19, 19),
    "round_big": (29, 29),
}
# The width every glyph slot gets, so labels on a row line up whatever is in the
# slot. The trigger's slot holds the blade plus its number.
GLYPH_SLOT_W = 40

# Device colours. Fill, rim, number.
_COLOURS = {
    # The stick's black plastic, lifted to charcoal so it stands off the dark well
    # it sits in; true black would be a hole with a rim.
    "trigger": ((66, 71, 80), (176, 188, 206), None),
    "lens": ((66, 71, 80), (176, 188, 206), (240, 244, 250)),
    "round_small": ((128, 18, 24), (214, 92, 92), (255, 236, 236)),
    "round_big": ((150, 155, 162), (214, 218, 224), (18, 20, 24)),
}
_SS = 4          # supersampling factor
_RIM = 1.6       # rim width at UI scale 1, px

_cache = {}


def kind(button):
    """Which physical shape device button `button` has."""
    if button == config.JOYSTICK_PLAY_PAUSE_BUTTON:
        return "trigger"
    if button == config.JOYSTICK_RESET_BUTTON:
        return "lens"
    if button in (config.JOYSTICK_PREV_PLAYGROUND_BUTTON,
                  config.JOYSTICK_NEXT_PLAYGROUND_BUTTON):
        return "round_small"
    return "round_big"


def _disc(size, centre, radius, colour):
    surf = pygame.Surface(size, pygame.SRCALPHA)
    pygame.draw.circle(surf, colour, centre, radius)
    return surf


def _shape(k, w, h, inset, colour):
    """The shape `k` filling a (w, h) supersampled canvas, shrunk by `inset` px
    all round, in `colour`."""
    size = (w, h)
    if k in ("round_small", "round_big"):
        return _disc(size, (w / 2, h / 2), min(w, h) / 2 - inset, colour)
    if k == "lens":
        # Two discs meeting at the lens's points, which sit on its long axis:
        # a chord of length w at distance d from each centre, with the lens's
        # half-thickness h/2 = r - d.
        half_w, half_h = w / 2 - inset, h / 2 - inset
        r = (half_w ** 2 + half_h ** 2) / (2 * half_h)
        d = r - half_h
        top = _disc(size, (w / 2, h / 2 + d), r, colour)
        top.blit(_disc(size, (w / 2, h / 2 - d), r, colour), (0, 0),
                 special_flags=pygame.BLEND_RGBA_MIN)
        return top
    # The trigger: a crescent, one disc minus a bigger one set off to the right,
    # so the blade bows to the left and its hollow faces the number. Both circles
    # pass through the two tips (x = tip, a whisker inside the top and bottom
    # edges); the outer one also through the blade's back at x = 0, the inner one
    # through its front at x = thick.
    pad = h * 0.03
    half = h / 2 - pad
    tip, thick = w * 0.86, w * 0.56

    def through_tips(x_mid):
        # Centre (and radius) of the circle through both tips and (x_mid, h/2).
        dx = tip - x_mid
        r = (dx * dx + half * half) / (2 * dx)
        return x_mid + r, r

    cx_out, r_out = through_tips(0.0)
    cx_in, r_in = through_tips(thick)
    surf = _disc(size, (cx_out, h / 2), r_out - inset, colour)
    pygame.draw.circle(surf, (0, 0, 0, 0), (cx_in, h / 2), r_in + inset)
    return surf


def glyph(button):
    """The device button `button`, drawn at its size for the current UI scale: a
    Surface with alpha. Its number is NOT on it (see draw)."""
    k = kind(button)
    key = (k, UI.factor)
    surf = _cache.get(key)
    if surf is None:
        bw, bh = GLYPH_SIZE[k]
        w, h = max(1, UI(bw)), max(1, UI(bh))
        sw, sh = w * _SS, h * _SS
        fill, rim, _ = _COLOURS[k]
        big = _shape(k, sw, sh, 0, rim)
        big.blit(_shape(k, sw, sh, UI.f(_RIM) * _SS, fill), (0, 0))
        surf = pygame.transform.smoothscale(big, (w, h))
        _cache[key] = surf
    return surf


def draw(screen, button, slot, font, text_colour):
    """Draw device button `button` centred in `slot` (a Rect), with its number on
    it -- or, for the trigger, in its hollow, in `text_colour` (the plate's own
    text colour, since there it sits on the plate rather than on the button)."""
    k = kind(button)
    g = glyph(button)
    num_colour = _COLOURS[k][2] or text_colour
    num = font.render(str(button), True, num_colour)
    if k == "trigger":
        # Blade and number side by side, centred together in the slot.
        gap = UI(2)
        total = g.get_width() + gap + num.get_width()
        x = slot.centerx - total // 2
        screen.blit(g, (x, slot.centery - g.get_height() // 2))
        screen.blit(num, num.get_rect(midleft=(x + g.get_width() + gap,
                                               slot.centery)))
        return
    rect = g.get_rect(center=slot.center)
    screen.blit(g, rect)
    # Optical centre: digits sit a pixel high in most fonts' boxes.
    screen.blit(num, num.get_rect(center=(rect.centerx, rect.centery + UI(1))))
