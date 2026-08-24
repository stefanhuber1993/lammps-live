"""Small reusable pygame UI widgets."""
import math

import pygame

from .scale import UI
from .theme import (
    BUTTON_ACTIVE_BG, BUTTON_ACTIVE_TEXT, BUTTON_BG, BUTTON_BORDER, BUTTON_TEXT,
    FOCUS_COLOR, FOCUS_WIDTH, MELT_MARK_COLOR, OPTIMUM_MARK_COLOR, SLIDER_HANDLE,
    SLIDER_HANDLE_ACTIVE, SLIDER_TRACK, TEXT_COLOR,
)


class Button:
    """A simple rectangular click button. Geometry + label only; the caller owns
    the action and decides when it reads as 'active' (highlighted). Used for the
    Play / Pause / Reset playback controls."""

    def __init__(self, name, label):
        self.name = name
        self.label = label
        self.rect = pygame.Rect(0, 0, 0, 0)

    def draw(self, screen, font, active=False, focused=False):
        """`active` is which button the caller recommends; `focused` is which one
        the joystick is on.

        Two marks rather than one, because they answer different questions and can
        land on different buttons -- "the one you probably want" is the panel's
        opinion and does not move, "the one a press would hit right now" is the
        hand's and does. The focus ring is the same cyan frame the viewport and the
        slider rows get (see control_focus.py), drawn just outside the button so it
        does not eat into the label.
        """
        bg = BUTTON_ACTIVE_BG if active else BUTTON_BG
        fg = BUTTON_ACTIVE_TEXT if active else BUTTON_TEXT
        pygame.draw.rect(screen, bg, self.rect, border_radius=UI(6))
        pygame.draw.rect(screen, BUTTON_BORDER, self.rect, width=UI.w(1),
                         border_radius=UI(6))
        if focused:
            pygame.draw.rect(screen, FOCUS_COLOR, self.rect.inflate(UI(6), UI(6)),
                             width=UI.w(FOCUS_WIDTH), border_radius=UI(8))
        surf = font.render(self.label, True, fg)
        screen.blit(surf, surf.get_rect(center=self.rect.center))

    def hit(self, pos):
        return self.rect.collidepoint(pos)


class Slider:
    """Horizontal, mouse-draggable value slider. Geometry-only + drag state
    -- callers own the semantics of what the value drives."""

    def __init__(self, rect, vmin, vmax, value, label, fmt="{:.3f}", unit="",
                 optimum=None, advanced=False):
        self.rect = pygame.Rect(rect)
        self.vmin, self.vmax = vmin, vmax
        self.value = max(vmin, min(vmax, value))
        self.label = label
        self.fmt = fmt
        self.unit = unit
        self.optimum = optimum
        self.advanced = advanced
        self.dragging = False

    @classmethod
    def from_spec(cls, rect, spec):
        """Build from an mdsystem.SliderSpec."""
        return cls(rect, spec.vmin, spec.vmax, spec.default, spec.label,
                   fmt=spec.fmt, unit=spec.unit, optimum=spec.optimum,
                   advanced=spec.advanced)

    def reset(self, spec):
        """Re-apply a (possibly different, on system switch) SliderSpec's
        range/default/label in place, so callers don't have to juggle new
        Slider instances or re-wire event handling."""
        self.vmin, self.vmax = spec.vmin, spec.vmax
        self.value = spec.default
        self.label = spec.label
        self.fmt = spec.fmt
        self.unit = spec.unit
        self.optimum = spec.optimum
        self.advanced = spec.advanced
        self.dragging = False

    def _frac(self):
        return (self.value - self.vmin) / (self.vmax - self.vmin)

    def _handle_pos(self):
        return (self.rect.x + self._frac() * self.rect.width, self.rect.centery)

    def _set_from_x(self, x):
        frac = max(0.0, min(1.0, (x - self.rect.x) / self.rect.width))
        self.value = self.vmin + frac * (self.vmax - self.vmin)

    def handle_event(self, event):
        """Returns True if this slider consumed the event."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            hx, hy = self._handle_pos()
            near_handle = math.hypot(event.pos[0] - hx, event.pos[1] - hy) < UI(12)
            on_track = self.rect.collidepoint(event.pos)
            if near_handle or on_track:
                self.dragging = True
                self._set_from_x(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.dragging = False
                return True
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_x(event.pos[0])
            return True
        return False

    def nudge(self, delta):
        self.value = max(self.vmin, min(self.vmax, self.value + delta))

    def _mark_x(self, value):
        return self.rect.x + (value - self.vmin) / (self.vmax - self.vmin) * self.rect.width

    def focus_rect(self):
        """The whole row this slider occupies -- label above, track, and the tick
        captions below -- as one rectangle, for the joystick-focus frame. Derived
        from the track rect rather than stored, since draw_panel re-lays the track
        out every frame and the frame has to follow it."""
        return pygame.Rect(self.rect.x - UI(6), self.rect.y - UI(22),
                           self.rect.width + UI(12), UI(42))

    def draw(self, screen, font, mark_value=None, mark_label=None, focused=False):
        # `focused` = the joystick is driving THIS slider (see control_focus.py).
        # Framed in the same cyan as the viewport frame, and the handle takes the
        # colour too: at a glance across a room the frame says which row, the
        # handle says where in it the value sits.
        if focused:
            pygame.draw.rect(screen, FOCUS_COLOR, self.focus_rect(),
                             width=UI.w(FOCUS_WIDTH), border_radius=UI(6))
        pygame.draw.line(screen, SLIDER_TRACK, (self.rect.x, self.rect.centery),
                          (self.rect.right, self.rect.centery), UI.w(4))
        # Recommended-value ("optimum") marker: a distinct-colored tick with an
        # "opt" caption below, so the paper's sweet spot is easy to find again.
        if self.optimum is not None and self.vmax > self.vmin:
            ox = self._mark_x(self.optimum)
            pygame.draw.line(screen, OPTIMUM_MARK_COLOR, (ox, self.rect.top - UI(3)),
                             (ox, self.rect.bottom + UI(3)), UI.w(2))
            cap = font.render("opt", True, OPTIMUM_MARK_COLOR)
            screen.blit(cap, (ox - cap.get_width() / 2, self.rect.bottom + UI(4)))
        if mark_value is not None:
            mx = self.rect.x + (mark_value - self.vmin) / (self.vmax - self.vmin) * self.rect.width
            pygame.draw.line(screen, MELT_MARK_COLOR, (mx, self.rect.top - UI(3)),
                             (mx, self.rect.bottom + UI(3)), UI.w(2))
            if mark_label:
                lbl = font.render(mark_label, True, MELT_MARK_COLOR)
                screen.blit(lbl, (mx - lbl.get_width() / 2, self.rect.bottom + UI(4)))
        hx, hy = self._handle_pos()
        if self.dragging:
            color = SLIDER_HANDLE_ACTIVE
        elif focused:
            color = FOCUS_COLOR
        else:
            color = SLIDER_HANDLE
        pygame.draw.circle(screen, color, (int(hx), int(hy)), UI(8))
        text = font.render(f"{self.label}: {self.fmt.format(self.value)}{self.unit}", True, TEXT_COLOR)
        screen.blit(text, (self.rect.x, self.rect.y - UI(18)))


class TextField:
    """A one-line text entry, for the only thing in this app that needs typing:
    the answer to whatever the cluster's login asks for (see remote_panel.py).

    Masked by default, because that answer is a password or a one-time code and it
    is shown on a screen someone may well be presenting from. The value never
    leaves this object except when the caller reads it on submit.
    """

    def __init__(self, masked=True, placeholder=""):
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.value = ""
        self.masked = masked
        self.placeholder = placeholder
        self.active = True

    def clear(self):
        self.value = ""

    def handle_event(self, event):
        """Returns "submit" on Enter, "changed" on an edit, or None.

        Only KEYDOWN is consumed, and only while active, so the caller can leave
        every other shortcut in the app working by checking the return value.
        """
        if not self.active or event.type != pygame.KEYDOWN:
            return None
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            return "submit"
        if event.key == pygame.K_BACKSPACE:
            self.value = self.value[:-1]
            return "changed"
        if event.key == pygame.K_u and (event.mod & pygame.KMOD_CTRL):
            self.value = ""            # the terminal habit, and it works here
            return "changed"
        char = event.unicode
        # Printable only: a stray control character in a one-time code is a login
        # failure that looks like a wrong code.
        if char and char.isprintable():
            self.value += char
            return "changed"
        return None

    def draw(self, screen, font, blink=True):
        pygame.draw.rect(screen, (14, 14, 18), self.rect, border_radius=UI(4))
        border = SLIDER_HANDLE_ACTIVE if self.active else BUTTON_BORDER
        pygame.draw.rect(screen, border, self.rect, width=UI.w(1),
                         border_radius=UI(4))
        if self.value:
            shown = ("*" * len(self.value)) if self.masked else self.value
            color = TEXT_COLOR
        else:
            shown = self.placeholder
            color = BUTTON_BORDER
        surf = font.render(shown, True, color)
        screen.blit(surf, (self.rect.x + UI(8),
                           self.rect.centery - surf.get_height() // 2))
        if self.active and blink:
            caret_x = (self.rect.x + UI(8)
                       + (surf.get_width() if self.value else 0) + UI(2))
            pygame.draw.line(screen, TEXT_COLOR, (caret_x, self.rect.y + UI(6)),
                             (caret_x, self.rect.bottom - UI(6)), UI.w(1))
