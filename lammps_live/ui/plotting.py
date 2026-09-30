"""Rolling time-series buffer and the generic line-plot drawer used for the
instrumentation panel (temperature/pressure/energy/RDF)."""
import math
from collections import deque

import pygame

from .scale import UI
from .theme import DIM_TEXT_COLOR, GRID_COLOR, PANEL_DIVIDER, TEXT_COLOR


class RollingHistory:
    """Fixed-length time series buffer, seconds-based (not frame-count) so
    the visible window doesn't change with fps."""

    def __init__(self, window_seconds, names):
        self.window_seconds = window_seconds
        self.t = deque()
        self.series = {name: deque() for name in names}
        # Each plot's y axis, by plot name (see axis_range).
        self.y_ranges = {}

    def reset(self):
        """Clear all buffered samples -- called on system switch, so a
        session's history doesn't leak visually into the next system's
        (different units/scale) plots. The axes go with them: a Reset is a new
        run, and it is allowed a new scale."""
        self.t.clear()
        for s in self.series.values():
            s.clear()
        self.y_ranges.clear()

    def axis_range(self, name, values, floor):
        """A y range for plot `name` that holds still.

        Fitting the axis to what is on screen every frame made the plots jump:
        the data sat still and the ruler under it moved, so a flat line
        twitched and a real change was indistinguishable from a rescale. Here
        the axis starts at `floor` -- a (lo, hi) that is a sensible span for
        this quantity -- and only ever WIDENS, in round steps, when the data
        leaves it; it never shrinks back until the history is reset. So it
        changes a handful of times in a run, each time to a number worth
        reading, and the trace moves while the axis does not.
        """
        lo, hi = self.y_ranges.get(name, floor)
        vals = [v for v in values if math.isfinite(v)]
        if vals:
            vlo, vhi = min(vals), max(vals)
            if vlo < lo or vhi > hi:
                lo, hi = _nice_range(min(lo, vlo), max(hi, vhi))
        self.y_ranges[name] = (lo, hi)
        return lo, hi

    def add(self, t, **values):
        self.t.append(t)
        for name, v in values.items():
            self.series[name].append(v)
        cutoff = t - self.window_seconds
        while self.t and self.t[0] < cutoff:
            self.t.popleft()
            for s in self.series.values():
                s.popleft()


def _nice_step(span):
    """The 1-2-5 step that cuts `span` into about four intervals."""
    raw = span / 4.0
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 5.0, 10.0):
        if raw <= m * mag:
            return m * mag
    return 10.0 * mag


def _nice_range(lo, hi):
    """(lo, hi) widened by a margin and snapped out to round numbers, so a
    widened axis is not immediately outgrown again by the next sample."""
    span = max(hi - lo, 1e-9)
    lo, hi = lo - 0.15 * span, hi + 0.15 * span
    step = _nice_step(hi - lo)
    return math.floor(lo / step) * step, math.ceil(hi / step) * step


def draw_plot(screen, font, rect, title, y_label, x_vals, series, y_range=None, ref_lines=()):
    """series: list of (label, color, values). y_range: (lo, hi) or None to
    auto-fit with padding. ref_lines: list of (value, color, label)."""
    pygame.draw.rect(screen, (30, 30, 36), rect)
    pygame.draw.rect(screen, PANEL_DIVIDER, rect, width=UI.w(1))
    title_surf = font.render(title, True, TEXT_COLOR)
    screen.blit(title_surf, (rect.x + UI(6), rect.y + UI(4)))

    plot_rect = pygame.Rect(rect.x + UI(44), rect.y + UI(22),
                            rect.width - UI(56), rect.height - UI(34))

    all_vals = [v for _, _, vals in series for v in vals]
    for v, _, _ in ref_lines:
        all_vals.append(v)
    if y_range is not None:
        lo, hi = y_range
    elif all_vals:
        lo, hi = min(all_vals), max(all_vals)
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
        pad = (hi - lo) * 0.1
        lo, hi = lo - pad, hi + pad
    else:
        lo, hi = 0.0, 1.0

    for frac in (0.0, 0.5, 1.0):
        gy = plot_rect.bottom - frac * plot_rect.height
        pygame.draw.line(screen, GRID_COLOR, (plot_rect.x, gy),
                         (plot_rect.right, gy), UI.w(1))
        val_lbl = font.render(f"{lo + frac * (hi - lo):.3g}", True, DIM_TEXT_COLOR)
        screen.blit(val_lbl, (rect.x + UI(2), gy - UI(6)))

    for value, color, label in ref_lines:
        if hi == lo:
            continue
        gy = plot_rect.bottom - (value - lo) / (hi - lo) * plot_rect.height
        for dash_x in range(plot_rect.x, plot_rect.right, UI(6)):
            pygame.draw.line(screen, color, (dash_x, gy), (dash_x + UI(3), gy),
                             UI.w(1))
        if label:
            lbl = font.render(label, True, color)
            screen.blit(lbl, (plot_rect.right - lbl.get_width(), gy - UI(12)))

    if not x_vals or len(x_vals) < 2:
        y_label_surf = font.render(y_label, True, DIM_TEXT_COLOR)
        screen.blit(y_label_surf, (plot_rect.x, rect.bottom - UI(14)))
        return
    x_lo, x_hi = x_vals[0], x_vals[-1]
    x_span = max(1e-9, x_hi - x_lo)

    for label, color, vals in series:
        if len(vals) < 2:
            continue
        pts = []
        n = min(len(x_vals), len(vals))
        for i in range(len(x_vals) - n, len(x_vals)):
            x = plot_rect.x + (x_vals[i] - x_lo) / x_span * plot_rect.width
            yv = vals[i - (len(x_vals) - n)]
            y = plot_rect.bottom - (yv - lo) / (hi - lo) * plot_rect.height if hi != lo else plot_rect.centery
            # Pinned to the frame: a spike past a fixed axis runs along its edge
            # rather than off the panel (see Playground.plot_ranges).
            y = min(max(y, plot_rect.top), plot_rect.bottom)
            pts.append((x, y))
        if len(pts) >= 2:
            pygame.draw.lines(screen, color, False, pts, UI.w(2))

    legend_x = plot_rect.x
    for label, color, _ in series:
        pygame.draw.line(screen, color, (legend_x, rect.bottom - UI(8)),
                         (legend_x + UI(14), rect.bottom - UI(8)), UI.w(3))
        lbl = font.render(label, True, DIM_TEXT_COLOR)
        screen.blit(lbl, (legend_x + UI(18), rect.bottom - UI(16)))
        legend_x += UI(18) + lbl.get_width() + UI(12)

    y_label_surf = font.render(y_label, True, DIM_TEXT_COLOR)
    screen.blit(y_label_surf, (plot_rect.x,
                               rect.y + UI(4) + title_surf.get_height()))
