"""Keeping the picture alive between the frames a slow wire actually delivers.

Visuals only, and the sibling of smoothing.py: both filter the DRAWN state and
neither is allowed anywhere near a measurement. Where the smoother takes motion
OUT (thermal rattle the eye does not want), this puts motion BACK IN (rattle the
wire did not have the bandwidth to send).

WHAT IT IS FOR. The remote playground streams 50k beads from a cluster GPU, and
at 60 fps that is 19.5 MB/s over an SSH tunnel. Dropping the wire to 20 fps cuts
it to 6.5 -- but then the client holds each frame for three drawn frames, and a
scene that was rattling at 60 Hz becomes a 20 Hz slideshow. The stutter is far
more visible than the bandwidth saving is worth, because thermal motion is
exactly the kind of signal the eye reads as "frozen" the moment it steps.

WHY NOT JUST PREDICT THE MISSING FRAMES. Because at these temperatures they are
not predictable. Measured on a real 1500-bead mesomem trajectory, replaying a
20 Hz wire against the 60 Hz truth (mean per-bead error, in screen pixels):

    scheme                                   smoothing off    smoothing tau=1.2
    hold the newest frame (a slideshow)         10.4 px            5.0 px
    interpolate between the last two             5.0 px            1.0 px
    extrapolate from the last two               11.2 px            2.8 px

Extrapolation is WORSE THAN FREEZING with the smoothing off, and the reason is
the whole design of this file: between wire frames the motion is dominated by the
thermal rattle, which is uncorrelated frame to frame, so a velocity estimated
from two samples is a random number and integrating it doubles the error instead
of reducing it. Only interpolation beats holding -- and it beats it by bracketing
the truth, which costs a whole wire frame of latency (50 ms at 20 Hz), paid on
every slider.

So: the missing motion cannot be recovered without latency, and this file does
not pretend to. IT IS A COSMETIC DEVICE, and it should be read as one. It does
not reduce the error against the true trajectory -- it slightly increases it.
What it buys is that the picture never looks frozen, at zero added latency, which
for a scene whose subject is a slow collective rearrangement is the trade that
matters: the aggregates still advance in 20 Hz steps, but they do it under a
surface that is moving at 60.

THE MODEL IS THE RIGHT ONE, THOUGH. An overdamped Langevin particle's excursion
about a slowly-moving centre IS an Ornstein-Uhlenbeck process -- a mean-reverting
random walk -- so that is what this synthesises, rather than the per-frame white
noise that would read as television static. Two parameters:

  * the AMPLITUDE is estimated live from the frames that do arrive, not declared
    per playground. That estimator is what makes the effect go quiet on its own
    exactly where it should: at temperature zero, while the run is paused, and on
    a system that has stopped rearranging. What it targets is the per-DRAWN-frame
    step, not the excursion, and the difference matters -- see `_sigma_for`,
    which is the one piece of arithmetic in this file worth reading.
  * the CORRELATION TIME is in WALL time, not simulated time, because what is
    being reproduced is a shimmer the viewer sees at the rate their screen
    refreshes. Two render frames at 60 Hz, which is also roughly where the real
    rattle's own autocorrelation sits (0.28 to 0.64 at lag one, measured).

WHAT IT DOES NOT HIDE, and the honest limit of a zero-latency scheme. The motion
a slow wire loses is not MISSING, it is LUMPY: a 20 Hz frame carries three frames'
worth of displacement and lands all of it on one drawn frame in three. So filling
the other two to full motion necessarily leaves that one moving more than the
rest, and the picture ends up carrying ~1.4x the true motion overall -- an excess
which IS the residual stutter, now under a surface that never freezes rather than
on top of one that does.

Removing the pulse means easing toward each arriving frame instead of snapping to
it, which is latency again -- less than interpolation's full wire frame, but not
zero, and zero is the requirement this file was written to. Both properties are
asserted in tests/test_jitter.py, so neither can quietly change.

THIS IS SIZED FOR SMOOTHING OFF, which is how the demo is actually watched. That
is the demanding case and it is worth being explicit about why: the Smoothing
slider's whole job is to remove the thermal rattle, so with it up the wire could
drop to 20 Hz and barely show it (5.0 px of error, against 10.4 with the slider
at zero). With it at zero the rattle is the majority of what is on screen at 60
Hz, and holding each frame for three of them is immediately visible. So the
default amplitude below reproduces the motion the wire drops rather than hedging
under it.

The jitter is nonetheless added BEFORE the smoother, so that turning the slider
up removes the synthetic rattle exactly as it removes the real one. The two are
the ends of one dial rather than two effects fighting for the same pixels, and
nothing about the ordering needs revisiting if the slider is ever used.
"""
from dataclasses import replace

import numpy as np

from .state import normalize_rows

# The slider, in units of "as much motion as a full-rate wire would have shown".
# 1.0 is the honest setting -- the frames BETWEEN arrivals, the ones that would
# otherwise be frozen, carry the same amount of movement they would have if every
# frame had been sent, which with the Smoothing slider at zero (where it lives)
# is what stops the scene reading as colder than the thermostat it is running at.
#
# It is meaningful as a ratio only because the amplitude is derived from a
# measured per-frame step rather than from an excursion; see `_sigma_for`.
#
# THE DEFAULT IS NONETHELESS ZERO, i.e. the effect ships OFF. Everything above is
# still true of what it does; what is no longer taken for granted is that the
# demo needs it. The wire is fast enough that the stutter it was written to hide
# is small, and a cosmetic rattle on a scene whose whole claim is "this is
# running now, not a recording" is a thing to be asked for rather than assumed.
# The slider is still there, still advanced, so turning it back up mid-demo is
# one drag -- and 1.0 is what it was tuned at, should this be reverted.
DEFAULT_JITTER = 0.0
# The slider's top end. Past 1 the beads visibly move more than the real ones do,
# which is the point at which the effect stops being invisible -- useful for
# seeing what it is doing, and for a demo room that wants the scene livelier than
# life.
JITTER_MAX = 2.0
# Correlation time of the synthetic rattle, in WALL seconds -- about two frames
# at 60 Hz. Shorter reads as static, longer as a slow swimming motion that the
# real thermal noise does not have.
TAU_WALL = 0.033
# Estimator smoothing: the amplitude is re-measured on every received frame, and
# a per-frame median is noisy enough that using it raw would make the liveliness
# itself flicker. One EMA over roughly half a second of wire frames.
AMPLITUDE_ALPHA = 0.15
# Anything past this many standard deviations is clamped. An OU process reaches
# it about three times in a million bead-frames, so this is not shaping the
# distribution -- it is a guard against a single bead being flung a visible
# distance by an outlier draw on the one frame a viewer happens to be looking.
CLAMP_SIGMAS = 3.0


class RattleFill:
    """A per-bead Ornstein-Uhlenbeck wobble, sized from the frames that arrive.

    `observe` is called once per RECEIVED frame and updates the amplitude
    estimate; `apply` is called once per DRAWN frame and advances the process by
    that frame's wall-clock slice. The two rates are independent, which is the
    entire point of the class.

    Beads are tracked in the stable id order every readout already hands back; a
    change in count reseeds, exactly as the smoother does.
    """

    def __init__(self, seed=None):
        self._rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """Forget the wobble and the amplitude estimate."""
        self._offset = None            # (N, 3) float32, the current excursion
        self._dir_offset = None        # (N, 3) float32, same for the directors
        # The MEDIAN LENGTH of a per-bead step between two received frames, not a
        # standard deviation -- `_sigma_for` is where it becomes one, and the
        # distinction is the bug this file was written around once already.
        self._step = 0.0
        self._dir_step = 0.0
        self._previous = None          # the last observed positions
        self._previous_dirs = None

    @property
    def active(self):
        return self._offset is not None

    def observe(self, state, dt_ignored=None):
        """Take in one RECEIVED frame: re-measure how much the wire is dropping.

        The measurement is a MEDIAN, not a mean, because a single bead that has
        just been remapped across the cell (or, on an unstable run, sent to
        infinity) would drag a mean into a visible over-estimate for the several
        seconds the estimator averages over.
        """
        pos = np.asarray(state.positions, dtype=float)
        if not len(pos) or not np.isfinite(pos).all():
            self.reset()
            return
        previous, self._previous = self._previous, pos.copy()
        dirs = None if state.directors is None else np.asarray(state.directors,
                                                               dtype=float)
        previous_dirs, self._previous_dirs = self._previous_dirs, (
            None if dirs is None else dirs.copy())
        if previous is None or len(previous) != len(pos):
            return
        delta = pos - previous
        if state.box is not None:
            # A bead that crossed the periodic wall has not moved a box length;
            # without this the estimator would read one as the typical step and
            # the whole scene would shake.
            delta = state.box.minimum_image(delta)
        self._blend_step(np.median(np.linalg.norm(delta, axis=1)))
        if dirs is not None and previous_dirs is not None \
                and len(previous_dirs) == len(dirs):
            self._blend_dir_step(
                np.median(np.linalg.norm(dirs - previous_dirs, axis=1)))

    def _blend_step(self, measured):
        measured = float(measured)
        if not np.isfinite(measured):
            return
        self._step += AMPLITUDE_ALPHA * (measured - self._step)

    def _blend_dir_step(self, measured):
        measured = float(measured)
        if not np.isfinite(measured):
            return
        self._dir_step += AMPLITUDE_ALPHA * (measured - self._dir_step)

    @staticmethod
    def _sigma_for(median_step, share, rho, dof=3):
        """The OU's stationary sigma that makes its per-frame step come out right.

        This is the piece that has to be got right, and the naive version -- set
        sigma to some fraction of the measured step -- is wrong three times over.
        Each correction is near 1 at the 20-fps-sent, 60-Hz-drawn default, which
        is exactly why they are worth writing down rather than folding into a
        constant: that constant would be right for one wire rate on one monitor
        and silently wrong for every other pair.

        1. THE MEASUREMENT IS A LENGTH, THE PROCESS WANTS A COMPONENT. `observe`
           reports the median of |displacement|, a 3-vector magnitude, while the
           OU is driven per component. For an isotropic Gaussian the median of
           |v| is sigma*sqrt(median of chi-squared_dof) -- 1.538 for the three
           degrees of freedom a position has, 1.177 for the two a unit director
           has left once the renormalisation has thrown the radial component
           away. Skipping this alone makes the scene run 54% hot.

        2. WHAT WE ARE MATCHING is how far a bead moves between two DRAWN frames,
           because that is the quantity the eye integrates into "this scene is at
           temperature T". A full-rate wire would have shown

               drawn_step = wire_step * sqrt(share)

           where `share` is the fraction of a wire frame one drawn frame covers
           (a third, at 20 sent and 60 drawn). The square root is not a fudge:
           these are overdamped Langevin beads, so displacement grows as sqrt(t)
           over the times in question, and using `share` un-rooted would
           under-state the motion by the same factor it is wrong by.

        3. WHAT THE PROCESS DELIVERS per frame is not sigma either. An OU step is
           x <- rho*x + sigma*sqrt(1 - rho^2)*N, so consecutive samples differ by
           sigma*sqrt(2*(1 - rho)), and rho depends on the frame time. Inverting
           that is what keeps the apparent temperature the same whether the window
           is running at 60 Hz or 30, instead of making a slow machine look cold.
        """
        component = median_step / (1.5382 if dof == 3 else 1.1774)
        per_frame = float(np.sqrt(2.0 * max(1.0 - rho, 1e-6)))
        return component * float(np.sqrt(max(share, 1e-6))) / per_frame

    def apply(self, state, strength, wall_dt, share=1.0):
        """`state` with a synthetic rattle added, for ONE drawn frame.

        `strength` is the slider (0 disables), `wall_dt` the wall-clock seconds
        since the last drawn frame, `share` the fraction of a wire frame this
        drawn frame covers. Returns the state unchanged -- and drops the wobble --
        whenever there is nothing to add, so that switching the effect off,
        pausing, or freezing the simulation all leave the picture exactly on the
        coordinates that arrived.
        """
        pos = None if state is None else np.asarray(state.positions, dtype=float)
        if (state is None or strength <= 0.0 or wall_dt <= 0.0
                or pos is None or not len(pos) or self._step <= 0.0):
            self._offset = self._dir_offset = None
            return state

        rho = float(np.exp(-wall_dt / TAU_WALL))
        sigma = float(strength) * self._sigma_for(self._step, share, rho)
        offset = self._advance(self._offset, len(pos), sigma, rho)
        self._offset = offset
        out_pos = pos + offset
        if state.box is not None:
            out_pos = state.box.wrap(out_pos)

        out_dirs = state.directors
        if out_dirs is not None and self._dir_step > 0.0:
            dirs = np.asarray(out_dirs, dtype=float)
            dir_sigma = float(strength) * self._sigma_for(self._dir_step, share,
                                                          rho, dof=2)
            dir_offset = self._advance(self._dir_offset, len(dirs), dir_sigma, rho)
            self._dir_offset = dir_offset
            # Perturb then renormalize: a small additive kick to a unit vector is
            # a rotation to first order, and unlike an explicit rotation it costs
            # no trigonometry per bead.
            out_dirs = normalize_rows(dirs + dir_offset)

        return replace(state, positions=out_pos, directors=out_dirs)

    def _advance(self, offset, n, sigma, rho):
        """One exact discrete Ornstein-Uhlenbeck step.

        x <- rho*x + sigma*sqrt(1 - rho^2)*N(0, 1). This is not an approximation
        of the continuous process sampled at dt -- it is its exact transition,
        which matters because the frame time it comes from is a real one and
        therefore jitters. An Euler update would make the wobble's amplitude
        depend on how steady the frame rate happened to be, i.e. would make a
        hitch visible as a change in temperature.
        """
        if offset is None or len(offset) != n:
            # Seed AT the stationary distribution rather than at zero, so the
            # effect comes up at full strength instead of visibly swelling over
            # its first tenth of a second.
            offset = self._rng.standard_normal((n, 3), dtype="f4")
            offset *= np.float32(sigma)
            return offset
        kick = self._rng.standard_normal((n, 3), dtype="f4")
        kick *= np.float32(sigma * np.sqrt(max(1.0 - rho * rho, 0.0)))
        offset = offset * np.float32(rho) + kick
        limit = np.float32(CLAMP_SIGMAS * sigma)
        return np.clip(offset, -limit, limit, out=offset)
