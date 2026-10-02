"""Interaction modes: how (or whether) the user drives the simulation.

This is the split that lets one playground be both a game and an experiment.
Previously "has a puller" was baked into each system class -- MDSystem declared 7
puller-shaped abstract methods, and the self-assembly system had to stub five of
them and invent a `SliderSpec("Puller damping (unused)", ...)` to satisfy a
required field. Now:

    GameMode  one controlled particle driven by joystick/mouse/keyboard force,
              held on a control plane inside a leash, with haptics and the
              force-field reaction force recovered for force feedback.
    SimMode   no control particle; Play / Pause / Reset, and Reset rebuilds from
              a fresh random state.

Because the mode is a separate object, `--mode game` and `--mode sim` both work
on any playground -- so you can watch a structure self-assemble and then grab a
particle and probe it, which was not possible at any price before.
"""
import math

import numpy as np


def select_controlled(positions, box, selector):
    """Index of the particle the user controls.

    Selectors:
      "first"           -- the first particle (a scenario that puts its centre
                           site first, like the hex patch, gets that centre)
      "nearest_center"  -- nearest the box centre, measured in-plane
      "nearest:x,y,z"   -- nearest a given point
    """
    if not len(positions):
        return None
    if selector in (None, "none"):
        return None
    if selector == "first":
        return 0
    if selector == "last":
        # A deposition scenario creates its slab first and the free atom last, so
        # the highest id is the one meant to be driven.
        return len(positions) - 1
    if selector == "nearest_center":
        target = np.array(box.center if box is not None else (0.0, 0.0, 0.0))
        return int(np.argmin(np.linalg.norm(positions[:, :2] - target[:2], axis=1)))
    if isinstance(selector, str) and selector.startswith("nearest:"):
        target = np.array([float(t) for t in selector[len("nearest:"):].split(",")])
        return int(np.argmin(np.linalg.norm(positions - target, axis=1)))
    if isinstance(selector, (int, np.integer)):
        return int(selector)
    raise ValueError(f"unrecognized control-atom selector {selector!r}")


_AXES = {"x": 0, "y": 1, "z": 2}


def signed_axis(name):
    """(axis index, sign) for a signed axis name: "y" -> (1, +1), "-x" -> (0, -1).

    Torque axes are given signed (see Control.torque_axes) because which way a
    director tips for a given stick push is a statement about the camera, and the
    sign is the whole content of it -- so it belongs in the playground file next to
    the axis, not folded into the mode as a minus somebody has to go and find.
    """
    text = str(name).strip()
    sign = 1.0
    if text[:1] in "+-":
        sign = -1.0 if text[0] == "-" else 1.0
        text = text[1:]
    if text not in _AXES:
        raise ValueError(f"axis must be one of x/y/z, optionally signed, "
                         f"got {name!r}")
    return _AXES[text], sign


def plane_axes(plane):
    """(u_axis, v_axis, pinned_axis) indices for a two-letter plane name.

    The control plane is the plane the two input axes slide the particle in;
    the third axis is pinned so a 2-axis stick fully determines its 3D position.
    "xz" -- the shipped choice -- puts the plane perpendicular to a membrane
    lying in xy and facing the camera, so input x is screen-horizontal and input
    y is out-of-plane.
    """
    if len(plane) != 2 or any(c not in _AXES for c in plane):
        raise ValueError(f"plane must be two of x/y/z, got {plane!r}")
    u, v = _AXES[plane[0]], _AXES[plane[1]]
    pin = ({0, 1, 2} - {u, v}).pop()
    return u, v, pin


class Mode:
    """Base mode. `attach` gives it the runtime it drives."""

    playback_controls = False
    needs_control_particle = False

    def attach(self, runtime):
        self.runtime = runtime

    # --- deck contributions ---------------------------------------------------

    def group_commands(self):
        return []

    def thermostat_group(self):
        """The group the Langevin bath acts on."""
        return "all"

    def control_commands(self, params):
        return []

    # --- per-frame ------------------------------------------------------------

    def before_step(self):
        """Called right before each chunk's `run`, with the inputs it will use."""
        pass

    def after_step(self, dt):
        pass

    # --- the app-facing readouts ---------------------------------------------

    def set_input_force(self, fx, fy):
        pass

    def set_damping(self, gamma):
        pass

    def steer_orientation(self, rate, dt):
        pass

    # Whether the input device is holding a particle. A mode with no controlled
    # particle is permanently "not holding one", and toggling is a no-op, so the
    # app can offer the key on every system without asking what mode it is in.
    attached = False

    def toggle_attached(self):
        return self.attached

    def controlled_index(self):
        return None

    def puller_state(self):
        return None, None

    def interaction_force(self):
        return np.array([0.0, 0.0])

    def torque_signals(self):
        return None

    def torque_vectors(self):
        return None

    def control_grid(self):
        return None


class SimMode(Mode):
    """Watch it run: Play / Pause / Reset, no interactive particle.

    Every puller-shaped readout returns a neutral value, which the app already
    handles -- but now that is one small class rather than five stub methods
    copied into a system that has no puller.
    """

    playback_controls = True

    def thermostat_group(self):
        return "all"


class GameMode(Mode):
    """Drive one particle with an input device, and feel the force field push
    back.

    TWO DRIVES, one stick, chosen by `Control.drive`:

      FORCE   the two axes push the particle, in the control plane. It is confined
              to that plane and a rectangular leash, and capped below a runaway
              speed: without this a sustained max pull would accelerate it
              (undamped by the thermostat, since it is deliberately excluded from
              the bath) straight out of the box. The leash is drawn in the scene as
              a net, whose extents ARE these limits, so the net marks exactly where
              the particle can be dragged. Its DIRECTOR is steered separately, by
              the twist axis, and only within the control plane.
      TORQUE  the two axes turn the particle's director instead, about two world
              axes, and nothing pushes the particle at all: it goes where the force
              field takes it. Nothing is confined -- no plane, no leash, no net, and
              the director tumbles in three dimensions rather than in a plane --
              because there is no longer an input whose two axes have to fully
              determine a position, which is the only reason the constraint existed.

    Both render the force field's REACTION back to the hand; which quantity that is
    follows the drive (see `interaction_force`).
    """

    needs_control_particle = True

    def __init__(self, control):
        self.control = control
        self.u_axis, self.v_axis, self.pin_axis = plane_axes(control.plane)
        # Which world axis each input axis torques about, and which way, for a
        # torque drive. Resolved once, here, so a bad axis name in a playground
        # file is a build-time error rather than a silent no-op per frame.
        self.torque_axes = (tuple(signed_axis(a) for a in control.torque_axes)
                            if control.drives_torque else ())
        self._input_u = 0.0
        self._input_v = 0.0
        self._yaw = 0.0
        self._damping = control.damping_default
        # The drive and the drag the LAST RUN actually integrated with, which is
        # what interaction_force has to take back out of LAMMPS' `f`. Not
        # _input_u/_damping: the app sets next frame's input BEFORE it reads this
        # frame's reaction (see App._tick), so those already hold the values for
        # the run that has not happened yet, and subtracting them was off by the
        # frame-to-frame change in the stick's force. Snapshotted in before_step.
        self._applied_u = 0.0
        self._applied_v = 0.0
        self._applied_damping = control.damping_default
        self._pin_value = 0.0
        self._has_group_force = False
        # Whether the input device is currently holding this particle. Released,
        # it is an ordinary particle of the simulation: no drive, no steering, no
        # leash -- it falls back into the membrane, or drifts, under the force
        # field alone. It keeps its control plane (so it stays where the net says
        # it is, ready to be grabbed again without a jump) and its speed cap.
        self.attached = True

    # --- deck ----------------------------------------------------------------

    def group_commands(self):
        cid = self.runtime.controlled_id
        if cid is None:
            # No particle could be selected (an empty scenario, or atom="none").
            # Degrade to thermostatting everything rather than emitting a broken
            # group command; every readout below already handles a missing
            # controlled particle.
            return []
        return [
            f"group controlled id {cid}",
            # The bath thermostats everything EXCEPT the controlled particle.
            # Never `all`: folding the controlled particle into the Langevin bath
            # gives its director thermal rotational noise that never goes away,
            # and silently breaks the reaction-force recovery below (this was a
            # real bug once -- see interaction_force).
            "group bath subtract all controlled",
        ]

    def thermostat_group(self):
        return "bath" if self.runtime.controlled_id is not None else "all"

    def control_commands(self, params):
        if self.runtime.controlled_id is None:
            return []
        # The viscous drag is here for both drives. A torque drive applies no
        # force, but the particle is still outside the Langevin bath (see
        # group_commands) and would otherwise be the one particle in the scene with
        # no translational friction at all -- so it keeps the damping slider, and
        # what that slider means does not change with the drive.
        cmds = [f"fix damp controlled viscous {self._damping}"]
        if not self.control.drives_torque:
            # The drive is variable-driven rather than a literal force, so that
            # steering it (set_input_force, potentially every frame) sets three
            # internal variables instead of REDEFINING the fix. Redefining a fix
            # invalidates `run ... pre no` -- so the old literal form silently cost
            # every interactive system a full neighbour rebuild and force evaluation
            # per chunk, for a number that fix addforce is perfectly happy to read
            # from a variable each step. See PlaygroundSystem.command.
            cmds = ["variable drive_x internal 0.0",
                    "variable drive_y internal 0.0",
                    "variable drive_z internal 0.0",
                    "fix drive controlled addforce v_drive_x v_drive_y v_drive_z",
                    *cmds]
        if self.control.displacement_cap:
            # An unconfined particle needs a per-step displacement cap to survive
            # a hard contact impact instead of tunnelling through the lattice.
            cmds.append(f"fix integ_controlled controlled nve/limit "
                        f"{self.control.displacement_cap}")
        if self.control.confine:
            # Zero the out-of-plane force every step. Defined last so it also
            # cancels any out-of-plane interaction force.
            pin = ["NULL", "NULL", "NULL"]
            pin[self.pin_axis] = "0.0"
            cmds.append(f"fix plane controlled setforce {' '.join(pin)}")
        if self.runtime.force_field.supports_single:
            # A pair style with single() can report the force between two groups
            # directly, so the reconstruction below is unnecessary -- and exact
            # rather than dependent on knowing every fix acting on the particle.
            cmds.append("compute pairforce controlled group/group crystal")
            self._has_group_force = True
        return cmds

    def on_built(self):
        """Record the controlled particle's home coordinate on the pinned axis --
        the plane it is held in, and the origin of the drawn net."""
        pos = self.runtime.controlled_position()
        if pos is not None:
            self._pin_value = float(pos[self.pin_axis])
        self.constrain()

    # --- inputs --------------------------------------------------------------

    def toggle_attached(self):
        """Grab or release the controlled particle. Returns the new state."""
        self.attached = not self.attached
        if not self.attached:
            # Drop the drive and the steering immediately, rather than leaving
            # the last frame's values latched on a particle nobody is holding.
            self._yaw = 0.0
            self.set_input_force(0.0, 0.0)
        return self.attached

    def set_input_force(self, fx, fy):
        """This frame's two input-axis values.

        On a FORCE drive: axis 1 -> the plane's u axis, axis 2 -> its v axis, both
        in force units, handed to the addforce fix.

        On a TORQUE drive: the two values are TORQUES about `torque_axes`, and there
        is no force fix to hand anything to -- they are stored and applied as
        angular-momentum kicks in the next `constrain()`, the same mechanism the
        twist axis has always used. Named `set_input_force` all the same, because it
        is the app's one "here is this frame's input" call and splitting it would
        mean every caller asking which drive it was talking to (see MDSystem).
        """
        if not self.attached:
            fx = fy = 0.0
        self._input_u = fx
        self._input_v = fy
        if self.runtime.controlled_id is None or self.control.drives_torque:
            return
        f = [0.0, 0.0, 0.0]
        f[self.u_axis] = fx
        f[self.v_axis] = fy
        # Not a command: setting the internal variables the drive fix reads leaves
        # the fix itself untouched, so the next chunk can still skip its setup.
        for name, value in zip(("drive_x", "drive_y", "drive_z"), f):
            self.runtime.lmp.set_internal_variable(name, value)

    def set_damping(self, gamma):
        lo, hi = self.control.damping_range
        gamma = max(lo, min(hi, gamma))
        if gamma == self._damping:
            return
        self._damping = gamma
        if self.runtime.controlled_id is not None:
            # Redefines a fix, so it goes through command(): the chunk after a
            # damping change does a full setup, the ones after it do not.
            self.runtime.command(f"fix damp controlled viscous {gamma}")

    def steer_orientation(self, rate, dt):
        # Sign flipped so the twist turns the director the way the hand expects
        # on screen. Applied in the next constrain().
        #
        # Ignored on a torque drive: the two main axes already turn the director
        # about both of the axes it can be turned about, and a third rotation --
        # about the director itself -- is the identity on a unit vector. Leaving the
        # twist wired up would mean one of the two mappings silently fought the
        # other on whichever axis they shared.
        if self.control.drives_torque:
            self._yaw = 0.0
            return
        self._yaw = -rate if self.attached else 0.0

    # --- per-frame constraint ------------------------------------------------

    def before_step(self):
        self._applied_u = self._input_u
        self._applied_v = self._input_v
        self._applied_damping = self._damping

    def after_step(self, dt):
        self.constrain()

    def constrain(self):
        """Hold the controlled particle where the input can reach it, and drive its
        director.

        Two independent halves, because the two drives want different ones:

          the PLANE, the leash and the speed cap -- only for a confined particle,
          which is what a force drive needs so that two input axes fully determine
          a 3D position;
          the DIRECTOR, driven either in the control plane by the twist axis (force
          drive) or freely about two world axes by the stick (torque drive).

        The spring-back the user feels is genuine force-field physics (the tilt
        term, integrated by LAMMPS) in both cases. What is added here is only the
        command itself, which enters as an angular-momentum kick (a torque is dL/dt),
        and a rotational drag that stands in for the controlled particle's share of
        the implicit solvent's rotational friction -- it sits outside the Langevin
        bath, so without this an undamped director would oscillate forever.
        """
        ic = self.runtime.controlled_local()
        if ic is None:
            return
        if self.control.confine:
            self._hold_in_leash(ic)
        if not self.runtime.has_directors:
            return
        if self.control.drives_torque:
            self._drive_director_freely(ic)
        elif self.control.confine:
            # Unchanged: on an unconfined FORCE drive nothing here ever ran, and
            # the deposition playgrounds that use it have no directors anyway.
            self._drive_director_in_plane(ic)

    def _hold_in_leash(self, ic):
        """The plane constraint, the leash and the speed cap."""
        lmp = self.runtime.lmp
        x = lmp.numpy.extract_atom("x")
        v = lmp.numpy.extract_atom("v")

        # Exact plane constraint (belt-and-braces with the setforce fix).
        x[ic][self.pin_axis] = self._pin_value
        v[ic][self.pin_axis] = 0.0
        # The leash only exists while the particle is being steered: a released
        # one is part of the simulation and free to go where the physics takes it
        # (the walls still catch it -- see Scenario.wall_commands).
        if self.attached:
            for axis, (lo, hi) in ((self.u_axis, self.control.u_range),
                                   (self.v_axis, self.control.v_range)):
                if x[ic][axis] < lo:
                    x[ic][axis] = lo
                    if v[ic][axis] < 0.0:
                        v[ic][axis] = 0.0
                elif x[ic][axis] > hi:
                    x[ic][axis] = hi
                    if v[ic][axis] > 0.0:
                        v[ic][axis] = 0.0
        speed = math.sqrt(v[ic][0] ** 2 + v[ic][1] ** 2 + v[ic][2] ** 2)
        cap = self.control.speed_cap
        if speed > cap:
            s = cap / speed
            v[ic][0] *= s
            v[ic][1] *= s
            v[ic][2] *= s

    def _drive_director_in_plane(self, ic):
        """The twist axis, on a force drive: spin about the plane normal only, so
        the director's swing stays in the control plane and the two axes the stick
        moves the particle along are not also rotating it."""
        lmp = self.runtime.lmp
        mu = lmp.numpy.extract_atom("mu")
        omega = lmp.numpy.extract_atom("omega")
        omega[ic][self.u_axis] = 0.0
        omega[ic][self.v_axis] = 0.0
        omega[ic][self.pin_axis] = (omega[ic][self.pin_axis] * self.control.rot_damp
                                    + self._yaw * self.control.yaw_torque)
        # Kill any out-of-plane director drift.
        nu, nv = mu[ic][self.u_axis], mu[ic][self.v_axis]
        m = math.hypot(nu, nv)
        if m > 1e-9:
            mu[ic][self.u_axis] = nu / m
            mu[ic][self.pin_axis] = 0.0
            mu[ic][self.v_axis] = nv / m

    def _drive_director_freely(self, ic):
        """The two input axes, on a torque drive: an angular-momentum kick about
        each of `torque_axes`, and the same rotational drag, with nothing projected
        away.

        No axis is zeroed and the director is not renormalized, both deliberately.
        The director tumbles in three dimensions here -- that is the point of this
        drive -- so there is no out-of-plane component to call drift, and LAMMPS'
        own `nve/sphere update dipole` preserves |mu| as it integrates, which the
        in-plane version above has to undo its own projection to keep.

        The input arrives already scaled by `max_input_torque` and enters as dL/dt,
        exactly as the twist axis does -- so `max_input_torque` here and
        `yaw_torque` there are the same kind of number, directly comparable, and
        deliberately have the same default. It is NOT multiplied by `yaw_torque`
        again: two knobs setting one gain is how a demo ends up 3x too twitchy for
        reasons nobody can find.
        """
        omega = self.runtime.lmp.numpy.extract_atom("omega")
        damp = self.control.rot_damp
        for (axis, sign), value in zip(self.torque_axes,
                                       (self._input_u, self._input_v)):
            omega[ic][axis] = omega[ic][axis] * damp + sign * value
        # The third axis is left to the physics: the force field's own torque about
        # it is real (a director being tipped by its neighbours), and zeroing it
        # would be inventing a constraint this drive exists not to have. It only
        # needs the same drag as the two driven ones, or it would ring forever
        # outside the bath.
        driven = {axis for axis, _ in self.torque_axes}
        for axis in (0, 1, 2):
            if axis not in driven:
                omega[ic][axis] *= damp

    # --- readouts ------------------------------------------------------------

    def controlled_index(self):
        return self.runtime.controlled_index

    def puller_state(self):
        """(position, velocity) projected onto the control plane, so the app's
        2D haptics pipeline works unchanged on a 3D scene."""
        ic = self.runtime.controlled_local()
        if ic is None:
            return None, None
        n = self.runtime.natoms
        x = self.runtime.lmp.numpy.extract_atom("x")[:n]
        v = self.runtime.lmp.numpy.extract_atom("v")[:n]
        return (np.array([x[ic][self.u_axis], x[ic][self.v_axis]]),
                np.array([v[ic][self.u_axis], v[ic][self.v_axis]]))

    def interaction_force(self):
        """The force field's reaction on the controlled particle, along the two axes
        the input drives -- what the arrows draw and the stick renders.

        ON A TORQUE DRIVE it is the reaction TORQUE about `torque_axes`, read
        straight off the pair style's per-atom torque. That is exact rather than
        reconstructed, because nothing this end adds to that array: the user's
        command is an angular-momentum kick applied directly to omega (see
        `_drive_director_freely`), not a torque competing with the force field's for
        a place in the same sum. It is the same quantity in the same units as the
        force case, one domain over, so the whole pipeline downstream -- shaping,
        smoothing, stiffness, the HUD -- works on it unchanged.

        ON A FORCE DRIVE, the reaction force, projected onto the control plane.

        Recovered as total force minus the two forces we apply ourselves: the
        input drive (addforce) and the viscous damping (-gamma*v). The setforce on
        the pinned axis is irrelevant because only the two in-plane components are
        read.

        This substitutes for `compute group/group`, which a pair style without a
        single() method cannot support -- and MesoMem's has none. It is therefore
        only correct while (a) exactly these two fixes act on the particle in the
        plane, (b) _applied_u/_applied_v are the addforce the last run integrated
        with (snapshotted in before_step, not the input already set for the next
        run), and (c) the
        particle is excluded from the thermostat. All three are enforced above,
        and a regression test pins the arithmetic.

        Two things then shape it, and both are about what the user's HAND should
        feel rather than about the physics, which is untouched either way:
        releasing the particle reports nothing (you are not holding it), and the
        leash boundary is faded out (see Control.leash_release).
        """
        if not self.attached:
            return np.array([0.0, 0.0])
        if self.control.drives_torque:
            return self._reaction_torque()
        if self._has_group_force:
            vec = self.runtime.lmp.extract_compute("pairforce", 0, 1)
            f_uv = np.array([vec[self.u_axis], vec[self.v_axis]])
        else:
            ic = self.runtime.controlled_local()
            if ic is None:
                return np.array([0.0, 0.0])
            n = self.runtime.natoms
            f = self.runtime.lmp.numpy.extract_atom("f")[:n]
            v = self.runtime.lmp.numpy.extract_atom("v")[:n]
            g = self._applied_damping
            f_uv = np.array([
                f[ic][self.u_axis] - self._applied_u + g * v[ic][self.u_axis],
                f[ic][self.v_axis] - self._applied_v + g * v[ic][self.v_axis],
            ])
        return f_uv * self._leash_release()

    def _reaction_torque(self):
        """The force field's restoring torque about the two driven axes, signed the
        same way the input to those axes is -- so a command and the reaction to it
        read as opposite, which is what the green/red pair means everywhere else.

        No leash fade: there is no leash on a torque drive, so there is no boundary
        for the signal to have to melt away at (see _leash_release).
        """
        ic = self.runtime.controlled_local()
        tau = (self.runtime.lmp.numpy.extract_atom("torque")
               if ic is not None and self.runtime.has_directors else None)
        if tau is None:
            return np.array([0.0, 0.0])
        row = tau[:self.runtime.natoms][ic]
        return np.array([sign * float(row[axis])
                         for axis, sign in self.torque_axes])

    def _leash_release(self):
        """Per-axis 1 -> 0 fade of the reported force as the particle nears its
        leash, so arriving at the limit is silent instead of a sustained shove.

        Position-based, so it is smooth in time however fast the particle is
        moving, and symmetric in the force's direction -- gating only the outward
        component would make the signal jump the moment the membrane's pull
        changed sign. Returns (gain_u, gain_v); all ones when there is no leash to
        approach.
        """
        release = self.control.leash_release
        if not self.control.confine or release <= 0.0:
            return np.array([1.0, 1.0])
        ic = self.runtime.controlled_local()
        if ic is None:
            return np.array([1.0, 1.0])
        x = self.runtime.lmp.numpy.extract_atom("x")
        gains = []
        for axis, (lo, hi) in ((self.u_axis, self.control.u_range),
                               (self.v_axis, self.control.v_range)):
            width = max(release * 0.5 * (hi - lo), 1e-9)
            # Distance to the nearer of the two limits, in units of that width.
            t = min(max(min(hi - x[ic][axis], x[ic][axis] - lo) / width, 0.0), 1.0)
            gains.append(t * t * (3.0 - 2.0 * t))       # smoothstep
        return np.array(gains)

    def torque_signals(self):
        """(applied, reaction) torque about ONE axis, normalized to [-1, 1] for the
        circular torque arrows.

        WHICH AXIS, and why one and not two. These feed the FLAT arcs, drawn as
        circles on the screen around the bead, so they can only depict a rotation
        the camera sees face on -- about the axis pointing at it. On a force drive
        that is the control plane's normal, the axis the twist rotates the director
        about, and the axis those scenes are looked down: honest, and the only one
        that is. On a torque drive it is `torque_axes[0]`, and there the flat arc is
        no longer drawn at all -- the rings (`torque_vectors`) show both driven axes
        where they actually point. The scalar is kept because it is still the
        first-axis reading, and cheap.

        `applied` is the user's command as a fraction of full deflection; `reaction`
        is the force field's restoring torque over `reaction_torque_max`.
        """
        # Both arcs describe a hand on the particle: your twist, and what the
        # membrane twists back with. Released, there is no hand, so there are no
        # arcs -- rather than a live restoring torque drawn around a particle
        # nobody is steering.
        ic = self.runtime.controlled_local()
        if ic is None or not self.runtime.has_directors or not self.attached:
            return None
        if self.control.drives_torque:
            axis, sign = self.torque_axes[0]
            ceiling = max(self.control.max_input_torque, 1e-9)
            applied = max(-1.0, min(1.0, sign * self._input_u / ceiling))
        else:
            axis, sign = self.pin_axis, 1.0
            applied = max(-1.0, min(1.0, self._yaw))
        tau = self.runtime.lmp.numpy.extract_atom("torque")
        reaction = 0.0
        if tau is not None:
            raw = sign * float(tau[:self.runtime.natoms][ic][axis])
            reaction = max(-1.0, min(1.0, raw / self.control.reaction_torque_max))
        return applied, reaction

    def torque_vectors(self):
        """(applied, reaction) as world 3-VECTORS, for the two torque RINGS drawn
        round the bead. None on a force drive, which has real force vectors to draw
        instead.

        A torque is an axial vector -- it points along the axis the rotation is
        about, by the right-hand rule -- so these are genuine directions in the
        scene and not a picture of one. The applied one is the two input axes
        combined, `sum(sign_i * value_i * e_i)`; the reaction is the pair style's
        own per-atom torque, all three components of it, because the membrane
        resists about whatever axis it likes and projecting that onto the two driven
        ones would be drawing the part of its answer we asked for.

        NORMALIZED, each to its own full scale: 1.0 is a full stick deflection for
        the applied one and `reaction_torque_max` for the reaction. They are in
        different units from each other in no useful sense -- both are torques --
        but they differ in SIZE by about the ratio of those two ceilings, and drawn
        raw against one ring scale the input would be a hairline next to the
        reaction. The header line carries the unnormalized numbers.

        WHAT THE RENDERER MAKES OF IT is a circle in the plane perpendicular to the
        vector, drawn in the scene and projected like everything else, with the
        length setting how far round it sweeps (see Renderer._draw_torque_ring). It
        is NOT drawn as an arrow along the axis: an axial vector is a bookkeeping
        convention, nothing travels along it, and an arrow there is a picture of a
        push this drive pointedly does not apply.
        """
        if not self.control.drives_torque:
            return None
        ic = self.runtime.controlled_local()
        if ic is None or not self.runtime.has_directors or not self.attached:
            return None
        applied = np.zeros(3)
        ceiling = max(self.control.max_input_torque, 1e-9)
        for (axis, sign), value in zip(self.torque_axes,
                                       (self._input_u, self._input_v)):
            applied[axis] += sign * value / ceiling
        reaction = np.zeros(3)
        tau = self.runtime.lmp.numpy.extract_atom("torque")
        if tau is not None:
            reaction = (np.asarray(tau[:self.runtime.natoms][ic], dtype=float)
                        / max(self.control.reaction_torque_max, 1e-9))
        return applied, reaction

    def control_grid(self):
        """The net: the control plane's basis and extents. Drawn exactly at the
        leash limits, so it marks precisely where the particle can be dragged.
        None for an unconfined particle -- there is no boundary to draw."""
        if not self.control.confine:
            return None
        origin = [0.0, 0.0, 0.0]
        origin[self.pin_axis] = self._pin_value
        u = [0.0, 0.0, 0.0]
        u[self.u_axis] = 1.0
        v = [0.0, 0.0, 0.0]
        v[self.v_axis] = 1.0
        return dict(origin=tuple(origin), u_axis=tuple(u), v_axis=tuple(v),
                    u_range=self.control.u_range, v_range=self.control.v_range,
                    step=self.control.grid_step)
