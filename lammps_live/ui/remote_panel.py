"""The connect panel: get a GPU on the cluster, from inside the running app.

A card over the sim view, shown whenever the remote playground is selected and not
streaming. It runs `remote.session.RemoteSession` -- allocate, deploy, launch,
tunnel -- and shows what it is doing, one line per step, with the far side's own
output underneath. The only thing it asks for is the answer to whatever the login
prompts for, because that is the one thing no amount of automation can supply.

WHY IT IS IN THE APP AND NOT A SCRIPT. The demo is a thing you stand in front of.
Allocating the GPU from a terminal first, then starting the app, then reconnecting
by hand when the tunnel drops, is three ways for a demo to fail in public. And the
teardown matters more than the setup: an A100 held by a forgotten allocation is
expensive, so the thing that ends the session has to be the thing you close --
which means the app has to own it.

ONE GPU, SEVERAL DEMOS. There is more than one remote playground now, and only ever
one session: the panel keeps it across playground changes rather than one per
playground. What follows from that is the behaviour a conference talk needs --

    cycling to the other remote playground with Tab or a number key costs nothing
    and takes nothing away. The card comes up saying the GPU is held and what is
    running on it; the run you left is still running, and going back to it is one
    socket.
    CONNECTING on the other one moves the allocation: the far side sets the
    simulation it was holding ASIDE and builds this one in its place, on the same
    node, through the same tunnel, with the same job (see
    session.switch_playground). Nothing queues, and nothing asks for a one-time code
    a second time.

-- so the GPU is requested once, at the start, and the rest of the hour is spent
switching between demos. A switch used to cost the state of the run being left
behind, which is why it takes a button press rather than happening on Tab; it does
not any more (the far side parks it and puts it back -- see
server.FrameServer.switch_playground), and what it costs now is about a second of
rebuild.

DRIVEABLE FROM THE STICK, not just the mouse. This card is modal, and it is up
precisely when there is no simulation for the joystick to steer -- so while it
shows, left/right (the hat, or the stick's own x axis) walks its buttons and the
trigger presses the one with the cyan ring on it. Which matters because standing in
front of a room with a joystick in one hand and reaching back for a trackpad to
press Connect is the kind of thing that goes wrong in public. See `push_axis` /
`step_focus` / `activate_focus` below, and App._poll_device_buttons for the mapping.

NOTHING HERE BLOCKS. Every step runs on the session's worker thread, the switch
included; this reads its state once per frame. The app keeps drawing at 60 fps
through an SSH login, a queue wait and a LAMMPS build on the far side.
"""
import time

import pygame

from . import clipboard
from .scale import UI
from .theme import (
    BUTTON_BORDER, DIM_TEXT_COLOR, HEADER_TEXT_COLOR, PANEL_BG, PANEL_DIVIDER,
    SLIDER_HANDLE_ACTIVE, TEXT_COLOR,
)
from .widgets import Button, TextField

# The palette this panel adds to the theme: one colour per resting state, so the
# card reads at a glance from across a room.
OK_COLOR = (120, 230, 170)
BUSY_COLOR = (255, 210, 90)
FAIL_COLOR = (255, 110, 90)

CARD_WIDTH = 620
LOG_LINES = 14


class RemotePanel:
    """Owns THE session -- one, for the whole app -- plus the buttons and the prompt
    field, pointed at whichever remote playground is on screen."""

    def __init__(self):
        self.visible = False
        self.system = None
        # One session, shared by every remote playground that describes the same
        # cluster. What it is currently serving is `session.playground_asked`, which is
        # not necessarily what is on screen -- that is `playground_key` below, and
        # the gap between the two is what the card is for.
        self.session = None
        # The remote playground on screen (or the last one, while a local playground
        # is showing). Connect acts on this one; the session may be holding another.
        self.playground_key = None
        self.field = TextField(masked=True, placeholder="type the answer, then Enter")
        self.buttons = {
            "connect": Button("connect", "Connect"),
            "cancel": Button("cancel", "Cancel"),
            "disconnect": Button("disconnect", "Disconnect"),
            "copy": Button("copy", "Copy report (C)"),
            "close": Button("close", "Close (N)"),
        }
        self._shown = ()               # which buttons are on screen right now
        # WHICH BUTTON THE JOYSTICK IS ON. The card is modal and it is the one place
        # in the app a demo can get stuck without a mouse: the GPU is not connected
        # yet, so there is nothing for the stick to steer and every button that
        # matters is here. Left/right (the hat, or the stick's own x axis) walks
        # these; the trigger presses one. See App._poll_device_buttons.
        #
        # An index into `_shown`, which is rebuilt every frame as the session
        # changes state -- so `_aim_focus` re-points it whenever that set changes
        # rather than leaving a stale number to be clamped into something arbitrary.
        self.focus_button = 0
        self._axis_armed = True
        self._last_state = None
        # What the copy button just did, and until when to say so. A copy that
        # produces no visible change is a copy the user does again, and again.
        self._notice = None
        self._notice_until = 0.0

    # ---- lifecycle ----------------------------------------------------------

    def attach_system(self, system, playground_key):
        """Point the panel at a newly built RemoteSystem.

        The session is KEPT, whichever remote playground is arriving: it is the
        allocation, and the allocation is the expensive thing. Four outcomes, and
        not one of them asks Slurm for a GPU:

          * the session is streaming the playground that just arrived -- resume it,
            and the card never appears;
          * it is still working on that playground -- leave it working, and `update`
            hands the link over when it lands. Cycling away mid-connect must not
            cancel a queue wait that is nearly done;
          * it holds a GPU, on the OTHER remote playground (or on this one with a
            dead link) -- the card comes up and Connect moves it here. Nothing has
            been given back and nothing has to be asked for again;
          * there is no session, or the one there describes a different cluster
            entirely -- open a new one and show the card. Connect on THAT one is
            what queues.
        """
        from ..remote.session import READY
        session = self.session
        if session is not None and not _same_allocation(session.target,
                                                        system.target):
            # A remote playground pointed somewhere else -- another cluster, another
            # account, another scratch path. There is nothing to share, so the old
            # session is given back rather than quietly reused for a login it does
            # not describe.
            self.release()
            session = None
        self.system = system
        self.playground_key = playground_key
        self.field.clear()
        if session is None:
            self._new_session(system, playground_key)
            return
        if session.serves(playground_key):
            if session.state == READY:
                link = session.link
                # A link the app closed on the way out (system.close -> detach) has
                # to be reopened; one that arrived while this playground was not on
                # screen is still good, and MUST be reused rather than reconnected
                # -- the server serves one client at a time, so a second socket
                # would sit in the backlog behind our own.
                if link is None or link.closed.is_set():
                    link = session.reopen_link()
                if link is not None:
                    system.attach(link)
                    self.visible = False
                    return
            elif session.busy:
                self.visible = True
                return
        if not session.holds_allocation:
            # Nothing left to come back to -- it never started, or it found the job
            # gone and gave the rest back. Replace it, so the card's Connect asks
            # for a GPU rather than offering to move one that is not there.
            self.release()
            self._new_session(system, playground_key)
            return
        # A GPU is held: on the other remote playground, or on this one with a link
        # that has gone. Either way the card goes up and Connect does the right thing
        # (see RemoteSession.connect_playground).
        self.visible = True

    def _new_session(self, system, playground_key):
        """A fresh session for this playground, with the card up: nothing is
        connected yet, so the panel should say so."""
        from ..remote.session import RemoteSession
        self.system = system
        self.playground_key = playground_key
        self.session = RemoteSession(system.target, playground_ref=playground_key)
        self.visible = True

    def detach_system(self):
        """Switch away from the remote playground WITHOUT giving the GPU back.

        The allocation, the tunnel and the server survive, holding the simulation
        where it was, so the demo can go and show something else and come back to
        the same coarsened box (see attach_system). What ends the session is closing
        the window, pressing Disconnect, or the server's own idle timeout -- so a
        forgotten allocation still gives itself back, it just is not this method's
        job.
        """
        self.system = None
        self.visible = False
        self.field.clear()

    def release(self):
        """Drop everything: cancels the job, closes the tunnel and the login."""
        if self.session is not None:
            self.session.shutdown()
            self.session = None
        self.system = None
        self.visible = False
        self.field.clear()

    @property
    def active(self):
        """Is there a session attached to the playground currently on screen?

        Both halves matter: N must not raise this card over an unrelated
        playground just because a session is still held in the background.
        """
        return self.session is not None and self.system is not None

    def standby_note(self):
        """One line for the panel when the GPU is held but no remote playground is
        on screen -- an allocation nobody can see is the one thing about this that
        would be expensive to forget. None when there is nothing being held."""
        session = self.session
        if session is None or self.system is not None:
            return None
        from ..remote.session import READY
        if session.state == READY:
            return (f"remote GPU still held: job {session.job_id or '-'} on "
                    f"{session.node or session.target.label}, running "
                    f"{session.playground_asked} -- switch back to it to use it")
        if session.busy:
            return (f"remote session still connecting ({session.detail}) -- switch "
                    f"back to {session.playground_asked} to watch it")
        if session.holds_allocation:
            # The awkward one, and the reason this line exists at all: a job that is
            # still ours with nothing connected to it and nothing on screen saying
            # so. It gives itself back eventually (the server's idle timeout), but
            # not before it has been paid for.
            return (f"remote GPU held with no link: job {session.job_id} on "
                    f"{session.node or session.target.label} -- "
                    f"{session.error or 'the link is down'}")
        return None

    def toggle(self):
        if self.active:
            self.visible = not self.visible

    # ---- per frame ----------------------------------------------------------

    def update(self):
        """Called once per frame. Hands a finished session's link to the system."""
        if self.session is None:
            return
        from ..remote.session import FAILED, READY
        session = self.session
        if session.state != self._last_state:
            self._last_state = session.state
            if self.system is None or not session.serves(self.playground_key):
                # Nothing on screen to hand a link to, or what is on screen is not
                # the playground this session just finished building. The state is
                # remembered here and `attach_system` acts on it when that
                # playground comes back -- handing the link over now would attach a
                # stream of one simulation to a system that expects another.
                pass
            elif session.state == READY and session.link is not None:
                self.system.attach(session.link)
                # Out of the way: the scene is what matters now, and the link's own
                # readouts are on the HUD. N brings it back.
                self.visible = False
            elif session.state == FAILED:
                self.visible = True
        # A link that dies on its own (the job hit its time limit, the network
        # dropped, the node failed) reopens the panel -- with the reason in the log
        # rather than a frozen picture and no explanation.
        if (self.system is not None and not self.system.connected
                and session.state == READY and session.serves(self.playground_key)):
            session.note_link_lost(self.system.link_error())
            self.visible = True

    # ---- input --------------------------------------------------------------

    def handle_event(self, event):
        """True if the panel consumed the event.

        While a prompt is pending the panel takes every keystroke: the answer may
        be all digits, and those are the app's system-switch shortcuts.
        """
        if not self.visible or self.session is None:
            return False
        # C copies whatever the card is showing, in every state -- including while a
        # step is still running, which is when a session that is going to fail is
        # most worth capturing. Not while a prompt is pending: those keystrokes
        # belong to the answer, and a one-time code can contain a 'c'.
        if (event.type == pygame.KEYDOWN and not self.session.prompt
                and event.key == pygame.K_c):
            self._copy_report()
            return True
        if event.type == pygame.KEYDOWN and self.session.prompt:
            action = self.field.handle_event(event)
            if action == "submit":
                self.session.answer(self.field.value)
                self.field.clear()
            return action is not None or event.key not in (pygame.K_ESCAPE,)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for name in self._shown:
                if self.buttons[name].hit(event.pos):
                    self._act(name)
                    return True
        return False

    # ---- the joystick's way in -----------------------------------------------

    @property
    def focused_button(self):
        """The name of the button a press would hit, or None when the card is not
        up (or has, somehow, no buttons on it)."""
        if not self.visible or not self._shown:
            return None
        return self._shown[self.focus_button % len(self._shown)]

    def step_focus(self, direction):
        """Move the focus `direction` buttons along, wrapping. True if it moved.

        Wrapping rather than stopping at the ends: there are two to four buttons on
        this card and they are in one row, so "keep pushing right" reaching the one
        you want is the behaviour a hand expects. Nothing else on the card takes
        left/right while it is up, so there is no other meaning to collide with.
        """
        if not self.visible or not self._shown or not direction:
            return False
        self.focus_button = ((self.focus_button + direction) % len(self._shown))
        return True

    def push_axis(self, x):
        """One frame of the stick's left/right deflection. True if the focus moved.

        A latched step, not a rate: these are four discrete buttons, so one push has
        to mean one button however long it is held, and the stick has to come back
        near centre before it will step again. Same Schmitt trigger, and the same two
        thresholds, as the bead-colouring choice in the panel proper -- see
        control_focus.Choice, which this deliberately borrows rather than inventing a
        second feel for the same gesture.
        """
        from ..control_focus import CHOICE_REARM, CHOICE_STEP
        if not self.visible:
            return False
        if abs(x) < CHOICE_REARM:
            self._axis_armed = True
            return False
        if not self._axis_armed or abs(x) < CHOICE_STEP:
            return False
        self._axis_armed = False
        return self.step_focus(1 if x > 0 else -1)

    def activate_focus(self):
        """Press the focused button. True if anything was pressed."""
        name = self.focused_button
        if name is None:
            return False
        self._act(name)
        return True

    def set_shown(self, names, active="connect"):
        """Declare which buttons are on the card, and re-aim the joystick focus if
        that set has changed. Called from `draw`, which is what lays them out."""
        self._aim_focus(names, active)
        self._shown = tuple(names)

    def _aim_focus(self, names, active):
        """Point the focus at the recommended button whenever the button SET
        changes, and leave it alone otherwise.

        The set changes exactly when the session changes state -- Connect gives way
        to Cancel while it queues, Cancel to Disconnect when it lands -- and at those
        moments an index is not a button any more: hold the number and "the third
        one" is whatever happens to be third now, which is how a hand that was
        resting on Cancel ends up on Disconnect. So each new set starts on the one
        the card itself recommends, which is the one a press was about to mean
        anyway.
        """
        if tuple(names) == tuple(self._shown):
            return
        self.focus_button = (list(names).index(active) if active in names else 0)
        self._axis_armed = True

    def _act(self, name):
        session = self.session
        if name == "connect":
            # One button, two things: get a GPU and put this playground on it, or
            # move the GPU we already hold to this playground. The session decides
            # which, because which it is is a fact about the session (see
            # RemoteSession.connect_playground).
            session.connect_playground(self.playground_key)
        elif name == "cancel":
            session.cancel()
        elif name == "disconnect":
            if self.system is not None:
                self.system.detach()
            session.shutdown()
        elif name == "copy":
            self._copy_report()
        elif name == "close":
            self.visible = False

    def _copy_report(self):
        """Put the whole session report on the clipboard, and on disk as well.

        Both, always: the clipboard is where it is wanted (a chat window, a ticket)
        and the file is what survives the next copy -- and on a machine with no
        clipboard command at all, the file is the only way the report gets out.
        """
        report = self.session.diagnostics()
        copied = clipboard.copy(report)
        path = self.session.save_report()
        lines = len(report.splitlines())
        if copied:
            note = f"copied {lines} lines to the clipboard"
        else:
            note = "could not reach the clipboard"
        if path:
            note += f"  --  saved to {path}"
        self._notice = note
        self._notice_until = time.monotonic() + 8.0

    # ---- drawing ------------------------------------------------------------

    def draw(self, renderer):
        """Draw the card over the sim view. Called by the renderer, before it
        composites and flips, so it lands on top of the 3D scene."""
        if not self.visible or self.session is None:
            return
        session = self.session
        screen = renderer.screen
        font, small = renderer.font, renderer.small_font
        header = renderer.header_font

        width = min(UI(CARD_WIDTH), max(UI(320), renderer.sim_width - UI(40)))
        # Height from the content, not a constant: the card carries a prompt block
        # only while something is being asked, and the log grows into it as the
        # session proceeds. Fixed, it is a third empty at the start and clips the
        # log at the end.
        lines = list(session.log)[-LOG_LINES:]
        error_lines = ([] if session.prompt or not session.error
                       else _wrap(session.error, small, width - UI(36)))
        held = self._held_note()
        height = (UI(58)                               # title and subtitle
                  + (UI(18) if held else 0)            # the held-GPU line
                  + UI(26) + UI(16)                    # state line, progress bar
                  + (UI(64) if session.prompt else 0)
                  + UI(15) * len(error_lines) + (UI(6) if error_lines else 0)
                  + UI(14) * max(len(lines), 3)
                  + UI(78))                            # hint line + button row
        height = min(height, renderer.window_size[1] - UI(40))
        rect = pygame.Rect(0, 0, width, height)
        rect.center = (renderer.sim_width // 2, renderer.window_size[1] // 2)

        # A dim wash over the scene behind it: this is modal, and it should look it.
        wash = pygame.Surface((renderer.sim_width, renderer.window_size[1]),
                              pygame.SRCALPHA)
        wash.fill((0, 0, 0, 150))
        screen.blit(wash, (0, 0))
        pygame.draw.rect(screen, PANEL_BG, rect, border_radius=UI(10))
        pygame.draw.rect(screen, PANEL_DIVIDER, rect, width=UI.w(1),
                         border_radius=UI(10))

        x = rect.x + UI(18)
        w = rect.width - UI(36)
        y = rect.y + UI(14)

        target = session.target
        title = header.render(f"Remote simulation -- {target.label}", True,
                              HEADER_TEXT_COLOR)
        screen.blit(title, (x, y))
        y += title.get_height() + UI(2)
        sub = (f"{target.destination}  |  {target.partition}, {target.gpus} GPU, "
               f"{target.cpus_per_task} cores, {target.time}")
        screen.blit(small.render(sub, True, DIM_TEXT_COLOR), (x, y))
        y += UI(20)

        # THE LINE THAT MAKES THE SWITCH LEGIBLE. Pressing Connect with a GPU
        # already held throws away a running simulation, so what is running has to
        # be on the card -- not inferable from a log line four rows down.
        if held:
            screen.blit(small.render(held, True, OK_COLOR), (x, y))
            y += UI(18)

        # State line: the step it is on, out of how many, and what it is doing.
        step, total = session.progress()
        from ..remote.session import DOWN, FAILED, READY, SWITCH
        color = {READY: OK_COLOR, FAILED: FAIL_COLOR,
                 DOWN: DIM_TEXT_COLOR}.get(session.state, BUSY_COLOR)
        label = session.state.upper()
        if session.busy:
            label = f"{label} ({step}/{total})"
        screen.blit(font.render(label, True, color), (x, y))
        detail = font.render(session.detail, True, TEXT_COLOR)
        screen.blit(detail, (x + UI(130), y))
        y += detail.get_height() + UI(6)

        # A progress bar, because the two slow steps (the queue, and LAMMPS
        # starting) can each take minutes and a static line looks like a hang.
        bar = pygame.Rect(x, y, w, UI(4))
        pygame.draw.rect(screen, PANEL_DIVIDER, bar, border_radius=UI(2))
        if step:
            filled = pygame.Rect(x, y, int(w * step / total), UI(4))
            pygame.draw.rect(screen, color, filled, border_radius=UI(2))
        y += UI(16)

        if session.prompt:
            prompt_box = pygame.Rect(x - UI(6), y - UI(4), w + UI(12), UI(60))
            pygame.draw.rect(screen, (32, 30, 22), prompt_box,
                             border_radius=UI(6))
            pygame.draw.rect(screen, SLIDER_HANDLE_ACTIVE, prompt_box,
                             width=UI.w(1), border_radius=UI(6))
            # The prompt verbatim: it may be a password, a one-time code, or a host
            # key to confirm, and only the thing that asked knows which.
            screen.blit(font.render(session.prompt, True, SLIDER_HANDLE_ACTIVE), (x, y))
            y += UI(22)
            self.field.rect = pygame.Rect(x, y, w, UI(26))
            self.field.draw(screen, font, blink=(time.monotonic() % 1.0) < 0.6)
            y += UI(40)
        elif error_lines:
            for line in error_lines:
                screen.blit(small.render(line, True, FAIL_COLOR), (x, y))
                y += UI(15)
            y += UI(6)

        # The log: this end's steps and the far side's own output, interleaved, which
        # is what makes a failure diagnosable without going to look for a log file.
        for line in lines:
            screen.blit(small.render(line[:96], True, DIM_TEXT_COLOR), (x, y))
            y += UI(14)

        # Buttons, contextual: only the ones that mean something in this state.
        if session.state == SWITCH:
            # A switch cannot be called off halfway: the far side has already thrown
            # the old simulation away, so there is nothing to cancel back to. Close
            # hides the card and it finishes in the background.
            names = ("copy", "close")
        elif session.busy:
            names = ("copy", "cancel")
        elif session.holds_allocation and (self._is_switch()
                                           or session.state != READY):
            # A GPU is held but not streaming THIS playground: the other one's run,
            # or a link here that has gone. Connect moves it -- and Disconnect is on
            # the card too, because "give the GPU back" has to be reachable from
            # every state that is holding one.
            names = ("copy", "disconnect", "connect", "close")
        elif session.state == READY:
            names = ("copy", "disconnect", "close")
        else:
            names = ("copy", "connect", "close")
        # The button says which of its two jobs it is about to do, because "Connect"
        # over a held GPU reads as free and is not.
        self.buttons["connect"].label = ("Move GPU here" if self._is_switch()
                                         else "Connect")
        # The joystick's focus follows the button SET, not the index -- see
        # `set_shown` / `_aim_focus`.
        self.set_shown(names, "connect")
        focused = self.focused_button
        bw, bh, gap = UI(130), UI(30), UI(10)
        bx = rect.right - UI(18) - (len(names) * bw + (len(names) - 1) * gap)
        by = rect.bottom - UI(18) - bh
        for i, name in enumerate(names):
            button = self.buttons[name]
            button.rect = pygame.Rect(bx + i * (bw + gap), by, bw, bh)
            button.draw(screen, font, active=(name == "connect"),
                        focused=(name == focused))

        # On its own line above the buttons, not beside them: with three buttons
        # there is no room left on that row, and a hint that runs under a button is
        # worse than no hint.
        if self._notice and time.monotonic() < self._notice_until:
            hint, tint = self._notice, OK_COLOR
        elif session.prompt:
            hint, tint = "Enter sends the answer. C copies the report.", BUTTON_BORDER
        elif self._is_switch():
            hint = (f"The run on {session.playground_asked} is thrown away; the "
                    f"allocation is not. N hides this panel.")
            tint = BUTTON_BORDER
        else:
            hint = ("Stick or hat left/right picks a button, trigger presses it. "
                    "N hides this panel, C copies the report.")
            tint = BUTTON_BORDER
        screen.blit(small.render(hint, True, tint), (x, by - UI(18)))

    def _is_switch(self):
        """Would Connect move a GPU we already hold, rather than ask for one?"""
        session = self.session
        return bool(session is not None and session.holds_allocation
                    and not session.serves(self.playground_key))

    def _held_note(self):
        """"There is a GPU held, and this is what is on it" -- or None."""
        session = self.session
        if session is None or not session.holds_allocation:
            return None
        if session.serves(self.playground_key):
            return None
        return (f"GPU held: job {session.job_id} on "
                f"{session.node or session.target.label}, running "
                f"{session.playground_asked}")


# What has to match for two remote playgrounds to be able to share one session.
# Not the whole target: the frame rate, the codec and the idle timeout are per-demo
# and are pushed over the link, and the wall clock is whatever the first request
# asked for. These are the fields that describe the LOGIN and the ALLOCATION, which
# is the thing being shared -- if any of them differs, the held session simply is
# not a session for the arriving playground.
_ALLOCATION_FIELDS = ("host", "user", "partition", "gpus", "ntasks",
                      "cpus_per_task", "account", "remote_dir", "env_script",
                      "deploy_dir", "python", "profile", "tunnel")


def _same_allocation(a, b):
    """Whether two RemoteTargets describe the same GPU on the same cluster."""
    if a is None or b is None:
        return a is b
    return all(getattr(a, f, None) == getattr(b, f, None)
               for f in _ALLOCATION_FIELDS)


def _wrap(text, font, width):
    """Greedy word wrap to a pixel width."""
    words, lines, current = str(text).split(), [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if font.size(trial)[0] <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:4]
