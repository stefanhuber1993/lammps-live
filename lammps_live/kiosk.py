"""Kiosk lock: keep an unattended demo machine from being walked away with.

The laptop running the demo may stand at a booth with nobody watching it, and
it is not an ordinary laptop while it does: it is on the institute VPN and holds
an open two-hop SSH session into a compute node. So with `--lock` the app will
not let itself be closed, minimised, hidden or taken out of fullscreen without a
password, and on macOS it asks the window server to take away the ways round
that (Cmd-Tab, the Dock, the menu bar, Force Quit, logout).

WHAT THIS IS NOT: a security boundary against someone determined. Anyone who
can power-cycle the machine, or who reaches a terminal some other way, is past
it. It is a lock on the front door, for visitors -- see the README section on
running unattended for what else to do.

The password is never stored: a salted SHA-256 of it is held for the session.
It comes from `LAMMPS_LIVE_LOCK_PASSWORD`, or is asked for on the terminal at
startup.
"""
import ctypes
import ctypes.util
import getpass
import hashlib
import os
import secrets
import sys
import threading
import time

# NSApplicationPresentationOptions (AppKit). DisableProcessSwitching needs the Dock
# hidden, and HideMenuBar needs HideDock, so these go together or not at all.
_HIDE_DOCK = 1 << 1
_HIDE_MENU_BAR = 1 << 3
_DISABLE_PROCESS_SWITCHING = 1 << 5
_DISABLE_FORCE_QUIT = 1 << 6
_DISABLE_SESSION_TERMINATION = 1 << 7
_DISABLE_HIDE_APPLICATION = 1 << 8
KIOSK_OPTIONS = (_HIDE_DOCK | _HIDE_MENU_BAR | _DISABLE_PROCESS_SWITCHING
                 | _DISABLE_FORCE_QUIT | _DISABLE_SESSION_TERMINATION
                 | _DISABLE_HIDE_APPLICATION)

# Wrong answers allowed before the prompt refuses input for a while.
MAX_TRIES = 5
LOCKOUT_SECONDS = 30.0


def password_from_environment_or_terminal():
    """The lock password: the environment's, or asked for (twice) on the
    terminal. Returns None if none was given."""
    pw = os.environ.get("LAMMPS_LIVE_LOCK_PASSWORD")
    if pw:
        return pw
    if not sys.stdin.isatty():
        return None
    first = getpass.getpass("[lammps-live] kiosk lock -- choose a password: ")
    if not first:
        return None
    if getpass.getpass("[lammps-live] and again: ") != first:
        raise SystemExit("[lammps-live] the two passwords differ -- not starting")
    return first


class KioskLock:
    """The password check, the retry limit, and the macOS lockdown."""

    def __init__(self, password):
        self._salt = secrets.token_bytes(16)
        self._hash = self._digest(password)
        self._failures = 0
        self._locked_until = 0.0
        self._mac_applied = False

    def _digest(self, password):
        return hashlib.sha256(self._salt + password.encode("utf-8")).digest()

    def locked_out_for(self):
        """Seconds left of a lockout after too many wrong answers (0 if none)."""
        return max(0.0, self._locked_until - time.monotonic())

    def check(self, attempt):
        """Whether `attempt` is the password. Too many wrong ones in a row lock
        the prompt for LOCKOUT_SECONDS, during which every attempt is refused."""
        if self.locked_out_for() > 0.0:
            return False
        if secrets.compare_digest(self._digest(attempt), self._hash):
            self._failures = 0
            return True
        self._failures += 1
        if self._failures >= MAX_TRIES:
            self._failures = 0
            self._locked_until = time.monotonic() + LOCKOUT_SECONDS
        return False

    # ---- macOS presentation options ---------------------------------------

    def apply_platform_lockdown(self):
        """On macOS: hide the Dock and menu bar and disable Cmd-Tab, Force Quit,
        hide and logout for this app's lifetime. Nothing elsewhere. Best effort:
        returns whether it took."""
        # Not under a headless SDL driver: there is no window to protect, and a
        # watchdog armed in a test run would end the test process.
        if (sys.platform != "darwin"
                or os.environ.get("SDL_VIDEODRIVER") == "dummy"):
            return False
        ok = _set_presentation_options(KIOSK_OPTIONS)
        self._mac_applied = ok
        return ok

    def release_platform_lockdown(self):
        """Undo apply_platform_lockdown (also undone by the process exiting)."""
        if self._mac_applied:
            _set_presentation_options(0)
            self._mac_applied = False

    @property
    def platform_locked(self):
        return self._mac_applied


class Watchdog:
    """Ends the process if the main loop stops ticking.

    With Force Quit and Cmd-Tab disabled, a hung app would take the whole
    machine with it: nothing on screen could close it. So while the lockdown is
    on, a thread checks that the main loop still calls `tick`, and after
    `timeout` seconds of silence exits the process -- which is also what gives
    the Dock, the menu bar and the app switcher back. A held GPU is then left to
    the server's own idle timeout.
    """

    def __init__(self, timeout=90.0, on_fire=None):
        self.timeout = timeout
        self._last = time.monotonic()
        self._stop = threading.Event()
        self._on_fire = on_fire
        self._thread = threading.Thread(target=self._run, name="kiosk-watchdog",
                                        daemon=True)
        self._thread.start()

    def tick(self):
        self._last = time.monotonic()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.wait(2.0):
            if time.monotonic() - self._last > self.timeout:
                print(f"[lammps-live] main loop silent for {self.timeout:.0f} s "
                      f"with the kiosk lock on -- exiting so the machine is "
                      f"usable again", flush=True)
                if self._on_fire is not None:
                    try:
                        self._on_fire()
                    except Exception:                  # noqa: BLE001
                        pass
                os._exit(3)


def _set_presentation_options(options):
    """[[NSApplication sharedApplication] setPresentationOptions:options] via the
    Objective-C runtime, so no PyObjC dependency. Must run on the main thread."""
    try:
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        ctypes.cdll.LoadLibrary(ctypes.util.find_library("AppKit"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        send = objc.objc_msgSend
        send.restype = ctypes.c_void_p
        send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        app = send(objc.objc_getClass(b"NSApplication"),
                   objc.sel_registerName(b"sharedApplication"))
        if not app:
            return False
        set_opts = ctypes.cast(objc.objc_msgSend, ctypes.CFUNCTYPE(
            None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong))
        set_opts(app, objc.sel_registerName(b"setPresentationOptions:"),
                 ctypes.c_ulong(options))
        get_opts = ctypes.cast(objc.objc_msgSend, ctypes.CFUNCTYPE(
            ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p))
        now = get_opts(app, objc.sel_registerName(b"presentationOptions"))
        return int(now) == int(options)
    except Exception as exc:                           # noqa: BLE001
        print(f"[lammps-live] kiosk: could not set presentation options: {exc}")
        return False
