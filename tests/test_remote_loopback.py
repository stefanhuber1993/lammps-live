"""The whole remote pipeline, on one machine: a real LAMMPS server, a real socket.

This is the test that says the split works. It builds a small remote playground,
serves it with the actual `FrameServer` over the actual protocol, connects the
actual `RemoteSystem`, and then checks the things the demo depends on:

  * the client draws what the server integrated, to the codec's precision;
  * a slider reaches the far end and changes the energy there;
  * Play/Pause/Reset are the server's state, not a local pretence;
  * the observables and energy panels -- which run on the CLIENT -- are populated
    from received frames;
  * every readout is safe before the first frame and after the link drops.

Kept at 900 beads so it runs on one CPU core in a couple of seconds. The physics is
the 10k playground's, scaled down; what is being tested is the pipe.
"""
import socket
import threading
import time

import numpy as np
import pytest

from lammps_live.forcefields.mesomem import ISO
from lammps_live.playground import jitter
from lammps_live.playground.system import JITTER_KEY
from lammps_live.remote import RemoteTarget, protocol
from lammps_live.remote.client import FrameLink, LinkClosed
from lammps_live.remote import server as server_mod
from lammps_live.remote.server import FrameServer

pytest.importorskip("lammps")

TOKEN = "test-token-not-a-secret"

PLAYGROUND_SOURCE = '''
"""A small remote playground, for the loopback test."""
from lammps_live.playground import Playground, random_fill
from lammps_live.remote import RemoteTarget

PLAYGROUND = Playground(
    name="loopback assembly",
    description="900 beads, served locally",
    force_field="mesomem",
    scenario=random_fill(n=900, box=16.0, k_upright=0.6, center_accel=0.05),
    mode="sim",
    observables=["nematic_S", "coordination"],
    temperature_default=0.2,
    trajectory_smoothing=True,
    seed=4242,
    remote=RemoteTarget(host="localhost", profile="local", port=0, local_port=0),
)
'''


@pytest.fixture(scope="module")
def playground_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("remote") / "loopback_playground.py"
    path.write_text(PLAYGROUND_SOURCE)
    return str(path)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(playground_file):
    """A FrameServer on its own thread, on a free port."""
    port = _free_port()
    srv = FrameServer(playground=playground_file, profile="local", port=port,
                      bind="127.0.0.1", token=TOKEN, fps=0.0, verbose=False)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    # Wait for the listener rather than sleeping a guessed interval.
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    else:
        pytest.fail("the server never started listening")
    yield srv, port
    # Stop the loop, THEN release LAMMPS: closing an instance while its thread is
    # inside `run` segfaults the interpreter (see FrameServer.stop).
    srv.stop()
    thread.join(timeout=30.0)
    assert not thread.is_alive(), "the server thread did not stop"
    srv.close()


@pytest.fixture
def system(server, playground_file):
    """A connected RemoteSystem, torn down after each test."""
    _srv, port = server
    from lammps_live.playground import registry
    sys_ = registry.build(playground_file,
                          remote_override=RemoteTarget(host="127.0.0.1",
                                                       local_port=port,
                                                       profile="local"))
    sys_.connect("127.0.0.1", port, TOKEN, timeout=30.0)
    yield sys_
    sys_.close()


def _advance(system, frames=6, timeout=30.0, draw=False):
    """Step until `frames` frames have actually been taken in.

    `draw=True` also reads the scene once per frame, which is what the app does and
    what the trajectory-smoothing filter needs -- it advances once per call to the
    render readout, not once per received frame.

    Waits for the measuring thread on the way out. The app never does that -- the
    analysis runs on its own clock precisely so no frame waits for it (see
    FrameAnalysis) -- but a test that reads an observable it just caused has to
    know the pass it is asking about has happened."""
    seen = 0
    deadline = time.monotonic() + timeout
    while seen < frames and time.monotonic() < deadline:
        before = system._seq
        system.step(20)
        if system._seq != before:
            seen += 1
            if draw:
                system.get_positions_3d()
    assert seen >= frames, f"only {seen} of {frames} frames arrived"
    assert system.wait_for_analysis(timeout=10.0), "the analysis never caught up"


# --- the pipe -----------------------------------------------------------------

def test_disconnected_system_is_safe_to_draw(playground_file):
    """Selecting the playground with nothing running must not raise anywhere. This
    is the state the app comes up in, before the connect panel has done anything."""
    from lammps_live.playground import registry
    sys_ = registry.build(playground_file)
    try:
        assert not sys_.connected
        assert sys_.get_positions_3d()[1].shape == (0, 3)
        assert sys_.get_dipoles_3d().shape[0] == 0
        assert sys_.get_thermo_state() == (0.0, 0.0, 0.0, 0.0, 0.0)
        assert sys_.get_total_potential_terms() is None
        assert sys_.get_rdf() is None
        assert "not connected" in " ".join(sys_.get_hud_lines())
        assert sys_.get_all_positions()[1].shape == (0, 2)
        # None, NOT an array of zeros: zero is the TOP of the energy ramp, so
        # zeros paint every bead white -- "maximally strained" for a scene that
        # simply has not been sent its energies yet. None means no data, and the
        # renderer keeps the director banding until some arrive.
        assert sys_.get_bead_energies() is None
        assert sys_.get_box_bounds_3d()[1] == pytest.approx(8.0)
        sys_.step(20)          # returns, does not raise
        sys_.set_extra_param("k_tilt", 4.0)
        sys_.set_target_temp(0.3)
        sys_.reset()
    finally:
        sys_.close()


def test_handshake_rejects_a_bad_token(server):
    _srv, port = server
    with pytest.raises(LinkClosed, match="bad token"):
        FrameLink.connect("127.0.0.1", port, "wrong", timeout=10.0)


def test_handshake_rejects_a_version_mismatch(server):
    """Hand-rolled, not monkeypatched: both ends are in this process, so patching
    protocol.VERSION would move them together and prove nothing."""
    _srv, port = server
    with socket.create_connection(("127.0.0.1", port), timeout=10.0) as sock:
        sock.sendall(protocol.pack({"t": "hello", "version": protocol.VERSION + 99,
                                    "token": TOKEN}))
        header, _payload = protocol.recv_message(sock)
    assert header["t"] == "error"
    assert "version mismatch" in header["msg"]


def test_a_silent_connection_cannot_wedge_the_server(server, monkeypatch):
    """The server serves one client at a time, so a connection that opens and says
    nothing must time out rather than blocking every later one. On a shared cluster
    network a stray port scan is all it takes."""
    _srv, port = server
    monkeypatch.setattr(server_mod, "HANDSHAKE_TIMEOUT", 1.0)
    silent = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        # A real client, arriving while the silent one is still being waited on.
        link = FrameLink.connect("127.0.0.1", port, TOKEN, timeout=30.0)
        try:
            assert link.welcome["natoms"] == 900
        finally:
            link.close()
    finally:
        silent.close()


def test_welcome_describes_the_far_side(system):
    welcome = system.link.welcome
    assert welcome["natoms"] == 900
    assert welcome["profile"] == "local"
    assert welcome["box"]["periodic"] == [True, True, True]
    # The client adopts the server's cell rather than trusting its own guess.
    assert system.box.lengths[0] == pytest.approx(16.0, abs=1e-6)


def test_frames_arrive_and_are_the_servers_state(system, server):
    srv, _port = server
    system.set_playing(True)
    _advance(system, frames=4)

    ids, positions, is_puller = system.get_positions_3d()
    assert len(positions) == 900
    assert np.isfinite(positions).all()
    assert not is_puller.any()
    assert ids[0] == 1 and ids[-1] == 900
    # Inside the padded cell, and spread through it rather than collapsed.
    assert positions.min() > -9.5 and positions.max() < 9.5
    assert positions.std() > 2.0

    # The same coordinates the server holds, to the codec's precision. Three
    # things have to be arranged before that comparison means anything:
    #
    #  * PAUSED, and with a frame taken since. The server is uncapped in this test
    #    and would otherwise have integrated past the frame being checked, which
    #    reads as a codec error of a tenth of a sigma.
    #  * LIVELINESS OFF. The drawn state is deliberately not the received state --
    #    the synthetic rattle (playground/jitter.py) moves every bead by design,
    #    so leaving it on turns this into an assertion about the fill's amplitude
    #    instead of the codec's precision. Same reasoning as the Smoothing slider,
    #    which is off by default and so never had to be said out loud.
    #  * A FRAME DRAWN AFTER the slider moved, so the render cache is rebuilt
    #    without the wobble rather than handing back the last filled one.
    system.set_playing(False)
    system.set_extra_param(JITTER_KEY, 0.0)
    _advance(system, frames=3)
    remote_state = srv.system.current_state()
    drawn = system.get_positions_3d()[1]
    assert np.abs(drawn - remote_state.positions).max() < 5e-3
    system.set_extra_param(JITTER_KEY, jitter.DEFAULT_JITTER)
    system.set_playing(True)

    directors = system.get_dipoles_3d()
    assert np.allclose(np.linalg.norm(directors, axis=1), 1.0, atol=1e-4)

    temp, press, ke, pe, etotal = system.get_thermo_state()
    assert 0.0 < temp < 2.0
    assert pe < 0.0                      # beads are attracting
    assert system.get_sim_time() > 0.0


def test_the_client_does_the_measuring(system):
    """The observables and the energy panel are computed here, from the frames --
    the server runs no analysis at all."""
    system.set_playing(True)
    _advance(system, frames=10)

    values = system.analysis.values()
    assert set(values) == {"nematic_S", "coordination"}
    assert 0.0 <= values["nematic_S"] <= 1.0
    assert values["coordination"] > 0.0

    panel = system.get_total_potential_terms()
    assert panel is not None
    title, terms, scale = panel
    assert len(terms) == 3
    # By the constant, not by the string: the label is display text and has been
    # reworded once already (it is "van der Waals ..." now), which broke this line.
    assert dict(terms)[ISO] < 0.0
    hud = " ".join(system.get_hud_lines())
    assert "nematic order" in hud and "MB/s" in hud


def test_a_slider_reaches_the_far_end(system, server):
    """k_tilt is the demo's whole point, so it is the one to prove: the value has
    to reach the far side's pair style, not just this side's panel."""
    srv, _port = server
    system.set_playing(True)
    _advance(system, frames=2)

    system.set_extra_param("k_tilt", 33.0)
    _advance(system, frames=3)
    assert srv.system.params["k_tilt"] == pytest.approx(33.0)
    # And it is in the coefficients LAMMPS is running, not only in the ParamSet.
    coeff = srv.system.force_field.coeff_commands(srv.system.params)[0]
    assert " 33.0 " in coeff

    # The clamped value is what is sent: wc is capped at rc, and the far end must
    # run the same number the local energy decomposition uses.
    system.set_extra_param("rc", 1.8)
    system.set_extra_param("wc", 3.0)
    _advance(system, frames=3)
    assert srv.system.params["wc"] == pytest.approx(1.8)


def test_play_pause_is_the_servers_state(system, server):
    srv, _port = server
    system.set_playing(False)
    _advance(system, frames=2)          # frames keep coming while paused
    assert srv.playing is False
    held = srv.system.get_sim_time()
    _advance(system, frames=3)
    assert srv.system.get_sim_time() == pytest.approx(held)

    system.set_playing(True)
    _advance(system, frames=3)
    assert srv.playing is True
    assert srv.system.get_sim_time() > held


def test_paused_readouts_still_take_in_the_current_frame(system):
    """After connecting, the app is paused and never calls step(). The scene must
    still fill in, or a fresh connection looks like an empty box."""
    system.set_playing(False)
    deadline = time.monotonic() + 15.0
    while system._state is None and time.monotonic() < deadline:
        system.get_thermo_state()       # a readout, not a step
        time.sleep(0.01)
    assert system._state is not None
    assert len(system.get_positions_3d()[1]) == 900


def test_reset_rebuilds_on_the_far_side(system, server):
    srv, _port = server
    system.set_playing(True)
    _advance(system, frames=4)
    before = srv.system.current_state().positions.copy()
    advanced = srv.system.get_sim_time()
    assert advanced > 0.0

    system.reset()
    # The rebuild happens on the far side and takes as long as it takes, so the HUD
    # says so rather than sitting on the last frame of the old run -- which is what
    # a Reset that did nothing looks like.
    assert any("rebuilding" in line for line in system.get_hud_lines())
    system.set_playing(True)
    _advance(system, frames=4)
    assert srv.system.get_sim_time() < advanced      # the clock restarted
    after = srv.system.current_state().positions
    assert np.abs(after - before).max() > 1.0        # a genuinely new configuration
    # ... and the notice clears itself once frames from the new run arrive.
    assert not any("rebuilding" in line for line in system.get_hud_lines())


def test_a_new_run_is_recognised_from_a_standing_start(playground_file):
    """The rule that clears the rebuild notice, and the hole that used to be in it.

    The far side restarts its frame numbering on a rebuild, so the client knows the
    new run by a frame numbered at or below the one it was on. With nothing ingested
    yet that number is 0, and no sequence number is ever <= 0 -- so a Reset pressed
    before the first frame landed left the notice on forever, over a run that was
    streaming perfectly well. Narrow, and exactly where somebody who has just
    connected and does not like what they see is standing.
    """
    from lammps_live.playground import registry
    sys_ = registry.build(playground_file)
    try:
        sys_._seq = 0                      # before the first frame of all
        assert sys_._is_new_run(1)         # <- used to be False, and latched
        sys_._seq = 37                     # mid-run, the ordinary case
        assert sys_._is_new_run(1)
        assert not sys_._is_new_run(38)    # just the next frame of the same run
    finally:
        sys_.close()


def test_a_reset_that_never_went_out_does_not_claim_to_be_rebuilding(system,
                                                                     monkeypatch):
    """`send` answers False on a socket that has died rather than raising -- a
    slider must not be able to end the app -- so latching the notice before asking
    left the HUD reporting a rebuild that nothing on the far side had heard of."""
    monkeypatch.setattr(system.link, "send", lambda message: False)
    system.reset()
    assert not system._resetting
    assert not any("rebuilding" in line for line in system.get_hud_lines())


def test_the_rebuild_notice_counts_the_seconds(system):
    """The only two explanations available to somebody watching a picture that has
    stopped are "slow" and "hung", and they look identical. A number that is going
    up is the difference -- and at this playground's size a rebuild there is a whole
    LAMMPS setup, so slow is the common answer."""
    system.reset()
    line = next(l for l in system.get_hud_lines() if "rebuilding" in l)
    assert line.rstrip().endswith("s"), line


def test_energies_are_only_sent_when_the_colouring_asks(system, server):
    srv, _port = server
    system.set_playing(True)
    _advance(system, frames=3)
    assert srv.want_energies is False

    for _ in range(4):
        system.get_bead_energies()       # what the renderer does when it paints
        _advance(system, frames=1)
    assert srv.want_energies is True
    energies = system.get_bead_energies()
    assert len(energies) == 900
    assert energies.min() < 0.0

    # Stop asking (the toggle went back to directors) and the far end stops sending.
    # Waited for rather than asserted on the spot: the request is sent at the START
    # of a step, so the frame that comes back may be one the server produced before
    # it drained the message. Four frames is normally enough and occasionally is not.
    deadline = time.monotonic() + 10.0
    while srv.want_energies and time.monotonic() < deadline:
        _advance(system, frames=1)
    assert srv.want_energies is False


def test_smoothing_filters_the_drawn_state_only(system):
    system.set_playing(True)
    _advance(system, frames=4)
    raw = system._state.positions.copy()
    assert np.abs(system.get_positions_3d()[1] - raw).max() == 0.0

    system.set_extra_param("view_smoothing", 2.0)
    _advance(system, frames=4, draw=True)
    drawn = system.get_positions_3d()[1]
    live = system._state.positions
    assert np.abs(drawn - live).max() > 0.0        # the picture lags the physics
    # ... and the measurement does not: the analysis ran on the unsmoothed state.
    assert system.analysis.pairs is not None


def test_smoothing_also_steadies_the_energy_colouring(system):
    """The colour is drawn from the same rattle the positions are.

    Smoothing the beads' positions and leaving their colours at full frame rate
    trades a wiggle for a twinkle: each bead's energy is a sum over neighbours that
    are still moving, so it swings frame to frame far more than the configuration
    it is painting. The filter is the same one, over the same window.
    """
    system.set_playing(True)
    for _ in range(6):                       # get the far side sending energies
        system.get_bead_energies()
        _advance(system, frames=1)
    assert system.get_bead_energies() is not None

    system.set_extra_param("view_smoothing", 2.0)
    for _ in range(6):
        system.get_bead_energies()
        _advance(system, frames=1, draw=True)
    drawn = system.get_bead_energies()
    live = system._energies
    assert drawn is not None and live is not None
    assert np.abs(drawn - live).max() > 0.0, "the colour lags the physics too"

    # And it is the DRAWN copy only: the raw frame is what everything that
    # measures still reads.
    system.set_extra_param("view_smoothing", 0.0)
    system.get_bead_energies()
    _advance(system, frames=1, draw=True)
    assert np.array_equal(system.get_bead_energies(), system._energies)


def test_a_client_waits_while_the_server_builds(server, playground_file):
    """The bug that looked like a broken tunnel: for the whole of LAMMPS' setup the
    server cannot answer, and the client used to give up at 15s, drop the socket and
    retry -- which made the server throw the half-built simulation away and start
    over, so the retry could not succeed either. The server now says it is building
    first, and that message is what turns the wait into a wait.
    """
    srv, port = server
    # A server with nothing built yet is the state a fresh allocation is in. The
    # build it then does is real, and the server keeps it -- so the instance it had
    # before is handed back here and closed, rather than left alive alongside it.
    previous, srv.system = srv.system, None
    notices = []
    link = FrameLink.connect("127.0.0.1", port, TOKEN, timeout=5.0,
                             on_notice=notices.append)
    try:
        assert notices and "building" in notices[0]
        assert link.welcome.get("natoms") == 900
        assert srv.system is not previous, "it really did build one"
    finally:
        link.close()
        if previous is not None and previous is not srv.system:
            previous.close()


def test_frames_are_dropped_not_queued(system):
    """A client that stops reading for a moment must resume at the present, not
    work through a backlog."""
    system.set_playing(True)
    _advance(system, frames=2)
    time.sleep(0.5)                     # a hitch: nothing consumed
    system.step(20)
    assert system.link.dropped > 0
    fresh = system._seq
    system.step(20)
    assert system._seq >= fresh


def test_a_dropped_link_leaves_a_drawable_system(system):
    system.set_playing(True)
    _advance(system, frames=3)
    last = system.get_positions_3d()[1].copy()
    system.link.sock.close()            # the tunnel dies
    deadline = time.monotonic() + 5.0
    while system.connected and time.monotonic() < deadline:
        system.step(20)
    assert not system.connected
    assert system.link_error()
    # The last frame is still there to draw, and nothing raises.
    assert np.array_equal(system.get_positions_3d()[1], last)
    assert "not connected" in " ".join(system.get_hud_lines())
    system.step(20)
    system.set_extra_param("k_tilt", 12.0)


def test_switching_playground_and_back_keeps_the_servers_simulation(server,
                                                                    playground_file):
    """Going off to show another playground must not cost the run, or the GPU.

    The client closes its link on the way out; the server holds the simulation where
    it was and waits for the next client (`serve_forever`), which is what lets the
    app come back to the same coarsened box over the tunnel it never closed -- see
    RemoteSession.reopen_link and RemotePanel.attach_system. What is checked here is
    the half that involves an actual socket: that a second client picks the run up
    rather than starting one.
    """
    srv, port = server
    from lammps_live.playground import registry

    def client():
        sys_ = registry.build(playground_file,
                              remote_override=RemoteTarget(host="127.0.0.1",
                                                           local_port=port,
                                                           profile="local"))
        sys_.connect("127.0.0.1", port, TOKEN, timeout=30.0)
        return sys_

    first = client()
    first.set_playing(True)
    _advance(first, frames=3)
    t_away = first.get_sim_time()
    assert t_away > 0.0
    first.close()                       # what the app does on a playground switch

    # Nobody is watching, so the far side stops integrating rather than burning a
    # GPU on frames nobody reads.
    deadline = time.monotonic() + 5.0
    while srv.playing and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not srv.playing

    second = client()
    try:
        # The welcome carries the server's own clock: this is the run that was left,
        # not a fresh box (a rebuild would put it back to zero).
        assert second.get_sim_time() >= t_away
        second.set_playing(True)
        _advance(second, frames=3)
        assert second.get_sim_time() > t_away
    finally:
        second.close()


def test_a_client_can_move_the_server_to_another_playground(server, tmp_path):
    """One allocation, two demos -- the half of it that crosses a socket.

    A client whose hello names a playground other than the one loaded gets that one
    built in its place, on the same server, through the same port: no second
    allocation, and no second queue wait. Switching back builds the first again --
    and, since parking a run is now cheaper than losing it, resumes where it left
    off (see test_a_switch_parks_the_run_it_leaves for that half).
    """
    srv, port = server
    from lammps_live.playground import registry

    other = tmp_path / "other_loopback.py"
    # The same deck at a different size, so "which one is loaded" is a fact the
    # client can read off the welcome rather than take on trust.
    other.write_text(PLAYGROUND_SOURCE.replace("n=900", "n=420")
                                      .replace("loopback assembly", "smaller"))
    first_ref = srv.playground_ref

    def client(ref):
        sys_ = registry.build(str(ref),
                              remote_override=RemoteTarget(host="127.0.0.1",
                                                           local_port=port,
                                                           profile="local"))
        sys_.attach(FrameLink.connect("127.0.0.1", port, TOKEN, timeout=60.0,
                                      playground=str(ref)))
        return sys_

    moved = client(other)
    try:
        assert srv.playground_ref == str(other)
        assert moved.natoms == 420
        assert moved.link.welcome["playground"] == str(other)
        assert moved.get_sim_time() == 0.0, "a run built for the first time is at zero"
        moved.set_playing(True)
        _advance(moved, frames=3)
    finally:
        moved.close()

    back = client(first_ref)
    try:
        assert srv.playground_ref == first_ref
        assert back.natoms == 900
        back.set_playing(True)
        _advance(back, frames=3)
    finally:
        back.close()


def test_a_switch_parks_the_run_it_leaves_and_resumes_it_on_the_way_back(server,
                                                                        tmp_path):
    """Cycling between two demos does not cost either of them.

    This used to be the honest price of one GPU serving two playgrounds: the run you
    switched away from was thrown out, and four minutes of coarsening with it. It is
    not a price worth paying any more -- the state is a few megabytes of arrays and
    building a fresh instance to scatter them into is about a second (see
    RandomFill.build) -- so the server parks it and puts it back.

    The thing that proves it is the SIMULATED TIME. A resumed run carries its own
    clock; a rebuilt one starts at zero. And the coordinates have to come back too,
    or "resumed" means only that a number was copied.
    """
    srv, port = server
    from lammps_live.playground import registry

    other = tmp_path / "parking_other.py"
    other.write_text(PLAYGROUND_SOURCE.replace("n=900", "n=360")
                                      .replace("loopback assembly", "parking other"))
    first_ref = srv.playground_ref

    def client(ref):
        sys_ = registry.build(str(ref),
                              remote_override=RemoteTarget(host="127.0.0.1",
                                                           local_port=port,
                                                           profile="local"))
        sys_.attach(FrameLink.connect("127.0.0.1", port, TOKEN, timeout=60.0,
                                      playground=str(ref)))
        return sys_

    # Run the first one for a while, then note exactly where it got to.
    here = client(first_ref)
    try:
        here.set_playing(True)
        _advance(here, frames=8)
        parked_time = srv.system.get_sim_time()
        parked_x = srv.system.frame_state().positions.copy()
        assert parked_time > 0.0
    finally:
        here.close()

    # Away, which is what parks it.
    away = client(other)
    try:
        assert srv.playground_ref == str(other)
        assert away.natoms == 360
        assert first_ref in srv._parked, "the run we left was not put aside"
        away.set_playing(True)
        _advance(away, frames=3)
    finally:
        away.close()

    # And back. Same clock, same coordinates, and the snapshot consumed.
    back = client(first_ref)
    try:
        assert srv.playground_ref == first_ref
        assert srv.system.get_sim_time() == pytest.approx(parked_time)
        assert np.allclose(srv.system.frame_state().positions, parked_x)
        assert first_ref not in srv._parked, "a resumed snapshot should be consumed"
        # And it is a working simulation, not just restored arrays.
        back.set_playing(True)
        _advance(back, frames=3)
        assert srv.system.get_sim_time() > parked_time
    finally:
        back.close()


def test_a_parked_run_that_no_longer_fits_is_dropped_not_forced(server, tmp_path):
    """A snapshot of a different bead count must be refused.

    It can happen for one honest reason -- the playground file was edited between
    switches -- and scattering it in anyway would produce a scene that is neither
    run. The build that was skipping its settle on the strength of the resume then
    has to be done again properly, or the state served is the scenario's made-up
    starting geometry (see FrameServer.build).
    """
    srv, port = server
    from lammps_live.playground import registry

    other = tmp_path / "misfit_other.py"
    other.write_text(PLAYGROUND_SOURCE.replace("n=900", "n=300")
                                      .replace("loopback assembly", "misfit"))
    ref = srv.playground_ref

    def client(target_ref):
        sys_ = registry.build(str(target_ref),
                              remote_override=RemoteTarget(host="127.0.0.1",
                                                           local_port=port,
                                                           profile="local"))
        sys_.attach(FrameLink.connect("127.0.0.1", port, TOKEN, timeout=60.0,
                                      playground=str(target_ref)))
        return sys_

    here = client(ref)
    try:
        here.set_playing(True)
        _advance(here, frames=3)
    finally:
        here.close()
    away = client(other)
    away.close()
    assert ref in srv._parked
    # Forge a mismatch, which is what an edited playground would produce.
    srv._parked[ref]["natoms"] = 12

    back = client(ref)
    try:
        assert srv.playground_ref == ref
        assert srv.system.natoms == 900
        assert srv.system.get_sim_time() == 0.0, "a refused resume is a fresh run"
        back.set_playing(True)
        _advance(back, frames=3)
        assert srv.system.get_sim_time() > 0.0
    finally:
        back.close()


def test_a_client_that_names_nothing_leaves_the_server_alone(server, system):
    """The CLI path sends no playground, and a server started with --playground must
    not be second-guessed by it -- that is the difference between "connect to what is
    running" and "put this on the GPU"."""
    srv, _port = server
    loaded = srv.playground_ref
    _advance(system, frames=2)
    assert srv.playground_ref == loaded


def test_reset_disarms_a_dangerous_value_before_the_rebuild_can_meet_it(system, server):
    """The failure that cost an A100, and the fix that makes it unreachable.

    A `zeta` below 1 is legal to the CPU pair style and rejected by the Kokkos one,
    and neither notices until a rebuild validates the coefficients -- so the value
    streamed fine and then killed the server on Reset, which cancelled its own
    allocation on the way out. The recovery for that was a fallback ladder inside
    the rebuild (see PlaygroundSystem._rebuild), and it still exists.

    But Reset now puts every parameter back to what the playground declares BEFORE
    it rebuilds, which is a better answer than recovering: the rebuild never sees
    the value that would have killed it. This pins both halves -- the bad value
    really does reach the far side and really is gone by the time anything is
    rebuilt with it -- by injecting the Kokkos rejection into the far side's force
    field (the local build has no such check) and watching whether it ever fires.
    """
    srv, _port = server
    system.set_playing(True)
    _advance(system, frames=2)

    far = srv.system
    good = float(far.params["zeta"])
    real = far.force_field.pair_commands
    saw_bad = []

    def sabotaged(params):
        if float(params["zeta"]) < 1.0:
            saw_bad.append(float(params["zeta"]))
            return real(params) + ["pair_coeff 1 1 nonsense"]
        return real(params)

    far.force_field.pair_commands = sabotaged
    try:
        system.set_extra_param("zeta", 0.4)
        _advance(system, frames=3)
        assert far.params["zeta"] == pytest.approx(0.4), "the bad value did travel"
        # Not through `pair_commands` yet: `zeta` is not a HOT_RESTYLE parameter, so
        # a live change re-issues the coefficients and never the pair style. That is
        # exactly why the old bug was invisible until something rebuilt.
        assert not saw_bad

        system.reset()                       # the Reset button, over the wire
        _advance(system, frames=6)
    finally:
        far.force_field.pair_commands = real

    # Nothing was rebuilt with the killer: it was put back first.
    assert not saw_bad, f"the rebuild issued zeta={saw_bad} after Reset"
    # The session survived: socket, server thread and simulation all still there.
    assert system.connected
    assert srv.system is not None and srv.system.lmp is not None
    # And both ends are back at the declared value, so the app's slider and the far
    # side agree without anyone having to be told about a fault.
    assert float(srv.system.params["zeta"]) == pytest.approx(good)
    assert float(system.params["zeta"]) == pytest.approx(good)


def test_a_rebuild_that_fails_is_reported_and_the_server_survives(system, server):
    """The fallback ladder, end to end, for the cases Reset's parameter restore
    cannot pre-empt: a build that fails for a reason the current values do not
    explain.

    Injected as a ONE-SHOT failure -- the first `pair_commands` of the rebuild is
    poisoned and the next is not -- because that is the shape of what the ladder
    can actually recover from, and because with the declared values already restored
    a value-dependent failure would fail every rung and be correctly fatal.

    What must be true afterwards: the link is still up, the far side is still
    integrating, and this end has been TOLD -- otherwise the picture quietly
    describes a different run from the one it is showing.
    """
    srv, _port = server
    system.set_playing(True)
    _advance(system, frames=2)

    far = srv.system
    real = far.force_field.pair_commands
    remaining = [1]

    def poison_once(params):
        if remaining and remaining.pop():
            return real(params) + ["pair_coeff 1 1 nonsense"]
        return real(params)

    far.force_field.pair_commands = poison_once
    try:
        system.reset()
        fault = None
        deadline = time.monotonic() + 30.0
        while fault is None and time.monotonic() < deadline:
            _advance(system, frames=1)
            fault = system.take_fault()
        assert fault is not None, "the client never heard about the failure"
    finally:
        far.force_field.pair_commands = real

    assert system.connected
    assert srv.system is not None and srv.system.lmp is not None
    assert not fault.fatal
    assert "nonsense" in fault.detail
    # Whatever the ladder had to put back, both ends agree on it -- otherwise the
    # panel describes coefficients nobody is running.
    for name, value in fault.reverted.items():
        assert float(system.params[name]) == pytest.approx(value)
        assert float(srv.system.params[name]) == pytest.approx(value)

    # Still integrating.
    system.set_playing(True)
    before = srv.system.get_sim_time()
    _advance(system, frames=4)
    assert srv.system.get_sim_time() >= before
    assert system.take_fault() is None       # shown once, then gone
