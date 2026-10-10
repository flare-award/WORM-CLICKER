"""End-to-end tests of the automation loop with the offline simulator."""

import threading
import time

import pytest

from wormclicker.automation import AutomationEngine, Status
from wormclicker.capture import CaptureError
from wormclicker.config import AppConfig
from wormclicker.geometry import Rect
from wormclicker.mouse import DryRunMouse, MouseError
from wormclicker.region import RegionProvider
from wormclicker.simulator import SimCapture, SimMouse, SimulatedGame

# secondary monitor on the LEFT of the primary one -> negative coordinates
ORIGIN = (-1280, -100)
MONS = [Rect(-1280, -100, 1280, 720), Rect(0, 0, 1920, 1080)]


def build(game=None, lag=0, cfg=None, mouse=None, capture=None, area="monitor:0"):
    game = game or SimulatedGame(1280, 720, seed=7)
    cap = capture or SimCapture(game, ORIGIN, lag_frames=lag, monitors=MONS)
    sim_mouse = mouse or SimMouse(game, ORIGIN)
    cfg = cfg or AppConfig(capture_area=area)
    eng = AutomationEngine(cfg, capture_factory=lambda: cap, mouse_factory=lambda dry: sim_mouse,
                           region_provider=RegionProvider(area, lambda: MONS), cursor_fn=lambda: None)
    return eng, game, cap, sim_mouse


def run_for(eng, seconds):
    assert eng.start()
    time.sleep(seconds)
    eng.stop()


@pytest.fixture
def cleanup():
    engines = []
    yield engines
    for e in engines:
        e.shutdown()


def test_hunts_worm_on_negative_coordinate_monitor(cleanup):
    eng, game, cap, mouse = build()
    cleanup.append(eng)
    run_for(eng, 1.5)
    assert game.hits >= 20
    assert game.misses <= max(1, game.hits // 50)
    assert all(x < 0 for x, _, _ in mouse.clicks)          # clicks really on the left monitor
    s = eng.stats.snapshot()
    assert s.clicks == len(mouse.clicks)
    assert s.jumps >= game.hits - 2


@pytest.mark.parametrize("latency,lag", [(0.03, 0), (0.03, 2), (0.0, 3)])
def test_never_clicks_old_position_with_render_latency(cleanup, latency, lag):
    game = SimulatedGame(1280, 720, seed=3, render_latency_s=latency)
    eng, game, cap, mouse = build(game=game, lag=lag)
    cleanup.append(eng)
    run_for(eng, 1.5)
    assert game.hits >= 10
    assert game.clicks_on_old_spot == 0
    assert game.misses <= 1
    assert eng.stats.snapshot().stale_frames > 0       # stale frames were seen and skipped


def test_no_clicks_while_worm_hidden(cleanup):
    eng, game, cap, mouse = build()
    cleanup.append(eng)
    game.set_visible(False)
    assert eng.start()
    time.sleep(0.6)
    assert len(mouse.clicks) == 0
    assert game.clicks_while_hidden == 0
    assert eng.status == Status.SEARCHING
    game.set_visible(True)
    time.sleep(0.4)
    eng.stop()
    assert game.hits >= 1


def test_temporary_disappearance_counts_lost_and_reacquired(cleanup):
    eng, game, cap, mouse = build()
    cleanup.append(eng)
    assert eng.start()
    time.sleep(0.3)
    game.set_visible(False)
    time.sleep(0.5)
    n_hidden = len(mouse.clicks)
    time.sleep(0.2)
    assert len(mouse.clicks) == n_hidden
    game.set_visible(True)
    time.sleep(0.3)
    eng.stop()
    s = eng.stats.snapshot()
    assert s.lost >= 1 and s.reacquired >= 1
    assert len(mouse.clicks) > n_hidden


def test_starts_inactive_and_toggle(cleanup):
    eng, game, cap, mouse = build()
    cleanup.append(eng)
    assert not eng.active and eng.status == Status.IDLE
    assert eng.toggle() is True and eng.active
    time.sleep(0.3)
    assert eng.toggle() is False and not eng.active
    n = len(mouse.clicks)
    assert n > 0
    time.sleep(0.3)
    assert len(mouse.clicks) == n         # stop really stops clicking


def test_no_competing_loops(cleanup):
    eng, game, cap, mouse = build()
    cleanup.append(eng)
    results = []
    threads = [threading.Thread(target=lambda: results.append(eng.start())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(True) == 1
    time.sleep(0.2)
    eng.stop()
    for _ in range(5):
        eng.start()
        eng.stop()
    assert eng.loops_started == 1
    assert sum(1 for t in threading.enumerate() if t.name == "automation") == 1


def test_shutdown_stops_thread_and_releases_capture(cleanup):
    eng, game, cap, mouse = build()
    eng.start()
    time.sleep(0.2)
    assert eng.shutdown(timeout=3)
    assert not any(t.name == "automation" and t.is_alive() for t in threading.enumerate())
    assert cap.closed
    n = len(mouse.clicks)
    time.sleep(0.2)
    assert len(mouse.clicks) == n
    assert eng.start() is False           # no restart after shutdown


def test_mouse_error_stops_automation(cleanup):
    class BadMouse:
        name = "bad"

        def click(self, x, y):
            raise MouseError("blocked")

    eng, game, cap, _ = build(mouse=BadMouse())
    cleanup.append(eng)
    eng.start()
    time.sleep(0.3)
    assert not eng.active
    assert eng.status == Status.ERROR
    assert "blocked" in eng.message


def test_capture_errors_lead_to_error_state(cleanup):
    class BrokenCapture:
        name = "broken"
        closed = False

        def grab(self, rect):
            raise CaptureError("no screen")

        def monitors(self):
            return MONS

        def close(self):
            self.closed = True

    eng, game, _, mouse = build(capture=BrokenCapture())
    cleanup.append(eng)
    eng.start()
    time.sleep(1.0)
    assert not eng.active and eng.status == Status.ERROR
    assert len(mouse.clicks) == 0


def test_region_unavailable_pauses_without_clicks(cleanup):
    eng, game, cap, mouse = build(area="monitor:5")
    cleanup.append(eng)
    eng.start()
    time.sleep(0.3)
    assert eng.status == Status.PAUSED
    assert len(mouse.clicks) == 0
    eng.stop()


def test_exclusion_rect_blocks_clicks(cleanup):
    eng, game, cap, mouse = build()
    cleanup.append(eng)
    eng.set_exclusions([MONS[0]])          # our window covers the whole game
    eng.start()
    time.sleep(0.4)
    eng.stop()
    assert len(mouse.clicks) == 0


def test_dry_run_mouse_factory_used():
    cfg = AppConfig(capture_area="monitor:0", dry_run=True)
    game = SimulatedGame(1280, 720, seed=1)
    cap = SimCapture(game, ORIGIN, monitors=MONS)
    created = []

    def factory(dry):
        created.append(dry)
        return DryRunMouse()

    eng = AutomationEngine(cfg, capture_factory=lambda: cap, mouse_factory=factory,
                           region_provider=RegionProvider("monitor:0", lambda: MONS))
    try:
        eng.start()
        time.sleep(0.3)
        eng.stop()
        assert created == [True]
        assert game.hits == 0                       # the "game" never received a click
        assert eng.stats.snapshot().clicks > 0      # but the decisions were made
    finally:
        eng.shutdown()


def test_calibration_request_processed_when_idle(cleanup):
    game = SimulatedGame(1280, 720, seed=2)
    eng, game, cap, mouse = build(game=game)
    cleanup.append(eng)
    cx, cy = game.center
    eng._cursor_fn = lambda: (int(cx) + ORIGIN[0], int(cy) + ORIGIN[1])
    got = []
    eng.on_calibration = got.append
    eng.request_calibration()
    for _ in range(50):
        if got:
            break
        time.sleep(0.05)
    assert got and got[0].ok, got and got[0].message
    assert eng.cfg.detector.expected_area > 0
    assert not eng.active and len(mouse.clicks) == 0


def test_debug_frame_published_only_when_enabled(cleanup):
    eng, game, cap, mouse = build()
    cleanup.append(eng)
    run_for(eng, 0.3)
    assert eng.debug_frame is None
    eng.cfg.debug_view = True
    run_for(eng, 0.4)
    df = eng.debug_frame
    assert df is not None and df.frame.ndim == 3
    from wormclicker.debug_view import annotate

    img, lines = annotate(df)
    assert img.shape[1] <= 720 and lines
