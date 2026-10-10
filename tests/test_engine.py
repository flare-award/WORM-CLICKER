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


def build(game=None, lag=0, cfg=None, mouse=None, capture=None, area="monitor:0", preview=False, **cfg_kw):
    game = game or SimulatedGame(1280, 720, seed=7)
    cap = capture or SimCapture(game, ORIGIN, lag_frames=lag, monitors=MONS)
    sim_mouse = mouse or SimMouse(game, ORIGIN)
    cfg = cfg or AppConfig(capture_area=area, debug_view=preview, **cfg_kw)
    eng = AutomationEngine(cfg, capture_factory=lambda: cap, mouse_factory=lambda dry: sim_mouse,
                           region_provider=RegionProvider(area, lambda: MONS), cursor_fn=lambda: None)
    return eng, game, cap, sim_mouse


def wait_until(cond, timeout=2.0):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def start_and_kick(eng, game, warm=0.15):
    """Start, let the program memorise the static picture, then the user clicks the worm once."""
    assert eng.start()
    time.sleep(warm)
    assert game.kick()


@pytest.fixture
def cleanup():
    engines = []
    yield engines
    for e in engines:
        e.shutdown()


def test_waits_for_motion_then_hunts_on_negative_coordinate_monitor(cleanup):
    eng, game, cap, mouse = build()
    cleanup.append(eng)
    assert eng.start()
    time.sleep(0.5)
    assert len(mouse.clicks) == 0                    # the worm has not moved yet: no target
    assert eng.status == Status.SEARCHING
    game.kick()
    time.sleep(1.5)
    eng.stop()
    assert game.hits >= 30
    assert all(x < 0 for x, _, _ in mouse.clicks)    # clicks really on the left monitor
    s = eng.stats.snapshot()
    assert s.clicks == len(mouse.clicks)
    assert s.jumps >= game.hits - 2


def test_static_decoy_worm_is_never_clicked(cleanup):
    game = SimulatedGame(1280, 720, seed=11, decoy=True)
    eng, game, cap, mouse = build(game=game)
    cleanup.append(eng)
    start_and_kick(eng, game)
    time.sleep(1.5)
    eng.stop()
    assert game.hits >= 30
    assert game.decoy_clicks == 0


@pytest.mark.parametrize("latency,lag", [(0.03, 0), (0.03, 2)])
def test_clicks_continuously_without_waiting_for_jump(cleanup, latency, lag):
    game = SimulatedGame(1280, 720, seed=3, render_latency_s=latency)
    eng, game, cap, mouse = build(game=game, lag=lag)
    cleanup.append(eng)
    start_and_kick(eng, game)
    time.sleep(1.5)
    eng.stop()
    assert game.hits >= 10
    # the old picture is still on screen after a hit - the program keeps clicking it
    assert len(mouse.clicks) > 2 * game.hits
    assert game.decoy_clicks == 0


@pytest.mark.parametrize("interval,expected", [(50.0, 20.0), (20.0, 50.0)])
def test_click_interval_sets_the_rate(cleanup, interval, expected):
    eng, game, cap, mouse = build(click_interval_ms=interval)
    cleanup.append(eng)
    start_and_kick(eng, game)
    time.sleep(0.3)
    n0 = len(mouse.clicks)
    time.sleep(2.0)
    rate = (len(mouse.clicks) - n0) / 2.0
    eng.stop()
    assert expected * 0.8 <= rate <= expected * 1.1, rate


def test_interval_change_applies_while_running(cleanup):
    eng, game, cap, mouse = build(click_interval_ms=100.0)
    cleanup.append(eng)
    start_and_kick(eng, game)
    time.sleep(1.0)
    slow = len(mouse.clicks)
    cfg = AppConfig(capture_area="monitor:0", debug_view=False, click_interval_ms=0.0)
    eng.apply_config(cfg)
    n0 = len(mouse.clicks)
    time.sleep(1.0)
    eng.stop()
    assert slow <= 12
    assert len(mouse.clicks) - n0 > 3 * slow


def test_no_clicks_while_worm_hidden(cleanup):
    eng, game, cap, mouse = build()
    cleanup.append(eng)
    start_and_kick(eng, game)
    time.sleep(0.3)
    game.set_visible(False)
    time.sleep(0.1)
    n = len(mouse.clicks)
    hidden0 = game.clicks_while_hidden
    time.sleep(0.5)
    assert len(mouse.clicks) == n
    assert game.clicks_while_hidden == hidden0
    eng.stop()


def test_temporary_disappearance_counts_lost_and_reacquired(cleanup):
    eng, game, cap, mouse = build()
    cleanup.append(eng)
    start_and_kick(eng, game)
    time.sleep(0.3)
    game.set_visible(False)
    time.sleep(0.5)
    n_hidden = len(mouse.clicks)
    game.set_visible(True)            # reappears (a new picture - pixels changed)
    time.sleep(0.4)
    eng.stop()
    s = eng.stats.snapshot()
    assert s.lost >= 1 and s.reacquired >= 1
    assert len(mouse.clicks) > n_hidden


def test_starts_inactive_and_toggle(cleanup):
    eng, game, cap, mouse = build()
    cleanup.append(eng)
    assert not eng.active and eng.status == Status.IDLE
    assert eng.toggle() is True and eng.active
    time.sleep(0.15)
    game.kick()
    time.sleep(0.3)
    assert eng.toggle() is False and not eng.active
    n = len(mouse.clicks)
    assert n > 0
    time.sleep(0.3)
    assert len(mouse.clicks) == n         # stop really stops clicking


def test_target_learned_in_preview_is_clicked_right_after_start(cleanup):
    eng, game, cap, mouse = build(preview=True)
    cleanup.append(eng)
    time.sleep(0.2)                       # watch mode: preview running, no clicks
    game.kick()
    assert wait_until(lambda: eng.finder.model is not None)
    assert len(mouse.clicks) == 0
    assert eng.start()
    assert wait_until(lambda: game.hits >= 5)
    eng.stop()


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
    eng, game, cap, mouse = build(preview=True)
    start_and_kick(eng, game)
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
    start_and_kick(eng, game)
    assert wait_until(lambda: not eng.active)
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
    assert wait_until(lambda: not eng.active, 3.0)
    assert eng.status == Status.ERROR
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
    start_and_kick(eng, game)
    time.sleep(0.4)
    eng.stop()
    assert len(mouse.clicks) == 0


def test_dry_run_mouse_factory_used():
    cfg = AppConfig(capture_area="monitor:0", dry_run=True, debug_view=False)
    game = SimulatedGame(1280, 720, seed=1)
    cap = SimCapture(game, ORIGIN, monitors=MONS)
    created = []

    def factory(dry):
        created.append(dry)
        return DryRunMouse()

    eng = AutomationEngine(cfg, capture_factory=lambda: cap, mouse_factory=factory,
                           region_provider=RegionProvider("monitor:0", lambda: MONS))
    try:
        start_and_kick(eng, game)
        time.sleep(0.3)
        eng.stop()
        assert created == [True]
        assert game.hits == 0                       # the "game" never received a click
        assert eng.stats.snapshot().clicks > 0      # but the decisions were made
        assert eng.finder.model is not None         # test mode never drops the target
    finally:
        eng.shutdown()


def test_unresponsive_object_is_dropped(cleanup):
    class Deaf:                       # clicks never reach the game -> the target never jumps
        name = "deaf"
        clicks = []

        def click(self, x, y):
            self.clicks.append((x, y, time.perf_counter()))

    eng, game, cap, mouse = build(mouse=Deaf())
    eng.cfg.target.unlock_after_ms = 300
    cleanup.append(eng)
    start_and_kick(eng, game)
    assert wait_until(lambda: eng.stats.snapshot().ignored >= 1, 3.0)
    n = len(mouse.clicks)
    time.sleep(0.3)
    assert len(mouse.clicks) == n               # ignored: no more clicks on it
    assert eng.finder.model is None             # wrong object forgotten (it never jumped)
    eng.stop()


def test_sample_from_preview_selection_locks_without_motion(cleanup):
    eng, game, cap, mouse = build(preview=True)
    cleanup.append(eng)
    got = []
    eng.on_sample = got.append
    assert wait_until(lambda: eng.debug_frame is not None and eng.debug_frame.image is not None)
    cx, cy = game.center
    box = (int(cx) - 60 + ORIGIN[0], int(cy) - 40 + ORIGIN[1], 120, 80)
    eng.request_sample(box)
    assert wait_until(lambda: bool(got))
    assert got[0].ok, got[0].message
    assert eng.finder.model.source == "sample" and eng.cfg.target.mode == "sample"
    assert eng.start()                          # no kick needed in the sample mode
    assert wait_until(lambda: game.hits >= 10)
    eng.stop()


def test_cursor_capture_processed_when_idle(cleanup):
    game = SimulatedGame(1280, 720, seed=2)
    eng, game, cap, mouse = build(game=game)
    cleanup.append(eng)
    ys, xs = game.mask.nonzero()
    j = len(xs) // 2
    eng._cursor_fn = lambda: (int(xs[j]) + ORIGIN[0], int(ys[j]) + ORIGIN[1])
    got = []
    eng.on_sample = got.append
    eng.request_cursor_capture()
    assert wait_until(lambda: bool(got))
    assert got[0].ok, got[0].message
    assert eng.cfg.detector.expected_area > 0
    assert not eng.active and len(mouse.clicks) == 0


def test_preview_published_at_high_rate_and_only_when_enabled(cleanup):
    eng, game, cap, mouse = build()
    cleanup.append(eng)
    time.sleep(0.3)
    assert eng.debug_frame is None and eng.loops_started == 0      # nothing runs when idle
    eng.cfg.preview_fps = 60
    eng.set_preview(True)
    time.sleep(0.3)
    n0 = eng.preview_frames
    time.sleep(1.0)
    fps = eng.preview_frames - n0
    assert fps >= 40, fps
    df = eng.debug_frame
    assert df is not None and df.image is not None and df.image.ndim == 3
    assert df.image.shape[1] <= eng.preview_box[0]
    from wormclicker.debug_view import render, to_ppm

    game.kick()
    assert wait_until(lambda: eng.debug_frame.target is not None)
    rgb = render(eng.debug_frame)
    assert rgb.shape[2] == 3
    assert to_ppm(rgb).startswith(b"P6 ")
    eng.set_preview(False)
