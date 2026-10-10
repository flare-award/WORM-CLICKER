"""UI tests: pure formatting everywhere, real Tk smoke test where a display exists."""

import time

import pytest

from wormclicker.stats import StatsSnapshot
from wormclicker.formatting import format_stats, interval_text, interval_to_pos, pos_to_interval


def test_format_stats_hits_are_marked_approximate():
    s = StatsSnapshot(clicks=10, jumps=9, cps=12.34, session_s=75)
    out = format_stats(s)
    assert out["clicks"] == "10"
    assert out["jumps"] == "9 · 90%"
    assert out["cps"] == "12"
    assert out["session"] == "1:15"
    assert format_stats(StatsSnapshot())["jumps"] == "0"
    assert "сессия 1:15" in out["detail"]


def test_interval_slider_mapping():
    assert pos_to_interval(0.0) == 0.0 and interval_to_pos(0) == 0.0
    for ms in (1, 5, 10, 50, 100, 250, 1000):
        assert abs(pos_to_interval(interval_to_pos(ms)) - ms) <= max(1, ms * 0.03)
    assert interval_text(0)[0] == "Максимум"
    assert interval_text(50) == ("50 мс", "≈ 20 кликов в секунду")


def test_ui_smoke(tmp_path, monkeypatch):
    tk = pytest.importorskip("tkinter")
    monkeypatch.setenv("WORMCLICKER_CONFIG", str(tmp_path / "config.json"))
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display: {exc}")
    from tests.test_engine import build
    from wormclicker.hotkeys import HotkeyManager
    from wormclicker.ui import App

    eng, game, cap, mouse = build(preview=True)
    app = App(root, eng, HotkeyManager(lambda a: None), eng.cfg)

    def pump(seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            root.update()
            time.sleep(0.005)

    try:
        pump(0.6)
        assert app._df is not None and app._img_geom[2] > 0          # preview shows frames
        for tab in ("target", "capture", "keys", "clicks"):
            app._show_tab(tab)
            pump(0.05)
        app._set_interval(50)
        pump(0.1)
        assert eng.cfg.click_interval_ms == 50
        app._toggle()
        pump(0.2)
        game.kick()
        pump(0.6)
        assert eng.active and len(mouse.clicks) > 0
        app.handle_action("stop")
        assert not eng.active
        app.debug_var.set(False)
        app._debug_changed()
        pump(0.1)
        app._apply()
        assert (tmp_path / "config.json").exists()
    finally:
        app.close()
    assert not eng.active
