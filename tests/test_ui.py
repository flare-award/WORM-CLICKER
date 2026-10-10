"""UI tests: pure formatting everywhere, real Tk smoke test where a display exists."""

import pytest

tk = pytest.importorskip("tkinter")

from wormclicker.stats import StatsSnapshot  # noqa: E402
from wormclicker.ui import format_stats  # noqa: E402


def test_format_stats_hits_are_marked_approximate():
    s = StatsSnapshot(clicks=10, jumps=9, cps=12.34, session_s=75)
    out = format_stats(s)
    assert out["clicks"] == "10"
    assert out["jumps"] == "9 (90%)"
    assert out["cps"] == "12.3"
    assert out["session"] == "1:15"
    assert format_stats(StatsSnapshot())["jumps"] == "0 (—)"


def test_ui_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("WORMCLICKER_CONFIG", str(tmp_path / "config.json"))
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display: {exc}")
    from tests.test_engine import build
    from wormclicker.hotkeys import HotkeyManager
    from wormclicker.ui import App

    eng, game, cap, mouse = build()
    app = App(root, eng, HotkeyManager(lambda a: None), eng.cfg)
    try:
        for _ in range(5):
            root.update()
        app._toggle()
        root.after(400, lambda: None)
        for _ in range(20):
            root.update()
        assert eng.active
        app.handle_action("stop")
        assert not eng.active
        app.debug_var.set(True)
        app._debug_changed()
        root.update()
        app._apply()
        assert (tmp_path / "config.json").exists()
    finally:
        app.close()
    assert not eng.active
