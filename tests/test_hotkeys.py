import time

import pytest

from wormclicker.hotkeys import (MOD_CONTROL, MOD_SHIFT, Debouncer, HotkeyError, HotkeyManager,
                                 hotkey_from_tk, parse_hotkey)


def test_parse_simple():
    hk = parse_hotkey("F6")
    assert hk.vk == 0x75 and hk.mods == 0 and hk.text == "F6"
    assert parse_hotkey("f7").vk == 0x76


def test_parse_modifiers():
    hk = parse_hotkey("ctrl+shift+F6")
    assert hk.mods == MOD_CONTROL | MOD_SHIFT and hk.text == "Ctrl+Shift+F6"
    assert parse_hotkey("Alt+Q").vk == ord("Q")


@pytest.mark.parametrize("bad", ["", "Ctrl", "F99", "F6+F7", "Hyper+F6"])
def test_parse_errors(bad):
    with pytest.raises(HotkeyError):
        parse_hotkey(bad)


def test_tk_conversion():
    assert hotkey_from_tk("F6", 0) == "F6"
    assert hotkey_from_tk("F9", 0x4) == "Ctrl+F9"
    assert hotkey_from_tk("Prior", 0) == "PAGEUP"
    assert hotkey_from_tk("Control_L", 0x4) is None
    assert hotkey_from_tk("KP_5", 0) == "NUM5"
    assert hotkey_from_tk("q", 0x1) == "Shift+Q"


def test_debouncer():
    t = [0.0]
    d = Debouncer(0.3, clock=lambda: t[0])
    assert d.accept("toggle")
    t[0] = 0.1
    assert not d.accept("toggle")          # auto-repeat while holding F6
    assert d.accept("stop")                 # other actions independent
    t[0] = 0.45
    assert d.accept("toggle")


def test_held_key_toggles_once_engine():
    """Holding F6 produces a burst of events -> exactly one toggle."""
    from tests.test_engine import build

    eng, game, cap, mouse = build()
    try:
        mgr = HotkeyManager(lambda a: eng.toggle() if a == "toggle" else eng.emergency_stop(), debounce_s=0.3)
        for _ in range(15):                 # key auto-repeat ~30 ms
            mgr.dispatch("toggle")
            time.sleep(0.005)
        assert eng.active
        time.sleep(0.35)
        mgr.dispatch("toggle")              # second, deliberate press
        assert not eng.active
        time.sleep(0.35)
        mgr.dispatch("toggle")
        assert eng.active
        mgr.dispatch("stop")                # F7 emergency stop
        assert not eng.active
    finally:
        eng.shutdown()


def test_base_manager_reports_unavailable():
    mgr = HotkeyManager(lambda a: None)
    errs = mgr.set_bindings({"toggle": "F6"})
    assert "toggle" in errs
    errs = mgr.set_bindings({"toggle": "F99"})
    assert "неизвестная" in errs["toggle"]
