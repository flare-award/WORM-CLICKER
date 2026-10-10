"""Platform-specific smoke tests (run on Windows CI, skipped elsewhere)."""

import ctypes
import sys
import time

import numpy as np
import pytest

win = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")


@win
def test_input_struct_size():
    from wormclicker import mouse

    assert ctypes.sizeof(mouse.INPUT) == (40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28)
    m = mouse.WinMouse()
    assert m._size == ctypes.sizeof(mouse.INPUT)


@win
def test_winapi_calls_do_not_crash():
    from wormclicker import winapi

    winapi.set_dpi_awareness()
    winapi.foreground_window()
    assert winapi.current_pid() > 0
    winapi.cursor_pos()
    assert isinstance(winapi.monitors_win32(), list)


@win
def test_hotkey_manager_register_and_stop():
    from wormclicker.hotkeys import Win32HotkeyManager

    got = []
    mgr = Win32HotkeyManager(got.append)
    mgr.start()
    errors = mgr.set_bindings({"toggle": "Ctrl+Alt+Shift+F12", "stop": "F99"})
    assert "stop" in errors                       # invalid key reported
    assert "toggle" not in errors or "занята" in errors["toggle"]
    mgr.stop()
    assert mgr._thread is None


def test_mss_capture_if_display_available():
    from wormclicker.capture import CaptureError, MssCapture
    from wormclicker.geometry import Rect

    try:
        cap = MssCapture()
    except CaptureError as exc:
        pytest.skip(f"no display: {exc}")
    try:
        mons = cap.monitors()
        if not mons:
            pytest.skip("no monitors")
        m = mons[0]
        f = cap.grab(Rect(m.left, m.top, 64, 48))
        assert f.shape == (48, 64, 4) and f.dtype == np.uint8
    finally:
        cap.close()


@win
def test_dxcam_capture_if_available():
    from wormclicker.capture import DxcamCapture
    from wormclicker.geometry import Rect

    try:
        cap = DxcamCapture()
    except Exception as exc:
        pytest.skip(f"dxcam unavailable: {exc}")
    try:
        m = cap.monitors()[0]
        f = None
        for _ in range(10):
            f = cap.grab(Rect(m.left, m.top, 64, 48))
            if f is not None:
                break
            time.sleep(0.02)
        assert f is None or f.shape[:2] == (48, 64)
    finally:
        cap.close()
