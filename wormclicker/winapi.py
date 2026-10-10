"""Thin ctypes wrappers around the Win32 functions the app needs.

Everything degrades gracefully on non-Windows systems (functions return
neutral values), which keeps the rest of the code importable and testable
on Linux CI.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys

from .geometry import Rect

log = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
    user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    user32.GetAncestor.restype = wintypes.HWND
    try:
        user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
    except AttributeError:  # pragma: no cover - very old Windows
        pass


def set_dpi_awareness() -> str:
    """Make the process per-monitor DPI aware (physical pixel coordinates)."""
    if not IS_WINDOWS:
        return "n/a"
    try:
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4 (Windows 10 1703+)
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return "per-monitor-v2"
    except (AttributeError, OSError):
        pass
    try:
        if ctypes.windll.shcore.SetProcessDpiAwareness(2) == 0:
            return "per-monitor"
    except (AttributeError, OSError):
        pass
    try:
        user32.SetProcessDPIAware()
        return "system"
    except (AttributeError, OSError):
        return "unaware"


def current_pid() -> int:
    return os.getpid()


def foreground_window() -> int:
    if not IS_WINDOWS:
        return 0
    return int(user32.GetForegroundWindow() or 0)


def window_pid(hwnd: int) -> int:
    if not IS_WINDOWS or not hwnd:
        return 0
    pid = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value)


def root_window(hwnd: int) -> int:
    if not IS_WINDOWS or not hwnd:
        return hwnd
    return int(user32.GetAncestor(hwnd, 2) or hwnd)  # GA_ROOT


def is_window_usable(hwnd: int) -> tuple[bool, str]:
    if not IS_WINDOWS:
        return False, "окна доступны только в Windows"
    if not hwnd or not user32.IsWindow(hwnd):
        return False, "окно браузера закрыто"
    if user32.IsIconic(hwnd):
        return False, "окно браузера свернуто"
    if not user32.IsWindowVisible(hwnd):
        return False, "окно браузера скрыто"
    return True, ""


def client_rect_on_screen(hwnd: int) -> Rect | None:
    """Client area of a window in physical screen coordinates."""
    if not IS_WINDOWS or not hwnd:
        return None
    rc = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rc)):
        return None
    pt = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        return None
    return Rect(pt.x, pt.y, rc.right - rc.left, rc.bottom - rc.top)


def window_title(hwnd: int) -> str:
    if not IS_WINDOWS or not hwnd:
        return ""
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def cursor_pos() -> tuple[int, int] | None:
    if not IS_WINDOWS:
        return None
    pt = wintypes.POINT()
    if not user32.GetCursorPos(ctypes.byref(pt)):
        return None
    return pt.x, pt.y


def exclude_from_capture(hwnd: int) -> bool:
    """Hide our own windows from screen capture (Windows 10 2004+).

    Prevents the program from seeing the worm inside its own preview.
    ``WORMCLICKER_ALLOW_CAPTURE=1`` disables it (UI screenshots in CI).
    """
    if not IS_WINDOWS or not hwnd or os.environ.get("WORMCLICKER_ALLOW_CAPTURE") == "1":
        return False
    try:
        WDA_EXCLUDEFROMCAPTURE = 0x11
        return bool(user32.SetWindowDisplayAffinity(root_window(hwnd), WDA_EXCLUDEFROMCAPTURE))
    except (AttributeError, OSError):
        return False


def monitors_win32() -> list[Rect]:
    """Monitor rectangles via EnumDisplayMonitors (fallback when mss is absent)."""
    if not IS_WINDOWS:
        return []
    out: list[Rect] = []
    MONITORENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_int, wintypes.HMONITOR, wintypes.HDC,
                                         ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)

    def cb(_hm, _hdc, rc, _lp):
        r = rc.contents
        out.append(Rect.from_ltrb(r.left, r.top, r.right, r.bottom))
        return 1

    user32.EnumDisplayMonitors(None, None, MONITORENUMPROC(cb), 0)
    return out


def set_timer_resolution(enable: bool, ms: int = 1) -> bool:
    """1 ms system timer resolution while the program runs (smooth 60-120 fps
    preview and precise click intervals; the default Windows tick is 15.6 ms)."""
    if not IS_WINDOWS:
        return False
    try:
        winmm = ctypes.WinDLL("winmm")
        fn = winmm.timeBeginPeriod if enable else winmm.timeEndPeriod
        return fn(ms) == 0
    except (AttributeError, OSError):
        return False
