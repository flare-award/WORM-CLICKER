"""Mouse output.

``WinMouse`` positions the cursor with ``SetCursorPos`` (exact physical
pixel, works with negative multi-monitor coordinates under per-monitor DPI
awareness) and sends button down+up as one ``SendInput`` batch, so the two
events can not be interleaved with other input. The INPUT array is built
once and reused - no allocations per click.

``DryRunMouse`` records clicks without touching the system (test mode).
"""

from __future__ import annotations

import ctypes
import sys
import threading
import time
from typing import Callable, Protocol


class MouseError(RuntimeError):
    pass


class Mouse(Protocol):
    name: str

    def click(self, x: int, y: int) -> None:
        ...


class DryRunMouse:
    """Collects clicks instead of sending them. Optional ``on_click`` hook."""

    name = "dry-run"

    def __init__(self, on_click: Callable[[int, int], None] | None = None, keep: int = 10000):
        self.clicks: list[tuple[int, int, float]] = []
        self._on_click = on_click
        self._keep = keep
        self._lock = threading.Lock()

    def click(self, x: int, y: int) -> None:
        with self._lock:
            self.clicks.append((int(x), int(y), time.perf_counter()))
            if len(self.clicks) > self._keep:
                del self.clicks[: len(self.clicks) - self._keep]
        if self._on_click is not None:
            self._on_click(int(x), int(y))


if sys.platform == "win32":
    from ctypes import wintypes

    ULONG_PTR = ctypes.c_size_t

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]

    class _INPUTUNION(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]

    INPUT_MOUSE = 0
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
    _user32.SendInput.restype = wintypes.UINT
    _user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
    _user32.SetCursorPos.restype = wintypes.BOOL


class WinMouse:
    name = "SendInput"

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise MouseError("Реальные клики поддерживаются только в Windows")
        arr = (INPUT * 2)()
        arr[0].type = INPUT_MOUSE
        arr[0].u.mi.dwFlags = MOUSEEVENTF_LEFTDOWN
        arr[1].type = INPUT_MOUSE
        arr[1].u.mi.dwFlags = MOUSEEVENTF_LEFTUP
        self._inputs = arr
        self._size = ctypes.sizeof(INPUT)

    def click(self, x: int, y: int) -> None:
        if not _user32.SetCursorPos(int(x), int(y)):
            raise MouseError(f"SetCursorPos не выполнен (ошибка {ctypes.get_last_error()})")
        sent = _user32.SendInput(2, self._inputs, self._size)
        if sent != 2:
            err = ctypes.get_last_error()
            raise MouseError(
                "SendInput заблокирован (код %d). Возможно, окно игры запущено с правами "
                "администратора (UIPI) или активен защищённый рабочий стол." % err)


def create_mouse(dry_run: bool) -> Mouse:
    if dry_run or sys.platform != "win32":
        return DryRunMouse()
    return WinMouse()
