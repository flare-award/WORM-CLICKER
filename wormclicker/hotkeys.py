"""Global hotkeys.

Windows implementation uses ``RegisterHotKey`` on a dedicated thread with its
own message loop:
  * works while the browser has focus and does not require admin rights;
  * the key press is consumed, so F6 does not also move the browser focus to
    the address bar;
  * ``MOD_NOREPEAT`` + a software debounce make holding the key toggle only
    once.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger(__name__)

MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, MOD_NOREPEAT = 0x1, 0x2, 0x4, 0x8, 0x4000

_MOD_NAMES = {"CTRL": MOD_CONTROL, "CONTROL": MOD_CONTROL, "ALT": MOD_ALT, "SHIFT": MOD_SHIFT,
              "WIN": MOD_WIN}

VK_CODES: dict[str, int] = {f"F{i}": 0x6F + i for i in range(1, 25)}
VK_CODES.update({chr(c): c for c in range(ord("A"), ord("Z") + 1)})
VK_CODES.update({str(d): 0x30 + d for d in range(10)})
VK_CODES.update({f"NUM{d}": 0x60 + d for d in range(10)})
VK_CODES.update({
    "PAUSE": 0x13, "SCROLLLOCK": 0x91, "INSERT": 0x2D, "DELETE": 0x2E, "HOME": 0x24, "END": 0x23,
    "PAGEUP": 0x21, "PAGEDOWN": 0x22, "SPACE": 0x20, "ESCAPE": 0x1B, "TAB": 0x09,
    "UP": 0x26, "DOWN": 0x28, "LEFT": 0x25, "RIGHT": 0x27,
})

# Tk keysym -> our key name
_TK_KEYSYMS = {"Prior": "PAGEUP", "Next": "PAGEDOWN", "Scroll_Lock": "SCROLLLOCK", "Pause": "PAUSE",
               "Insert": "INSERT", "Delete": "DELETE", "Home": "HOME", "End": "END",
               "space": "SPACE", "Escape": "ESCAPE", "Tab": "TAB", "Up": "UP", "Down": "DOWN",
               "Left": "LEFT", "Right": "RIGHT"}


class HotkeyError(ValueError):
    pass


@dataclass(frozen=True)
class Hotkey:
    mods: int
    vk: int
    text: str


def parse_hotkey(text: str) -> Hotkey:
    """'F6', 'ctrl+shift+F6', 'Alt+Q' -> Hotkey."""
    if not text or not text.strip():
        raise HotkeyError("пустая горячая клавиша")
    parts = [p.strip().upper() for p in text.replace("-", "+").split("+") if p.strip()]
    mods = 0
    key = None
    for p in parts:
        if p in _MOD_NAMES:
            mods |= _MOD_NAMES[p]
        elif key is None and p in VK_CODES:
            key = p
        else:
            raise HotkeyError(f"неизвестная клавиша: {p}")
    if key is None:
        raise HotkeyError("не указана основная клавиша")
    names = [n for n, m in (("Ctrl", MOD_CONTROL), ("Alt", MOD_ALT), ("Shift", MOD_SHIFT), ("Win", MOD_WIN))
             if mods & m]
    return Hotkey(mods, VK_CODES[key], "+".join(names + [key]))


def hotkey_from_tk(keysym: str, state: int) -> str | None:
    """Convert a Tk key event into hotkey text, e.g. ('F6', 0x4) -> 'Ctrl+F6'."""
    if keysym in ("Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R", "Win_L", "Win_R"):
        return None
    name = _TK_KEYSYMS.get(keysym)
    if name is None:
        k = keysym.upper()
        if k.startswith("KP_") and k[3:].isdigit():
            name = "NUM" + k[3:]
        elif k in VK_CODES:
            name = k
    if name is None:
        return None
    mods = []
    if state & 0x4:
        mods.append("Ctrl")
    alt_mask = 0x20000 if sys.platform == "win32" else 0x8
    if state & alt_mask:
        mods.append("Alt")
    if state & 0x1:
        mods.append("Shift")
    return "+".join(mods + [name])


class Debouncer:
    """Accept an action at most once per ``interval`` seconds."""

    def __init__(self, interval: float = 0.3, clock: Callable[[], float] = time.monotonic):
        self.interval = interval
        self._clock = clock
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def accept(self, action: str) -> bool:
        now = self._clock()
        with self._lock:
            last = self._last.get(action)
            if last is not None and now - last < self.interval:
                return False
            self._last[action] = now
            return True


class HotkeyManager:
    """Base class: no global hotkeys (non-Windows). Subclasses register keys."""

    available = False

    def __init__(self, on_action: Callable[[str], None], debounce_s: float = 0.3):
        self._on_action = on_action
        self._debounce = Debouncer(debounce_s)

    def dispatch(self, action: str) -> None:
        if self._debounce.accept(action):
            try:
                self._on_action(action)
            except Exception:  # pragma: no cover - never kill the hotkey thread
                log.exception("hotkey handler failed")

    def set_bindings(self, bindings: dict[str, str]) -> dict[str, str]:
        """Register {action: 'F6'}; returns {action: error message} for failures."""
        errors = {}
        for action, text in bindings.items():
            try:
                parse_hotkey(text)
            except HotkeyError as exc:
                errors[action] = str(exc)
        if not errors:
            errors = {a: "глобальные горячие клавиши доступны только в Windows" for a in bindings}
        return errors

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


if sys.platform == "win32":
    from ctypes import wintypes

    _u32 = ctypes.WinDLL("user32", use_last_error=True)
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _u32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
    _u32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
    _u32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
    _u32.PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT,
                                  wintypes.UINT, wintypes.UINT]
    _u32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]

    WM_HOTKEY = 0x0312
    WM_QUIT = 0x0012
    WM_APP_REBIND = 0x8000 + 1


class Win32HotkeyManager(HotkeyManager):
    available = True

    def __init__(self, on_action: Callable[[str], None], debounce_s: float = 0.3):
        super().__init__(on_action, debounce_s)
        self._thread: threading.Thread | None = None
        self._tid = 0
        self._ready = threading.Event()
        self._rebind_done = threading.Event()
        self._pending: dict[str, str] = {}
        self._errors: dict[str, str] = {}
        self._ids: dict[int, str] = {}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._ready.clear()
        self._thread = threading.Thread(target=self._run, name="hotkeys", daemon=True)
        self._thread.start()
        self._ready.wait(2.0)

    def stop(self) -> None:
        if self._thread and self._thread.is_alive() and self._tid:
            _u32.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)
            self._thread.join(2.0)
        self._thread = None

    def set_bindings(self, bindings: dict[str, str]) -> dict[str, str]:
        self.start()
        self._pending = dict(bindings)
        self._rebind_done.clear()
        if not _u32.PostThreadMessageW(self._tid, WM_APP_REBIND, 0, 0):
            return {a: "поток горячих клавиш не запущен" for a in bindings}
        self._rebind_done.wait(2.0)
        return dict(self._errors)

    # --------------------------------------------------------------- thread
    def _run(self) -> None:
        msg = wintypes.MSG()
        _u32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0)  # create the message queue
        self._tid = _k32.GetCurrentThreadId()
        self._ready.set()
        try:
            while True:
                r = _u32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if r == 0 or r == -1:
                    break
                if msg.message == WM_HOTKEY:
                    action = self._ids.get(int(msg.wParam))
                    if action:
                        self.dispatch(action)
                elif msg.message == WM_APP_REBIND:
                    self._rebind()
                    self._rebind_done.set()
        finally:
            self._unregister_all()

    def _unregister_all(self) -> None:
        for hid in list(self._ids):
            _u32.UnregisterHotKey(None, hid)
        self._ids.clear()

    def _rebind(self) -> None:
        self._unregister_all()
        errors: dict[str, str] = {}
        used: set[tuple[int, int]] = set()
        for n, (action, text) in enumerate(self._pending.items(), start=1):
            if not text:
                continue
            try:
                hk = parse_hotkey(text)
            except HotkeyError as exc:
                errors[action] = str(exc)
                continue
            if (hk.mods, hk.vk) in used:
                errors[action] = f"{hk.text} уже назначена другому действию"
                continue
            if _u32.RegisterHotKey(None, n, hk.mods | MOD_NOREPEAT, hk.vk):
                self._ids[n] = action
                used.add((hk.mods, hk.vk))
            else:
                errors[action] = f"{hk.text} занята другой программой (код {ctypes.get_last_error()})"
        self._errors = errors


def create_hotkey_manager(on_action: Callable[[str], None]) -> HotkeyManager:
    if sys.platform == "win32":
        return Win32HotkeyManager(on_action)
    return HotkeyManager(on_action)
