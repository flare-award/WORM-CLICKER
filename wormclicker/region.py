"""Which part of the screen is scanned.

Modes (``AppConfig.capture_area``):
  * ``window``    - client area of the browser window that was in the
                    foreground when the automation was started (F6 pressed
                    in the browser). Position/size are re-read every 250 ms,
                    so moving or resizing the window is handled. When the
                    window loses focus, is minimised or closed - pause.
  * ``monitor:N`` - the whole monitor N (0-based).
  * ``all``       - the whole virtual desktop (all monitors).
"""

from __future__ import annotations

import time
from types import ModuleType
from typing import Callable, Sequence

from . import winapi as _winapi
from .geometry import Rect, bounding_rect, clip_to_monitors


class RegionProvider:
    WINDOW_REFRESH_S = 0.25
    MONITOR_REFRESH_S = 2.0

    def __init__(self, area: str, monitors_fn: Callable[[], Sequence[Rect]],
                 require_focus: bool = True, win: ModuleType | object = _winapi,
                 clock: Callable[[], float] = time.perf_counter):
        self.area = area
        self.require_focus = require_focus
        self._monitors_fn = monitors_fn
        self._win = win
        self._clock = clock
        self._monitors: list[Rect] = []
        self._mon_t = -1e9
        self.reset()

    def reset(self) -> None:
        self.hwnd = 0
        self.title = ""
        self._rect: Rect | None = None
        self._rect_t = -1e9

    def monitors(self, now: float | None = None) -> list[Rect]:
        now = self._clock() if now is None else now
        if not self._monitors or now - self._mon_t > self.MONITOR_REFRESH_S:
            try:
                mons = list(self._monitors_fn())
            except Exception:
                mons = []
            if mons:
                self._monitors = mons
            self._mon_t = now
        return self._monitors

    def get(self, now: float | None = None) -> tuple[Rect | None, str]:
        now = self._clock() if now is None else now
        mons = self.monitors(now)
        if not mons:
            return None, "мониторы не найдены"
        area = self.area
        if area == "all":
            return bounding_rect(mons), ""
        if area.startswith("monitor:"):
            try:
                idx = int(area.split(":", 1)[1])
            except ValueError:
                idx = 0
            if not 0 <= idx < len(mons):
                return None, f"монитор {idx + 1} не найден"
            return mons[idx], ""
        return self._window_rect(now, mons)

    def _window_rect(self, now: float, mons: list[Rect]) -> tuple[Rect | None, str]:
        w = self._win
        if not getattr(w, "IS_WINDOWS", False):
            return None, "режим «окно браузера» доступен только в Windows — выберите монитор"
        fg = w.root_window(w.foreground_window())
        if not self.hwnd:
            if not fg or w.window_pid(fg) == w.current_pid():
                return None, "переключитесь в окно браузера с игрой"
            self.hwnd = fg
            self.title = w.window_title(fg)
            self._rect_t = -1e9
        ok, why = w.is_window_usable(self.hwnd)
        if not ok:
            return None, why
        if self.require_focus and fg != self.hwnd:
            return None, "окно браузера не в фокусе — пауза"
        if now - self._rect_t > self.WINDOW_REFRESH_S or self._rect is None:
            rc = w.client_rect_on_screen(self.hwnd)
            self._rect = clip_to_monitors(rc, mons) if rc is not None else None
            self._rect_t = now
        if self._rect is None or self._rect.width < 16 or self._rect.height < 16:
            return None, "окно браузера вне экрана или слишком маленькое"
        return self._rect, ""
