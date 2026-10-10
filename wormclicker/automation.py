"""Automation engine: the single worker thread running

    capture -> find the moving target -> click -> capture next frame ...

* While active, every frame with a visible target produces a click at the
  target's current position (no waiting for the jump). The pause between
  clicks is ``click_interval_ms`` (0 = as fast as frames come).
* While inactive but the preview is shown, the same loop runs without
  clicking ("watch" mode): the user sees what the program sees and the
  target can be locked before starting.

Guarantees
  * exactly one worker thread per engine; ``start()`` while running is a
    no-op, so competing click loops are impossible;
  * after ``stop()`` returns no further click is sent (the active flag is
    checked under the same lock that wraps the click call);
  * the worker sleeps on an Event when there is nothing to do;
  * capture objects live in the worker thread only;
  * the UI only reads snapshots (stats, status, latest preview frame).
"""

from __future__ import annotations

import enum
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .capture import Capture, CaptureError, create_capture, list_monitors
from .config import AppConfig
from .geometry import Rect, frame_to_screen
from .mouse import Mouse, MouseError, create_mouse
from .region import RegionProvider
from .stats import Stats
from .target import Blob, FindResult, SampleResult, TargetFinder, TargetState
from . import winapi

log = logging.getLogger(__name__)


class Status(str, enum.Enum):
    IDLE = "Ожидание"
    PAUSED = "Пауза"
    SEARCHING = "Поиск цели"
    ACTIVE = "Кликаю"
    ERROR = "Ошибка"


@dataclass
class DebugFrame:
    image: np.ndarray | None        # downscaled BGR(A) copy of the frame, None = no picture
    scale: float                    # preview px per frame px
    rect: Rect | None               # captured screen region
    target: Blob | None
    state: str
    message: str
    ms: float                       # analysis time of this frame
    seq: int
    t: float
    ignored: tuple = ()


class AutomationEngine:
    MAX_CAPTURE_ERRORS = 20

    def __init__(
        self,
        cfg: AppConfig,
        capture_factory: Callable[[], Capture] | None = None,
        mouse_factory: Callable[[bool], Mouse] | None = None,
        region_provider: RegionProvider | None = None,
        cursor_fn: Callable[[], tuple[int, int] | None] = winapi.cursor_pos,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.cfg = cfg
        self.clock = clock
        self.stats = Stats(clock)
        self.finder = TargetFinder(cfg.detector, cfg.target)
        self._capture_factory = capture_factory
        self._mouse_factory = mouse_factory or create_mouse
        self.region = region_provider or RegionProvider(cfg.capture_area, list_monitors, cfg.require_focus)
        self._cursor_fn = cursor_fn

        self._lock = threading.Lock()          # start/stop/shutdown
        self._click_lock = threading.Lock()    # active-check + click atomicity
        self._active = threading.Event()
        self._shutdown = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._mouse: Mouse | None = None
        self._recreate_capture = False
        self._requests: queue.SimpleQueue = queue.SimpleQueue()
        self._exclusions: list[Rect] = []

        self.status = Status.IDLE
        self.message = "нажмите F6 в окне браузера с игрой"
        self.target_message = ""
        self.error: str | None = None
        self.capture_name = "-"
        self.region_rect: Rect | None = None
        self.preview_enabled = cfg.debug_view
        self.preview_box = (960, 540)          # max preview size, set by the UI
        self.debug_frame: DebugFrame | None = None
        self._debug_seq = 0
        self.on_sample: Callable[[SampleResult], None] | None = None
        self.on_state_change: Callable[[], None] | None = None
        self.loops_started = 0                 # for tests: number of worker threads ever started
        self.preview_frames = 0                # for tests: published preview frames
        if self.preview_enabled:
            self._ensure_thread()

    # ================================================================ control
    @property
    def active(self) -> bool:
        return self._active.is_set()

    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name="automation", daemon=True)
            self.loops_started += 1
            self._thread.start()

    def _ensure_thread_safe(self) -> None:
        with self._lock:
            if not self._shutdown.is_set():
                self._ensure_thread()

    def start(self) -> bool:
        with self._lock:
            if self._shutdown.is_set() or self._active.is_set():
                return False
            try:
                self._mouse = self._mouse_factory(self.cfg.dry_run)
            except MouseError as exc:
                self._set_status(Status.ERROR, str(exc))
                return False
            self.error = None
            self.region.rebind()
            self.stats.session_started()
            self._ensure_thread()
            self._set_status(Status.SEARCHING, "запуск…")
            self._active.set()
            self._wake.set()
        self._notify()
        return True

    def stop(self, reason: str = "остановлено") -> None:
        with self._lock:
            self._active.clear()
            with self._click_lock:   # waits for an in-flight click, then no more clicks
                pass
            if self.status != Status.ERROR:
                self._set_status(Status.IDLE, reason)
            self._wake.set()
        self._notify()

    def toggle(self) -> bool:
        if self._active.is_set():
            self.stop()
            return False
        return self.start()

    def emergency_stop(self) -> None:
        self.stop("аварийная остановка")

    def shutdown(self, timeout: float = 3.0) -> bool:
        with self._lock:
            self._shutdown.set()
            self._active.clear()
            with self._click_lock:
                pass
            self._wake.set()
            th = self._thread
        if th is not None and th is not threading.current_thread():
            th.join(timeout)
            return not th.is_alive()
        return True

    def set_preview(self, enabled: bool) -> None:
        self.preview_enabled = bool(enabled)
        if enabled:
            self._ensure_thread_safe()
        self._wake.set()

    # ------------------------------------------------------------- requests
    def request_sample(self, box: tuple[int, int, int, int]) -> None:
        """Learn the target from a box in screen coordinates (outlined on the preview)."""
        self._request(("sample", tuple(int(v) for v in box)))

    def request_cursor_capture(self) -> None:
        """Learn the target under the mouse cursor (F8)."""
        self._request(("cursor", None))

    # backwards compatible name (hotkey action "calibrate")
    request_calibration = request_cursor_capture

    def reset_target(self) -> None:
        self._request(("reset", None))

    def _request(self, req) -> None:
        self._requests.put(req)
        self._ensure_thread_safe()
        self._wake.set()

    def apply_config(self, cfg: AppConfig) -> None:
        """Apply new settings while running (called from the UI thread)."""
        cfg.validate()
        backend_changed = cfg.capture_backend != self.cfg.capture_backend
        mode_changed = cfg.target.mode != self.cfg.target.mode
        self.cfg = cfg
        self.finder.update_config(cfg.detector, cfg.target)
        if mode_changed:
            self._request(("reset", None))
        if self.region.area != cfg.capture_area:
            self.region.area = cfg.capture_area
            self.region.reset()
        self.region.require_focus = cfg.require_focus
        if backend_changed:
            self._recreate_capture = True
        self.set_preview(cfg.debug_view)

    def set_exclusions(self, rects: list[Rect]) -> None:
        """Screen rectangles of our own windows: never analysed, never clicked."""
        self._exclusions = list(rects)

    def _set_status(self, status: Status, message: str) -> None:
        self.status = status
        self.message = message

    def _notify(self) -> None:
        cb = self.on_state_change
        if cb is not None:
            try:
                cb()
            except Exception:  # pragma: no cover
                log.exception("state callback failed")

    # ================================================================ worker
    def _make_capture(self) -> Capture:
        if self._capture_factory is not None:
            return self._capture_factory()
        return create_capture(self.cfg.capture_backend)

    def _run(self) -> None:
        self._capture: Capture | None = None
        try:
            self._loop()
        except Exception as exc:  # pragma: no cover - last resort
            log.exception("automation loop crashed")
            self._active.clear()
            self._set_status(Status.ERROR, f"внутренняя ошибка: {exc}")
            self._notify()
        finally:
            if self._capture is not None:
                try:
                    self._capture.close()
                except Exception:
                    pass
                self._capture = None

    def _idle_wait(self, seconds: float) -> None:
        self._wake.wait(seconds)
        self._wake.clear()

    def _loop(self) -> None:
        cap_errors = 0
        last_res: FindResult | None = None
        last_rect: Rect | None = None
        last_frame: np.ndarray | None = None
        last_click_t = -1e9
        first_click_t: float | None = None     # first click on the current target position
        clicks_on_pos = 0
        last_pub = -1e9
        last_found_t = self.clock()
        lost_since: float | None = None
        lost_counted = False
        had_target = False
        while not self._shutdown.is_set():
            active = self._active.is_set()
            watching = self.preview_enabled
            if self._recreate_capture:
                self._recreate_capture = False
                if self._capture is not None:
                    self._capture.close()
                    self._capture = None
                last_res = None
            if not active and not watching and self._requests.empty():
                had_target = False
                self._idle_wait(0.5)
                continue
            if self._capture is None:
                try:
                    self._capture = self._make_capture()
                    self.capture_name = self._capture.name
                except Exception as exc:
                    self._capture_problem(f"захват экрана недоступен: {exc}", fatal=True)
                    self._idle_wait(1.0)
                    continue
            capture = self._capture

            now = self.clock()
            rect, why = self.region.get(now, need_focus=active, follow=not active)
            self.region_rect = rect
            if rect is None:
                if active:
                    self._set_status(Status.PAUSED, why)
                else:
                    self.target_message = why
                self._publish_placeholder(why)
                last_res = None
                self._idle_wait(0.05)
                continue
            if rect != last_rect:
                self.finder.reset_map()
                last_res = None

            # ------------------------------------------------ capture
            t0 = self.clock()
            try:
                frame = capture.grab(rect)
                cap_errors = 0
            except CaptureError as exc:
                cap_errors += 1
                self.stats.incr("errors")
                log.warning("%s", exc)
                if cap_errors % 3 == 0:
                    capture.close()
                    self._capture = None
                    if self.cfg.capture_backend != "mss" and cap_errors >= 6:
                        self._capture_factory = self._capture_factory or (lambda: create_capture("mss"))
                if cap_errors >= self.MAX_CAPTURE_ERRORS:
                    self._capture_problem(f"ошибка захвата экрана: {exc}", fatal=True)
                    cap_errors = 0
                else:
                    time.sleep(0.01)
                continue
            t1 = self.clock()

            # ------------------------------------------------ analyse
            new_frame = frame is not None
            if new_frame:
                frame = self._mask_exclusions(frame, rect)
                last_frame = frame
                res = self.finder.process(frame, t1)
                last_res = res
            elif last_res is not None and rect == last_rect:
                res = last_res      # screen unchanged: previous result is still valid
            else:
                time.sleep(0.0005)
                continue
            last_rect = rect
            if not self._requests.empty() and last_frame is not None:
                self._handle_requests(last_frame, rect)
                continue
            t2 = self.clock()
            self.stats.frame((t1 - t0) * 1000.0, res.ms if new_frame else None, new_frame, res.target is not None)
            tg = res.target
            self.target_message = res.message

            # ------------------------------------------------ bookkeeping
            if new_frame and (res.jumped or res.acquired):
                if active and res.jumped and first_click_t is not None:
                    self.stats.jump((t2 - first_click_t) * 1000.0)
                first_click_t, clicks_on_pos = None, 0
            if active:
                if tg is None:
                    if had_target and lost_since is None:
                        lost_since = t2
                    if lost_since is not None and not lost_counted \
                            and t2 - lost_since > self.cfg.target.lost_timeout_ms / 1000.0:
                        self.stats.incr("lost")
                        lost_counted = True
                else:
                    if lost_counted and lost_since is not None:
                        self.stats.reacquired((t2 - lost_since) * 1000.0)
                    lost_since, lost_counted, had_target = None, False, True

            # ------------------------------------------------ click
            if active and tg is not None:
                interval = self.cfg.click_interval_ms / 1000.0
                gap = interval if interval > 0 else (0.0 if new_frame else 0.001)
                due = last_click_t + gap
                wait = due - t2
                if 0 < wait <= 0.006:
                    time.sleep(wait)          # almost due: click with this (fresh) frame
                    wait = 0
                if wait <= 0:
                    sx, sy = frame_to_screen(rect, tg.click[0], tg.click[1])
                    if any(r.contains(sx, sy) for r in self._exclusions):
                        self.target_message = "цель под окном программы — клик пропущен"
                    else:
                        ok = self._click(sx, sy, t0)
                        if ok is None:
                            continue
                        if ok:
                            tc = self.clock()
                            # keep the average rate exact (no drift), but never "catch up" with bursts
                            last_click_t = due if interval > 0 and tc - due < gap else tc
                            clicks_on_pos += 1
                            if first_click_t is None:
                                first_click_t = tc
                else:
                    time.sleep(min(wait - 0.005, 0.004))
                # a target that never reacts to clicks is not the worm
                unlock = self.cfg.target.unlock_after_ms / 1000.0
                if (unlock > 0 and not self.cfg.dry_run and clicks_on_pos >= 10
                        and first_click_t is not None and t2 - first_click_t > unlock):
                    self.finder.mark_unresponsive(t2)
                    self.stats.incr("ignored")
                    first_click_t, clicks_on_pos = None, 0
                    self.target_message = "объект не реагирует на клики — это не червь, ищу дальше"

            # ------------------------------------------------ status
            if active:
                if tg is not None:
                    last_found_t = t2
                    self._set_status(Status.ACTIVE, "кликаю по цели" + (" (тест — без кликов)"
                                                                        if self.cfg.dry_run else ""))
                else:
                    self._set_status(Status.SEARCHING, res.message)
            elif self.status not in (Status.IDLE, Status.ERROR):
                self._set_status(Status.IDLE, "остановлено")

            # ------------------------------------------------ preview
            fps = max(10, self.cfg.preview_fps)
            if watching and new_frame and t2 - last_pub >= 0.92 / fps:
                last_pub = t2
                self._publish(frame, rect, res, t2)
            if not active:
                # watch mode: no need to analyse faster than the preview shows
                rest = 1.0 / fps - (self.clock() - t0)
                if rest > 0:
                    self._idle_wait(rest)
            elif tg is None and self.cfg.idle_sleep_ms > 0 and t2 - last_found_t > 0.5:
                time.sleep(self.cfg.idle_sleep_ms / 1000.0)
            elif not new_frame and tg is None:
                time.sleep(0.0005)

    def _click(self, sx: int, sy: int, t_frame: float) -> bool | None:
        mouse = self._mouse
        with self._click_lock:
            if not self._active.is_set() or mouse is None:
                return False
            tc0 = self.clock()
            try:
                mouse.click(sx, sy)
            except MouseError as exc:
                self._active.clear()
                self._set_status(Status.ERROR, str(exc))
                self.error = str(exc)
                self._notify()
                return None
            tc1 = self.clock()
        self.stats.click((tc1 - tc0) * 1000.0, (tc1 - t_frame) * 1000.0)
        return True

    def _capture_problem(self, message: str, fatal: bool) -> None:
        if self._active.is_set() and fatal:
            self._active.clear()
            self.error = message
            self._set_status(Status.ERROR, message)
            self._notify()
        else:
            self.target_message = message
        self._publish_placeholder(message)

    def _mask_exclusions(self, frame: np.ndarray, rect: Rect) -> np.ndarray:
        for r in self._exclusions:
            inter = rect.intersect(r)
            if inter is None:
                continue
            if not frame.flags.writeable:
                frame = frame.copy()
            frame[inter.top - rect.top:inter.bottom - rect.top,
                  inter.left - rect.left:inter.right - rect.left] = 0
        return frame

    # ============================================================ preview
    def _publish(self, frame: np.ndarray, rect: Rect, res: FindResult, t: float) -> None:
        fh, fw = frame.shape[:2]
        bw, bh = self.preview_box
        scale = min(1.0, bw / max(1, fw), bh / max(1, fh))
        if scale < 1.0:
            size = (max(1, int(fw * scale)), max(1, int(fh * scale)))
            img = cv2.resize(frame, size, interpolation=cv2.INTER_AREA if scale > 0.34 else cv2.INTER_LINEAR)
        else:
            img = frame.copy()
        self._debug_seq += 1
        self.preview_frames += 1
        self.debug_frame = DebugFrame(img, scale, rect, res.target, res.state.name, res.message, res.ms,
                                      self._debug_seq, t, tuple(self.finder.ignored))

    def _publish_placeholder(self, message: str) -> None:
        df = self.debug_frame
        if df is not None and df.image is None and df.message == message:
            return
        self._debug_seq += 1
        self.debug_frame = DebugFrame(None, 1.0, None, None, "NONE", message, 0.0, self._debug_seq, self.clock())

    # ============================================================ requests
    def _handle_requests(self, frame: np.ndarray, rect: Rect) -> None:
        while True:
            try:
                kind, arg = self._requests.get_nowait()
            except queue.Empty:
                return
            res: SampleResult | None = None
            try:
                if kind == "reset":
                    self.finder.reset_target()
                    continue
                if kind == "sample":
                    x, y, w, h = arg
                    res = self.finder.learn_sample(frame, box=(x - rect.left, y - rect.top, w, h), now=self.clock())
                elif kind == "cursor":
                    pos = self._cursor_fn()
                    if pos is None:
                        res = SampleResult(False, "не удалось получить позицию курсора")
                    elif not rect.contains(*pos):
                        res = SampleResult(False, "курсор вне области захвата — наведите его на червя в игре")
                    else:
                        res = self.finder.learn_sample(frame, point=(pos[0] - rect.left, pos[1] - rect.top),
                                                       now=self.clock())
            except Exception as exc:  # pragma: no cover - defensive
                log.exception("sample failed")
                res = SampleResult(False, f"ошибка: {exc}")
            if res is not None:
                cb = self.on_sample
                if cb is not None:
                    try:
                        cb(res)
                    except Exception:  # pragma: no cover
                        log.exception("sample callback failed")


__all__ = ["AutomationEngine", "DebugFrame", "Status", "TargetState"]
