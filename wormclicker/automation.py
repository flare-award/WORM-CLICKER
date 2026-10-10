"""Automation engine: the single worker thread running

    capture -> detect -> decide -> click -> capture next frame ...

Guarantees
  * exactly one worker thread per engine; ``start()`` while running is a
    no-op, so competing click loops are impossible;
  * after ``stop()`` returns no further click is sent (the active flag is
    checked under the same lock that wraps the click call);
  * the worker idles on an Event while inactive (no CPU use);
  * capture objects live in the worker thread only;
  * the UI only reads snapshots (stats, status, last debug frame).
"""

from __future__ import annotations

import enum
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .calibration import PATCH, CalibrationResult, apply_calibration, calibrate_patch
from .capture import Capture, CaptureError, create_capture, list_monitors
from .config import AppConfig
from .detector import Detection, WormDetector
from .geometry import Rect, clip_to_monitors, frame_to_screen
from .mouse import Mouse, MouseError, create_mouse
from .region import RegionProvider
from .stats import Stats
from .tracker import Decision, Target, Tracker
from . import winapi

log = logging.getLogger(__name__)


class Status(str, enum.Enum):
    IDLE = "Ожидание"
    PAUSED = "Пауза"
    SEARCHING = "Поиск"
    ACTIVE = "Работа"
    ERROR = "Ошибка"


@dataclass
class DebugFrame:
    frame: np.ndarray
    rect: Rect
    detection: Detection
    decision: Decision | None
    blacklist: list[tuple[float, float, float]]
    t: float


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
        self.detector = WormDetector(cfg.detector)
        self.tracker = Tracker(cfg.tracker, self.stats)
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
        self.message = "нажмите F6 в окне браузера, чтобы начать"
        self.error: str | None = None
        self.capture_name = "-"
        self.region_rect: Rect | None = None
        self.debug_frame: DebugFrame | None = None
        self.on_calibration: Callable[[CalibrationResult], None] | None = None
        self.on_state_change: Callable[[], None] | None = None
        self.loops_started = 0                 # for tests: number of worker threads ever started

    # ================================================================ control
    @property
    def active(self) -> bool:
        return self._active.is_set()

    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name="automation", daemon=True)
            self.loops_started += 1
            self._thread.start()

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
            self.tracker.reset()
            self.region.reset()
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

    def request_calibration(self) -> None:
        self._requests.put("calibrate")
        self._ensure_thread_safe()
        self._wake.set()

    def _ensure_thread_safe(self) -> None:
        with self._lock:
            if not self._shutdown.is_set():
                self._ensure_thread()

    def apply_config(self, cfg: AppConfig) -> None:
        """Apply new settings while running (called from the UI thread)."""
        backend_changed = cfg.capture_backend != self.cfg.capture_backend
        self.cfg = cfg
        cfg.validate()
        self.detector.update_config(cfg.detector)
        self.tracker.cfg = cfg.tracker
        if self.region.area != cfg.capture_area:
            self.region.area = cfg.capture_area
            self.region.reset()
        self.region.require_focus = cfg.require_focus
        if backend_changed:
            self._recreate_capture = True

    def set_exclusions(self, rects: list[Rect]) -> None:
        """Screen rectangles of our own windows: never detected, never clicked."""
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
        capture: Capture | None = None
        try:
            capture = self._loop()
        except Exception as exc:  # pragma: no cover - last resort
            log.exception("automation loop crashed")
            self._active.clear()
            self._set_status(Status.ERROR, f"внутренняя ошибка: {exc}")
            self._notify()
        finally:
            if capture is not None:
                try:
                    capture.close()
                except Exception:
                    pass

    def _loop(self) -> Capture | None:
        capture: Capture | None = None
        cap_errors = 0
        last_det: Detection | None = None
        last_rect: Rect | None = None
        last_found_t = self.clock()
        last_debug_t = -1e9
        while not self._shutdown.is_set():
            # ------------------------------------------------ idle / requests
            if not self._active.is_set() or self._recreate_capture:
                if self._recreate_capture and capture is not None:
                    capture.close()
                    capture = None
                    self._recreate_capture = False
                capture = self._handle_requests(capture)
                if not self._active.is_set():
                    last_det = None
                    self._wake.wait(0.5)
                    self._wake.clear()
                    continue
            capture = self._handle_requests(capture)
            if capture is None:
                try:
                    capture = self._make_capture()
                    self.capture_name = capture.name
                except Exception as exc:
                    self._fail(f"захват экрана недоступен: {exc}")
                    continue

            now = self.clock()
            rect, why = self.region.get(now)
            self.region_rect = rect
            if rect is None:
                self._set_status(Status.PAUSED, why)
                last_det = None
                self.tracker.reset()
                self._wake.wait(0.05)
                self._wake.clear()
                continue

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
                    capture = None
                    if self.cfg.capture_backend != "mss" and cap_errors >= 6:
                        self._capture_factory = self._capture_factory or (lambda: create_capture("mss"))
                if cap_errors >= self.MAX_CAPTURE_ERRORS:
                    self._fail(f"ошибка захвата экрана: {exc}")
                else:
                    time.sleep(0.01)
                continue
            t1 = self.clock()

            # ------------------------------------------------ detect
            new_frame = frame is not None
            if new_frame:
                frame = self._mask_exclusions(frame, rect)
                hint = self._hint_for(rect)
                det = self.detector.detect(frame, hint)
                last_det, last_rect = det, rect
                det_ms = det.ms
            elif last_det is not None and last_rect == rect:
                det = last_det     # screen unchanged: previous detection still valid
                det_ms = None
            else:
                time.sleep(0.0005)
                continue
            t2 = self.clock()

            targets = self._targets(det, rect)
            decision = self.tracker.update(targets, t2, det.reason)
            self.stats.frame((t1 - t0) * 1000.0, det_ms, new_frame, bool(targets))

            # ------------------------------------------------ click
            clicked = False
            if decision.click and decision.target is not None:
                tg = decision.target
                sx, sy = frame_to_screen(rect, tg.x - rect.left, tg.y - rect.top)
                if any(r.contains(sx, sy) for r in self._exclusions):
                    decision = Decision(False, tg, "точка под окном программы — клик пропущен", decision.state)
                else:
                    clicked = self._click(sx, sy, tg, t0)
                    if clicked is None:
                        continue

            # ------------------------------------------------ status / debug
            if targets:
                last_found_t = t2
                self._set_status(Status.ACTIVE, decision.reason)
            else:
                self._set_status(Status.SEARCHING, decision.reason)
            if self.cfg.debug_view and new_frame and t2 - last_debug_t > 0.1:
                last_debug_t = t2
                self.debug_frame = DebugFrame(np.array(frame, copy=True), rect, det, decision,
                                              self.tracker.blacklist, t2)
            if not clicked:
                if not new_frame:
                    time.sleep(0.0005)
                elif (not targets and self.cfg.idle_sleep_ms > 0 and t2 - last_found_t > 0.5):
                    time.sleep(self.cfg.idle_sleep_ms / 1000.0)
        return capture

    def _click(self, sx: int, sy: int, tg: Target, t_frame: float) -> bool | None:
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
        self.tracker.on_click(tg, tc1)
        self.stats.click((tc1 - tc0) * 1000.0, (tc1 - t_frame) * 1000.0, tg.score)
        return True

    def _fail(self, message: str) -> None:
        self._active.clear()
        self.error = message
        self._set_status(Status.ERROR, message)
        self._notify()

    def _hint_for(self, rect: Rect) -> tuple[int, int, int, int] | None:
        h = self.tracker.hint()
        if h is None:
            return None
        return (h.left - rect.left, h.top - rect.top, h.width, h.height)

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

    @staticmethod
    def _targets(det: Detection, rect: Rect) -> list[Target]:
        out = []
        for c in det.candidates:
            if not c.accepted:
                continue
            bx, by, bw, bh = c.bbox
            out.append(Target(rect.left + c.click[0], rect.top + c.click[1],
                              rect.left + c.center[0], rect.top + c.center[1],
                              c.length, c.angle, c.area, c.score,
                              Rect(rect.left + bx, rect.top + by, bw, bh)))
        return out

    # ============================================================ calibration
    def _handle_requests(self, capture: Capture | None) -> Capture | None:
        while True:
            try:
                req = self._requests.get_nowait()
            except queue.Empty:
                return capture
            if req == "calibrate":
                capture = self._calibrate(capture)

    def _calibrate(self, capture: Capture | None) -> Capture | None:
        res: CalibrationResult
        try:
            if capture is None:
                capture = self._make_capture()
                self.capture_name = capture.name
            pos = self._cursor_fn()
            if pos is None:
                res = CalibrationResult(False, "не удалось получить позицию курсора")
            else:
                mons = self.region.monitors()
                half = PATCH // 2
                want = Rect(pos[0] - half, pos[1] - half, PATCH, PATCH)
                rect = clip_to_monitors(want, mons) if mons else want
                frame = None
                for _ in range(20):
                    frame = capture.grab(rect)
                    if frame is not None:
                        break
                    time.sleep(0.01)
                if frame is None:
                    res = CalibrationResult(False, "не удалось захватить экран")
                else:
                    res = calibrate_patch(frame, (pos[0] - rect.left, pos[1] - rect.top))
                    if res.ok:
                        apply_calibration(self.cfg.detector, res)
                        self.detector.update_config(self.cfg.detector)
        except Exception as exc:
            res = CalibrationResult(False, f"ошибка калибровки: {exc}")
        cb = self.on_calibration
        if cb is not None:
            try:
                cb(res)
            except Exception:  # pragma: no cover
                log.exception("calibration callback failed")
        return capture
