"""Thread-safe runtime statistics.

The worker thread writes, the UI thread reads a snapshot a few times per
second. Rolling rates use a fixed-size deque of timestamps (no unbounded
growth), latencies use exponential moving averages.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field


class Ema:
    __slots__ = ("alpha", "value", "n")

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.value = 0.0
        self.n = 0

    def add(self, x: float) -> None:
        self.n += 1
        if self.n == 1:
            self.value = x
        else:
            self.value += self.alpha * (x - self.value)


class RateMeter:
    """Events per second over a sliding window."""

    def __init__(self, window_s: float = 1.0, maxlen: int = 4096):
        self.window = window_s
        self._t: deque[float] = deque(maxlen=maxlen)

    def add(self, t: float) -> None:
        self._t.append(t)

    def rate(self, now: float) -> float:
        lim = now - self.window
        while self._t and self._t[0] < lim:
            self._t.popleft()
        return len(self._t) / self.window

    def clear(self) -> None:
        self._t.clear()


@dataclass
class StatsSnapshot:
    session_s: float = 0.0
    clicks: int = 0
    jumps: int = 0                 # target jumped after being clicked (~ hits)
    cps: float = 0.0
    jumps_ps: float = 0.0
    avg_cps: float = 0.0
    fps: float = 0.0
    frames: int = 0
    no_new_frame: int = 0
    no_target_frames: int = 0
    capture_ms: float = 0.0
    detect_ms: float = 0.0
    click_ms: float = 0.0
    pipeline_ms: float = 0.0       # frame grab start -> click sent
    jump_ms: float = 0.0           # first click on a position -> new position seen
    lost: int = 0
    reacquired: int = 0
    reacquire_ms: float = 0.0
    ignored: int = 0               # objects ignored because they did not react to clicks
    errors: int = 0
    extra: dict = field(default_factory=dict)


class Stats:
    def __init__(self, clock=time.perf_counter):
        self._clock = clock
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._start = None
            self._clicks = RateMeter(maxlen=20000)
            self._jumps = RateMeter()
            self._frames = RateMeter()
            self.s = StatsSnapshot()
            self._cap = Ema()
            self._det = Ema()
            self._clk = Ema()
            self._pipe = Ema()
            self._jump = Ema()
            self._reacq = Ema()

    # ----------------------------------------------------------- writers
    def session_started(self) -> None:
        with self._lock:
            if self._start is None:
                self._start = self._clock()

    def frame(self, capture_ms: float, detect_ms: float | None, new_frame: bool, found: bool) -> None:
        now = self._clock()
        with self._lock:
            self.s.frames += 1
            self._frames.add(now)
            self._cap.add(capture_ms)
            if detect_ms is not None:
                self._det.add(detect_ms)
            if not new_frame:
                self.s.no_new_frame += 1
            if not found:
                self.s.no_target_frames += 1

    def click(self, click_ms: float, pipeline_ms: float) -> None:
        now = self._clock()
        with self._lock:
            self.s.clicks += 1
            self._clicks.add(now)
            self._clk.add(click_ms)
            self._pipe.add(pipeline_ms)

    def jump(self, jump_ms: float) -> None:
        now = self._clock()
        with self._lock:
            self.s.jumps += 1
            self._jumps.add(now)
            self._jump.add(jump_ms)

    def incr(self, name: str, n: int = 1) -> None:
        with self._lock:
            setattr(self.s, name, getattr(self.s, name) + n)

    def reacquired(self, ms: float) -> None:
        with self._lock:
            self.s.reacquired += 1
            self._reacq.add(ms)

    # ----------------------------------------------------------- reader
    def snapshot(self) -> StatsSnapshot:
        now = self._clock()
        with self._lock:
            s = StatsSnapshot(**{k: v for k, v in self.s.__dict__.items() if k != "extra"})
            s.session_s = (now - self._start) if self._start is not None else 0.0
            s.cps = self._clicks.rate(now)
            s.jumps_ps = self._jumps.rate(now)
            s.fps = self._frames.rate(now)
            s.avg_cps = s.clicks / s.session_s if s.session_s > 0 else 0.0
            s.capture_ms = self._cap.value
            s.detect_ms = self._det.value
            s.click_ms = self._clk.value
            s.pipeline_ms = self._pipe.value
            s.jump_ms = self._jump.value
            s.reacquire_ms = self._reacq.value
            return s
