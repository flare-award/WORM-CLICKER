"""Offline game simulator for end-to-end tests and benchmarks.

``SimulatedGame`` renders the synthetic worm with distractors; a click
inside the worm counts as a hit and teleports it to a random position with
a random rotation. ``render_latency_s`` keeps the old picture on screen for
a while after a hit (browser + compositor latency). ``decoy`` adds a static
picture of the same worm to the page: it must never become the target.
``kick()`` is the user's first manual click on the worm.

``SimCapture`` / ``SimMouse`` connect the simulator to the real
``AutomationEngine`` at an arbitrary virtual-desktop offset (negative
coordinates of a left secondary monitor are covered by tests).
"""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from .geometry import Rect
from .sprite import make_background, render_worm
from .synthetic import scene_with_distractors


class SimulatedGame:
    def __init__(self, width: int = 1280, height: int = 720, pixel_size: int = 4, seed: int = 0,
                 distractors: bool = True, render_latency_s: float = 0.0, smooth: bool = False,
                 decoy: bool = False, clock=time.perf_counter):
        self.w, self.h = width, height
        self.ps = pixel_size
        self.rng = np.random.default_rng(seed)
        self.smooth = smooth
        self.clock = clock
        self.render_latency_s = render_latency_s
        self.distractors = distractors
        if distractors:
            self._bg = scene_with_distractors(width, height, seed)
        else:
            self._bg = make_background(width, height, seed=seed)
        self.decoy_center: tuple[float, float] | None = None
        self.decoy_mask: np.ndarray | None = None
        if decoy:
            self.decoy_center = (width * 0.85, height * 0.85)
            self.decoy_mask = render_worm(self._bg, self.decoy_center, 25.0, pixel_size, smooth)
        self.decoy_clicks = 0
        self._lock = threading.Lock()
        self.center = (width * 0.5, height * 0.5)
        self.angle = 0.0
        self.visible = True
        self._pending: tuple[float, tuple[float, float], float] | None = None
        self._frame: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._old_masks: deque[np.ndarray] = deque(maxlen=3)
        self.hits = 0
        self.misses = 0
        self.clicks_on_old_spot = 0
        self.clicks_while_hidden = 0
        self.hit_times: list[float] = []
        self._new_position()

    # ------------------------------------------------------------- state
    def _random_state(self) -> tuple[tuple[float, float], float]:
        m = 12 * self.ps
        # the synthetic UI distractors occupy the top band (y < 240); the
        # worm overlapping them is covered by detector tests instead
        top = 240 + m if self.distractors else m
        while True:
            c = (float(self.rng.uniform(m, self.w - m)), float(self.rng.uniform(top, self.h - m)))
            dc = self.decoy_center
            if dc is None or abs(c[0] - dc[0]) > 30 * self.ps or abs(c[1] - dc[1]) > 30 * self.ps:
                return c, float(self.rng.uniform(0, 360))

    def _new_position(self) -> None:
        self.center, self.angle = self._random_state()
        self._frame = None

    def _apply_pending(self, now: float) -> None:
        if self._pending is not None and now >= self._pending[0]:
            _, self.center, self.angle = self._pending
            self._pending = None
            self._frame = None

    def set_visible(self, visible: bool) -> None:
        with self._lock:
            self.visible = visible
            self._frame = None

    def teleport(self, center: tuple[float, float], angle: float) -> None:
        with self._lock:
            self.center, self.angle = center, angle
            self._pending = None
            self._frame = None

    # ------------------------------------------------------------- render
    def frame(self, now: float | None = None) -> np.ndarray:
        now = self.clock() if now is None else now
        with self._lock:
            self._apply_pending(now)
            if self._frame is None:
                f = self._bg.copy()
                if self.visible:
                    self._mask = render_worm(f, self.center, self.angle, self.ps, self.smooth)
                else:
                    self._mask = np.zeros((self.h, self.w), bool)
                f.setflags(write=False)
                self._frame = f
            return self._frame

    @property
    def mask(self) -> np.ndarray:
        self.frame()
        assert self._mask is not None
        return self._mask

    # ------------------------------------------------------------- input
    def kick(self) -> bool:
        """The user's manual click on the worm (makes it jump)."""
        mask = self.mask
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return False
        j = int(np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2))
        return self.click(int(xs[j]), int(ys[j]), user=True)

    def click(self, x: int, y: int, now: float | None = None, user: bool = False) -> bool:
        now = self.clock() if now is None else now
        with self._lock:
            self._apply_pending(now)
            inside = 0 <= x < self.w and 0 <= y < self.h
            if inside and self.decoy_mask is not None and self.decoy_mask[y, x]:
                self.decoy_clicks += 1
            if not self.visible:
                self.clicks_while_hidden += 1
                return False
            mask = self._mask
            if mask is None:
                return False
            hit = inside and bool(mask[y, x]) and self._pending is None
            if hit:
                if not user:
                    self.hits += 1
                    self.hit_times.append(now)
                self._old_masks.append(mask)
                c, a = self._random_state()
                if self.render_latency_s > 0:
                    self._pending = (now + self.render_latency_s, c, a)
                else:
                    self.center, self.angle = c, a
                    self._frame = None
                return True
            self.misses += 1
            if inside and (any(m[y, x] for m in self._old_masks) or (self._pending is not None and mask[y, x])):
                self.clicks_on_old_spot += 1
            return False


class SimCapture:
    """Capture backend reading the simulator placed at ``origin`` on the desktop."""

    name = "sim"

    def __init__(self, game: SimulatedGame, origin: tuple[int, int] = (0, 0), lag_frames: int = 0,
                 grab_delay_s: float = 0.0, monitors: list[Rect] | None = None):
        self.game = game
        self.origin = origin
        self.lag = deque(maxlen=lag_frames + 1)
        self.grab_delay_s = grab_delay_s
        self._monitors = monitors or [Rect(origin[0], origin[1], game.w, game.h)]
        self.grabs = 0
        self.closed = False

    def monitors(self) -> list[Rect]:
        return list(self._monitors)

    def grab(self, rect: Rect) -> np.ndarray:
        if self.grab_delay_s:
            time.sleep(self.grab_delay_s)
        self.grabs += 1
        f = self.game.frame()
        self.lag.append(f)
        f = self.lag[0]
        x0, y0 = rect.left - self.origin[0], rect.top - self.origin[1]
        return f[max(0, y0):y0 + rect.height, max(0, x0):x0 + rect.width]

    def close(self) -> None:
        self.closed = True


class SimMouse:
    name = "sim"

    def __init__(self, game: SimulatedGame, origin: tuple[int, int] = (0, 0)):
        self.game = game
        self.origin = origin
        self.clicks: list[tuple[int, int, float]] = []

    def click(self, x: int, y: int) -> None:
        self.clicks.append((x, y, time.perf_counter()))
        self.game.click(x - self.origin[0], y - self.origin[1])
