"""Click decision logic between frames.

Main rule: **every click is based on a fresh detection, and the spot of the
previous click is never clicked again until the worm has demonstrably left
it** (or ``stale_timeout`` expired, meaning the click was not registered).

States
  search      - no confident target in the frame, nothing is clicked
  track       - target visible, click when stable / rate limit allows
  await_jump  - a click was sent; while the worm is still visible at the
                same spot with the same pose the frame is considered stale
                (capture/browser latency) and no click is sent

Additionally:
  * a confirmed jump (worm vanished from the click spot and appeared
    elsewhere) is counted as an approximate hit;
  * a spot clicked ``blacklist_after`` times without any jump (a static
    picture of a worm in the page UI, a decoy) is ignored for a while;
  * loss / re-acquisition of the target is measured.
All coordinates are screen coordinates, so moving the browser window does
not confuse the logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import TrackerConfig
from .geometry import Rect
from .stats import Stats


@dataclass
class Target:
    x: float            # click point
    y: float
    cx: float           # silhouette centre
    cy: float
    length: float
    angle: float        # 0..180
    area: int
    score: float
    bbox: Rect


@dataclass
class Decision:
    click: bool
    target: Target | None
    reason: str
    state: str


@dataclass
class _Pending:
    target: Target
    t: float


@dataclass
class _Spot:
    x: float
    y: float
    r: float
    count: int
    t: float


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


class Tracker:
    def __init__(self, cfg: TrackerConfig | None = None, stats: Stats | None = None):
        self.cfg = cfg or TrackerConfig()
        self.stats = stats
        self.reset()

    def reset(self) -> None:
        self._pending: _Pending | None = None
        self._prev: Target | None = None
        self._streak = 0
        self._last_seen: float | None = None
        self._lost = False
        self._lost_since = 0.0
        self._last_click = -1e9
        self._spots: list[_Spot] = []
        self._black: list[_Spot] = []
        self.state = "search"

    # ---------------------------------------------------------------- helpers
    def _same_spot(self, t: Target, p: Target) -> bool:
        r = max(10.0, 0.5 * max(p.length, t.length))
        if math.hypot(t.cx - p.cx, t.cy - p.cy) > r:
            return False
        # nearby but clearly different pose -> it is a new position
        if _angle_diff(t.angle, p.angle) > 25.0:
            return False
        ratio = t.area / max(1, p.area)
        return 0.65 <= ratio <= 1.55

    def _blacklisted(self, t: Target) -> bool:
        return any(math.hypot(t.cx - b.x, t.cy - b.y) <= b.r for b in self._black)

    def _expire(self, now: float) -> None:
        ttl = self.cfg.blacklist_ttl_s
        if self._black:
            self._black = [b for b in self._black if now - b.t < ttl]
        if self._spots:
            self._spots = [s for s in self._spots if now - s.t < ttl]

    def _register_no_jump(self, p: Target, now: float) -> None:
        if self.stats:
            self.stats.incr("no_jump")
        n = self.cfg.blacklist_after
        if n <= 0:
            return
        for s in self._spots:
            if math.hypot(p.cx - s.x, p.cy - s.y) <= s.r:
                s.count += 1
                s.t = now
                if s.count >= n:
                    self._black.append(_Spot(s.x, s.y, s.r, s.count, now))
                    self._spots.remove(s)
                    if self.stats:
                        self.stats.incr("blacklisted")
                return
        self._spots.append(_Spot(p.cx, p.cy, max(10.0, 0.5 * p.length), 1, now))

    @property
    def blacklist(self) -> list[tuple[float, float, float]]:
        return [(b.x, b.y, b.r) for b in self._black]

    def hint(self) -> Rect | None:
        """Search window for the short-term tracking optimisation."""
        if self._pending is not None or self._prev is None or self._streak <= 0:
            return None
        if self._streak >= self.cfg.stable_frames:
            return None
        p = self._prev
        m = int(p.length * 0.75) + 8
        return Rect(int(p.cx - m), int(p.cy - m), 2 * m, 2 * m)

    # ---------------------------------------------------------------- update
    def update(self, targets: list[Target], now: float, no_target_reason: str = "цель не найдена") -> Decision:
        self._expire(now)
        cands = [t for t in targets if not self._blacklisted(t)]
        if targets and not cands:
            no_target_reason = "найдена только исключённая статичная цель"

        if self._pending is not None:
            p = self._pending
            stale = [t for t in cands if self._same_spot(t, p.target)]
            if stale:
                if now - p.t < self.cfg.stale_timeout_ms / 1000.0:
                    if self.stats:
                        self.stats.incr("stale_frames")
                    self.state = "await_jump"
                    return Decision(False, stale[0], "ожидание прыжка после клика", self.state)
                # the worm stayed: the click was not registered -> allow a retry
                self._pending = None
                self._register_no_jump(p.target, now)
                cands = [t for t in cands if not self._blacklisted(t)]
            elif cands:
                self._pending = None
                if self.stats:
                    self.stats.jump((now - p.t) * 1000.0)

        if not cands:
            self._streak = 0
            self._prev = None
            if (not self._lost and self._last_seen is not None
                    and now - self._last_seen > self.cfg.lost_timeout_ms / 1000.0):
                self._lost = True
                self._lost_since = self._last_seen
                if self.stats:
                    self.stats.incr("lost")
            self.state = "await_jump" if self._pending is not None else "search"
            return Decision(False, None, no_target_reason, self.state)

        best = cands[0]
        if self._lost:
            self._lost = False
            if self.stats:
                self.stats.reacquired((now - self._lost_since) * 1000.0)
        self._last_seen = now

        prev = self._prev
        if prev is not None and math.hypot(best.cx - prev.cx, best.cy - prev.cy) < max(4.0, 0.25 * best.length):
            self._streak += 1
        else:
            self._streak = 1
        self._prev = best

        self.state = "track"
        if self._streak < self.cfg.stable_frames:
            return Decision(False, best, f"подтверждение цели {self._streak}/{self.cfg.stable_frames}", self.state)
        if self.cfg.max_cps > 0 and now - self._last_click < 1.0 / self.cfg.max_cps:
            return Decision(False, best, "ограничение кликов в секунду", self.state)
        return Decision(True, best, "клик", self.state)

    def on_click(self, target: Target, now: float) -> None:
        self._pending = _Pending(target, now)
        self._last_click = now
        self._streak = 0
        self._prev = None
        self.state = "await_jump"
