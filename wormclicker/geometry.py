"""Screen geometry helpers: rectangles, monitors, virtual desktop.

All coordinates are physical pixels of the Windows virtual desktop. The
process is made per-monitor DPI aware at start-up, so screen capture,
window rectangles and SetCursorPos all use the same coordinate space,
including monitors with negative coordinates (left of / above primary).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def empty(self) -> bool:
        return self.width <= 0 or self.height <= 0

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x < self.right and self.top <= y < self.bottom

    def contains_rect(self, other: "Rect") -> bool:
        return (other.left >= self.left and other.top >= self.top
                and other.right <= self.right and other.bottom <= self.bottom)

    def intersect(self, other: "Rect") -> "Rect | None":
        x0, y0 = max(self.left, other.left), max(self.top, other.top)
        x1, y1 = min(self.right, other.right), min(self.bottom, other.bottom)
        if x1 <= x0 or y1 <= y0:
            return None
        return Rect(x0, y0, x1 - x0, y1 - y0)

    def to_mss(self) -> dict:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    @classmethod
    def from_ltrb(cls, left: int, top: int, right: int, bottom: int) -> "Rect":
        return cls(int(left), int(top), int(right - left), int(bottom - top))

    @classmethod
    def from_mss(cls, m: dict) -> "Rect":
        return cls(int(m["left"]), int(m["top"]), int(m["width"]), int(m["height"]))


def bounding_rect(rects: Iterable[Rect]) -> Rect | None:
    rects = list(rects)
    if not rects:
        return None
    x0 = min(r.left for r in rects)
    y0 = min(r.top for r in rects)
    x1 = max(r.right for r in rects)
    y1 = max(r.bottom for r in rects)
    return Rect(x0, y0, x1 - x0, y1 - y0)


def clip_to_monitors(rect: Rect, monitors: Sequence[Rect]) -> Rect | None:
    """Clip a rectangle to the virtual desktop (bounding box of monitors)."""
    vs = bounding_rect(monitors)
    if vs is None:
        return None
    return rect.intersect(vs)


def monitor_index_containing(rect: Rect, monitors: Sequence[Rect]) -> int | None:
    """Index of the monitor that fully contains ``rect``."""
    for i, m in enumerate(monitors):
        if m.contains_rect(rect):
            return i
    return None


def monitor_at_point(x: float, y: float, monitors: Sequence[Rect]) -> int | None:
    for i, m in enumerate(monitors):
        if m.contains(x, y):
            return i
    return None


def frame_to_screen(region: Rect, fx: float, fy: float) -> tuple[int, int]:
    """Frame (captured region) coordinates -> integer screen coordinates."""
    sx = region.left + int(round(fx))
    sy = region.top + int(round(fy))
    # keep the click inside the captured region even after rounding
    sx = min(max(sx, region.left), region.right - 1)
    sy = min(max(sy, region.top), region.bottom - 1)
    return sx, sy


def normalize_absolute(x: int, y: int, virtual: Rect) -> tuple[int, int]:
    """Screen -> 0..65535 coordinates for SendInput MOUSEEVENTF_VIRTUALDESK."""
    nx = int(round((x - virtual.left) * 65535 / max(1, virtual.width - 1)))
    ny = int(round((y - virtual.top) * 65535 / max(1, virtual.height - 1)))
    return min(65535, max(0, nx)), min(65535, max(0, ny))
