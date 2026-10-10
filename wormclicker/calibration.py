"""Colour/size calibration from the object under the mouse cursor.

The user hovers the cursor over the worm and presses F8 (or the button in
the UI). A small patch around the cursor is analysed: the coloured blob
under the cursor gives the real hue range of the worm's pink and its size,
which makes the detector robust to a different palette, browser zoom or
monitor colour profile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import DetectorConfig

PATCH = 320  # px, square around the cursor


@dataclass
class CalibrationResult:
    ok: bool
    message: str
    hue_ranges: list[list[int]] | None = None
    area: float = 0.0
    min_saturation: int | None = None


def _circular_range(h: np.ndarray, margin: int = 6) -> list[list[int]]:
    ang = h.astype(np.float64) * (2 * math.pi / 180.0)
    mean = math.atan2(np.sin(ang).mean(), np.cos(ang).mean())
    mean_h = (mean * 180.0 / (2 * math.pi)) % 180.0
    shifted = (h.astype(np.float64) - mean_h + 90.0) % 180.0
    lo = np.percentile(shifted, 2) - margin
    hi = np.percentile(shifted, 98) + margin
    lo_h = int(math.floor(lo + mean_h - 90.0))
    hi_h = int(math.ceil(hi + mean_h - 90.0))
    if hi_h - lo_h >= 179:
        return [[0, 179]]
    lo_h %= 180
    hi_h %= 180
    if lo_h <= hi_h:
        return [[lo_h, hi_h]]
    return [[lo_h, 179], [0, hi_h]]


def calibrate_patch(patch: np.ndarray, cursor: tuple[int, int] | None = None) -> CalibrationResult:
    """Analyse a BGR/BGRA patch; ``cursor`` is the cursor position inside it."""
    bgr = patch[..., :3]
    if not bgr.flags["C_CONTIGUOUS"]:
        bgr = np.ascontiguousarray(bgr)
    h, w = bgr.shape[:2]
    cx, cy = cursor if cursor is not None else (w // 2, h // 2)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]
    colorful = ((sat >= 50) & (val >= 60)).astype(np.uint8) * 255
    white = ((sat <= 70) & (val >= 180)).astype(np.uint8) * 255
    body = cv2.morphologyEx(cv2.bitwise_or(colorful, white), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(body, connectivity=8)
    if n <= 1:
        return CalibrationResult(False, "под курсором нет цветного объекта")
    best, best_d = -1, 1e9
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 40:
            continue
        dx = max(x - cx, 0, cx - (x + bw - 1))
        dy = max(y - cy, 0, cy - (y + bh - 1))
        d = math.hypot(dx, dy)
        if d < best_d:
            best, best_d = i, d
    if best < 0 or best_d > 40:
        return CalibrationResult(False, "наведите курсор точно на червя и повторите")
    x, y, bw, bh, _ = stats[best]
    if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
        return CalibrationResult(False, "объект под курсором слишком большой (это не червь?)")
    comp = (labels == best).astype(np.uint8) * 255
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sil = np.zeros_like(comp)
    cv2.drawContours(sil, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    area = float(cv2.countNonZero(sil))
    sel = (sil > 0) & (colorful > 0)
    hues = hsv[..., 0][sel]
    if hues.size < 20:
        return CalibrationResult(False, "слишком мало цветных пикселей для калибровки")
    ranges = _circular_range(hues)
    s_p5 = int(np.percentile(sat[sel], 5))
    msg = (f"калибровка выполнена: оттенки {ranges}, площадь ≈ {area:.0f} px")
    return CalibrationResult(True, msg, ranges, area, max(30, min(60, s_p5 - 10)))


def apply_calibration(cfg: DetectorConfig, res: CalibrationResult) -> None:
    if not res.ok or not res.hue_ranges:
        return
    cfg.hue_ranges = [list(r) for r in res.hue_ranges]
    cfg.expected_area = float(res.area)
    cfg.min_area = int(max(30, res.area * 0.3))
    cfg.max_area = int(max(cfg.min_area + 1, res.area * 3.5))
    if res.min_saturation is not None:
        cfg.min_saturation = int(res.min_saturation)
