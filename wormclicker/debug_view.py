"""Preview rendering (runs in the UI thread, never in the click loop).

Only the target is drawn: its outline, corner brackets and the click point.
Text is drawn by Tk on top of the image (OpenCV can not render Cyrillic -
that is where the "???" labels of the old debug window came from).
"""

from __future__ import annotations

import cv2
import numpy as np

from .automation import DebugFrame

TARGET = (162, 95, 255)       # BGR, pink accent
OUTLINE = (255, 255, 255)
CLICK = (120, 255, 140)


def render(df: DebugFrame) -> np.ndarray | None:
    """Return an RGB image with the target overlay, or None when there is no picture."""
    img = df.image
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tg = df.target
    if tg is None:
        return rgb
    s = df.scale
    tgt = TARGET[::-1]            # drawing on RGB
    cnt = np.round(tg.contour.astype(np.float32) * s).astype(np.int32)
    cv2.drawContours(rgb, [cnt], -1, tgt, 2, cv2.LINE_AA)
    x, y, w, h = (v * s for v in tg.bbox)
    pad = 6
    x0, y0, x1, y1 = int(x - pad), int(y - pad), int(x + w + pad), int(y + h + pad)
    arm = max(6, int(min(x1 - x0, y1 - y0) * 0.28))
    for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(rgb, (cx, cy), (cx + dx * arm, cy), OUTLINE, 2, cv2.LINE_AA)
        cv2.line(rgb, (cx, cy), (cx, cy + dy * arm), OUTLINE, 2, cv2.LINE_AA)
    px, py = int(round(tg.click[0] * s)), int(round(tg.click[1] * s))
    cv2.circle(rgb, (px, py), 5, CLICK[::-1], 2, cv2.LINE_AA)
    cv2.line(rgb, (px - 10, py), (px - 6, py), CLICK[::-1], 2, cv2.LINE_AA)
    cv2.line(rgb, (px + 6, py), (px + 10, py), CLICK[::-1], 2, cv2.LINE_AA)
    cv2.line(rgb, (px, py - 10), (px, py - 6), CLICK[::-1], 2, cv2.LINE_AA)
    cv2.line(rgb, (px, py + 6), (px, py + 10), CLICK[::-1], 2, cv2.LINE_AA)
    return rgb


def to_ppm(rgb: np.ndarray) -> bytes:
    """Binary PPM (P6) - Tk's PhotoImage parses it much faster than PNG."""
    h, w = rgb.shape[:2]
    if not rgb.flags["C_CONTIGUOUS"]:
        rgb = np.ascontiguousarray(rgb)
    return b"P6 %d %d 255\n" % (w, h) + rgb.tobytes()


def target_box_on_preview(df: DebugFrame) -> tuple[float, float, float, float] | None:
    """Target bbox in preview pixels (for the Tk label next to it)."""
    if df.target is None or df.image is None:
        return None
    x, y, w, h = df.target.bbox
    s = df.scale
    return x * s, y * s, w * s, h * s
