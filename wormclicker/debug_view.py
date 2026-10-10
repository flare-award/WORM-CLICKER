"""Annotated debug image (runs in the UI thread, never in the click loop)."""

from __future__ import annotations

import cv2
import numpy as np

from .automation import DebugFrame

GREEN = (80, 230, 80)
YELLOW = (40, 210, 240)
GRAY = (150, 150, 150)
RED = (70, 70, 240)
CYAN = (230, 200, 60)


def annotate(df: DebugFrame, max_w: int = 720, max_h: int = 460) -> tuple[np.ndarray, list[str]]:
    """Return (BGR image scaled to fit, text lines describing candidates)."""
    frame = df.frame
    img = frame[..., :3].copy() if frame.shape[2] == 4 else frame.copy()
    h, w = img.shape[:2]
    scale = min(1.0, max_w / w, max_h / h)
    lines: list[str] = []
    det = df.detection
    th = max(1, int(round(1.5 / max(scale, 0.25))))
    font = 0.45 / max(scale, 0.35)
    for i, c in enumerate(det.candidates):
        x, y, bw, bh = c.bbox
        best = det.best is c
        color = GREEN if best else (YELLOW if c.accepted else GRAY)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), color, th)
        if c.contour is not None and (best or c.accepted):
            cv2.drawContours(img, [c.contour], -1, CYAN, th)
        label = f"{c.score:.2f}" if c.accepted else (c.reject or "")[:28]
        cv2.putText(img, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, font, color, th, cv2.LINE_AA)
        if i < 8:
            f = c.features
            extra = (f" глаза={f.get('eyes', '-')} оттенки={f.get('shades', '-')} "
                     f"aspect={f.get('aspect', '-')} hu={f.get('hu', '-')}") if f else ""
            lines.append(f"{'★' if best else '•'} ({x},{y}) {c.area}px  "
                         f"{'OK ' + format(c.score, '.2f') if c.accepted else c.reject}{extra}")
    if det.best is not None:
        cx, cy = int(round(det.best.click[0])), int(round(det.best.click[1]))
        r = max(6, int(8 / max(scale, 0.25)))
        cv2.circle(img, (cx, cy), r, RED, th)
        cv2.line(img, (cx - 2 * r, cy), (cx + 2 * r, cy), RED, th)
        cv2.line(img, (cx, cy - 2 * r), (cx, cy + 2 * r), RED, th)
    for bx, by, br in df.blacklist:
        fx, fy = int(bx - df.rect.left), int(by - df.rect.top)
        cv2.circle(img, (fx, fy), int(br), RED, th)
        cv2.line(img, (fx - int(br), fy - int(br)), (fx + int(br), fy + int(br)), RED, th)
    if scale < 1.0:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    head = f"кадр {w}x{h}  обработка {det.ms:.1f} мс  кандидатов {len(det.candidates)}"
    if det.tiny_rejected:
        head += f"  мелких точек {det.tiny_rejected}"
    if df.decision is not None:
        head += f"  | {df.decision.reason}"
    if det.best is None:
        head += f"  | {det.reason}"
    return img, [head] + lines


def to_png_base64(img: np.ndarray) -> str:
    import base64

    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise ValueError("png encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")
