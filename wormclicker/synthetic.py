"""Synthetic game frames: worm + typical distractors.

Used by the unit tests, the benchmark and the simulator. No real game
frames are used anywhere in the automated checks.
"""

from __future__ import annotations

import cv2
import numpy as np

from .sprite import make_background, render_worm

PINK_UI = (174, 111, 255)        # BGR of a uniform pink UI colour
LIGHT_PINK_UI = (216, 184, 255)


def add_white_text(frame: np.ndarray, text: str = "SCORE: 12345", org=(20, 50), scale=1.2) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 2, cv2.LINE_AA)


def add_pink_text(frame: np.ndarray, text: str = "LEADERBOARD", org=(20, 120), scale=1.4) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, PINK_UI, 3, cv2.LINE_AA)


def add_pink_button(frame: np.ndarray, x: int = 40, y: int = 160, w: int = 130, h: int = 44,
                    text: str = "PLAY") -> None:
    cv2.rectangle(frame, (x, y), (x + w, y + h), PINK_UI, -1)
    cv2.putText(frame, text, (x + 18, y + h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (250, 250, 250), 2,
                cv2.LINE_AA)


def add_particles(frame: np.ndarray, n: int = 60, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    h, w = frame.shape[:2]
    for _ in range(n):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(1, 4))
        col = PINK_UI if rng.random() < 0.6 else (255, 255, 255)
        cv2.circle(frame, (x, y), r, col, -1)


def add_glow(frame: np.ndarray, center=(600, 120), radius: int = 60) -> None:
    """Soft pink radial glow (smooth gradient, no pixel-art palette)."""
    h, w = frame.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    d = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / radius
    a = np.clip(1.0 - d, 0, 1)[..., None]
    col = np.array(PINK_UI, np.float32)
    frame[:] = np.clip(frame * (1 - a) + col * a, 0, 255).astype(np.uint8)


def add_pink_heart_icon(frame: np.ndarray, x: int = 700, y: int = 40, s: int = 4) -> None:
    pattern = [".PP.PP.", "PPPPPPP", "PPPPPPP", ".PPPPP.", "..PPP..", "...P..."]
    for j, row in enumerate(pattern):
        for i, ch in enumerate(row):
            if ch == "P":
                cv2.rectangle(frame, (x + i * s, y + j * s), (x + i * s + s - 1, y + j * s + s - 1),
                              PINK_UI, -1)


def scene_with_distractors(width: int = 1280, height: int = 720, seed: int = 0) -> np.ndarray:
    f = make_background(width, height, seed=seed)
    add_white_text(f)
    add_pink_text(f)
    add_pink_button(f)
    add_particles(f, seed=seed + 1)
    add_glow(f, center=(int(width * 0.6), int(height * 0.18)))
    add_pink_heart_icon(f, x=int(width * 0.8), y=30)
    return f


def scene(width=1280, height=720, worm_center=(640, 400), angle=0.0, pixel_size=4,
          smooth=False, distractors=True, seed=0, wiggle=0.0):
    """Return (frame, ground-truth worm mask)."""
    f = scene_with_distractors(width, height, seed) if distractors else make_background(width, height, seed=seed)
    mask = render_worm(f, worm_center, angle, pixel_size, smooth, wiggle)
    return f, mask
