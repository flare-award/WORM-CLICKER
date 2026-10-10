"""Procedural reference sprite of the pink pixel worm.

The original game screenshot is not part of the repository, so this module
contains a hand-made pixel-art analogue built from the visual description:
pink body with several shades (pink, light pink, dark pink, maroon outline)
and two light rectangular eyes with dark pupils.

The sprite is used for:
  * the rotation/scale invariant shape reference of the detector (Hu moments);
  * synthetic test frames and the built-in simulator;
  * the HTML practice target (``practice/index.html`` contains the same grid).
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

# 22 x 8 pixel grid, head on the right side.
#   D maroon outline, M dark pink, P pink, L light pink,
#   W eye white, K pupil, '.' transparent
SPRITE_GRID: tuple[str, ...] = (
    "..............DDDDDD..",
    ".............DWWPWWPD.",
    "..DDDDDDDDDDDPWKPWKPD.",
    ".DLLLLLLLLLLLPWKPWKPPD",
    "DPPPPPPPPPPPPPPPPPPPPD",
    "DPMPPMPPMPPMPPPPPPLPPD",
    ".DMMMMMMMMMMMMMMMMMMD.",
    "..DDDDDDDDDDDDDDDDDD..",
)

# RGB colours of the palette.
PALETTE_RGB: dict[str, tuple[int, int, int]] = {
    "D": (0x7D, 0x1F, 0x4B),
    "M": (0xD9, 0x4A, 0x8C),
    "P": (0xFF, 0x6F, 0xAE),
    "L": (0xFF, 0xB8, 0xD8),
    "W": (0xF2, 0xF0, 0xF5),
    "K": (0x1A, 0x10, 0x20),
}

BACKGROUND_BGR: tuple[int, int, int] = (0x14, 0x0E, 0x0D)  # almost black

GRID_W = len(SPRITE_GRID[0])
GRID_H = len(SPRITE_GRID)


@lru_cache(maxsize=1)
def sprite_rgba_grid() -> np.ndarray:
    """Return the sprite as a (GRID_H, GRID_W, 4) BGRA uint8 array (1 px per cell)."""
    out = np.zeros((GRID_H, GRID_W, 4), np.uint8)
    for y, row in enumerate(SPRITE_GRID):
        assert len(row) == GRID_W, "sprite rows must have equal width"
        for x, ch in enumerate(row):
            if ch == ".":
                continue
            r, g, b = PALETTE_RGB[ch]
            out[y, x] = (b, g, r, 255)
    out.setflags(write=False)
    return out


def sprite_bgra(pixel_size: int = 4, wiggle: float = 0.0) -> np.ndarray:
    """Sprite scaled by an integer pixel size.

    ``wiggle`` (radians phase, 0 = no deformation) shifts the tail columns up
    and down by one sprite pixel to emulate a crawling animation.
    """
    grid = np.array(sprite_rgba_grid())
    if wiggle:
        pad = np.zeros((GRID_H + 2, GRID_W, 4), np.uint8)
        pad[1:-1] = grid
        shifted = np.zeros_like(pad)
        for x in range(GRID_W):
            # tail (left part) moves, head stays still
            amp = 1 if x < 13 else 0
            dy = int(round(amp * np.sin(wiggle + x * 0.55)))
            shifted[:, x] = np.roll(pad[:, x], dy, axis=0)
        grid = shifted
    ps = max(1, int(pixel_size))
    return np.repeat(np.repeat(grid, ps, axis=0), ps, axis=1)


def rotate_bgra(img: np.ndarray, angle_deg: float, smooth: bool = False) -> np.ndarray:
    """Rotate a BGRA image around its centre on an enlarged transparent canvas."""
    h, w = img.shape[:2]
    diag = int(np.ceil(np.hypot(w, h))) + 2
    canvas = np.zeros((diag, diag, 4), np.uint8)
    oy, ox = (diag - h) // 2, (diag - w) // 2
    canvas[oy:oy + h, ox:ox + w] = img
    m = cv2.getRotationMatrix2D((diag / 2.0, diag / 2.0), angle_deg, 1.0)
    interp = cv2.INTER_LINEAR if smooth else cv2.INTER_NEAREST
    return cv2.warpAffine(canvas, m, (diag, diag), flags=interp,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def render_worm(
    frame: np.ndarray,
    center: tuple[float, float],
    angle_deg: float = 0.0,
    pixel_size: int = 4,
    smooth: bool = False,
    wiggle: float = 0.0,
) -> np.ndarray:
    """Alpha-blend the worm onto a BGR frame in place.

    Returns a boolean mask (frame sized) of the pixels covered by the worm,
    which is used as the ground-truth hit box in tests and the simulator.
    """
    spr = rotate_bgra(sprite_bgra(pixel_size, wiggle), angle_deg, smooth)
    sh, sw = spr.shape[:2]
    fh, fw = frame.shape[:2]
    x0 = int(round(center[0] - sw / 2.0))
    y0 = int(round(center[1] - sh / 2.0))
    mask = np.zeros((fh, fw), bool)
    fx0, fy0 = max(0, x0), max(0, y0)
    fx1, fy1 = min(fw, x0 + sw), min(fh, y0 + sh)
    if fx0 >= fx1 or fy0 >= fy1:
        return mask
    sub = spr[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    alpha = sub[..., 3:4].astype(np.float32) / 255.0
    dst = frame[fy0:fy1, fx0:fx1, :3].astype(np.float32)
    frame[fy0:fy1, fx0:fx1, :3] = (sub[..., :3] * alpha + dst * (1 - alpha)).astype(np.uint8)
    mask[fy0:fy1, fx0:fx1] = sub[..., 3] >= 128
    return mask


def worm_extent(pixel_size: int, angle_deg: float = 0.0) -> tuple[int, int]:
    """Size (w, h) of the rotated sprite canvas."""
    w, h = GRID_W * pixel_size, GRID_H * pixel_size
    diag = int(np.ceil(np.hypot(w, h))) + 2
    return diag, diag


@lru_cache(maxsize=1)
def reference_contour() -> np.ndarray:
    """Outer contour of the closed sprite silhouette, used for Hu-moment matching."""
    spr = sprite_bgra(8)
    mask = (spr[..., 3] > 0).astype(np.uint8) * 255
    mask = cv2.copyMakeBorder(mask, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea)


def make_background(width: int, height: int, noise: int = 3, seed: int = 0) -> np.ndarray:
    """Dark game-like background with light noise."""
    rng = np.random.default_rng(seed)
    bg = np.empty((height, width, 3), np.uint8)
    bg[:] = BACKGROUND_BGR
    if noise:
        n = rng.integers(-noise, noise + 1, size=(height, width, 1), dtype=np.int16)
        bg = np.clip(bg.astype(np.int16) + n, 0, 255).astype(np.uint8)
    return bg
