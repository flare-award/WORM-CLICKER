"""Target finder: locks onto the small pink object that MOVES.

Idea
----
Everything static on the page (pink text, buttons, logos, even a static
picture of a worm) is ignored. The program keeps, for every pixel of a
downscaled frame, the time of its last "pink / not pink" change. A pink
blob whose pixels mostly changed recently is *moving*. The first small
moving pink blob is the worm: typically the user clicks the worm once, it
jumps, and the blob that appears at the new place is locked as the target.

After locking
  * the target is followed frame to frame by overlap (it is clicked again
    and again while it stays there - no waiting for the jump);
  * when it vanishes from its place, the new moving blob that looks like it
    (size, shape, hue - the appearance learned at lock time) is the new
    position (a "jump", counted as an approximate hit);
  * an object that does not react to clicks for a while is not the worm:
    it is ignored for a few seconds and the search continues.

Sample mode (fallback)
  The user outlines the worm on the preview (or presses F8 with the cursor
  over it). Its colours calibrate the pink segmentation and its appearance
  becomes the model; then the most similar blob is the target, whether it
  moves or not.

All work is done on a nearest-neighbour downscaled frame (pixel art keeps
its exact colours), typically 3-5 ms for a 1920x1080 frame.
"""

from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .calibration import apply_calibration, calibrate_patch
from .config import DetectorConfig, TargetConfig
from .detector import WormDetector


class TargetState(str, enum.Enum):
    WAITING = "жду движения: кликните по червю один раз"
    NEED_SAMPLE = "выделите червя мышью на превью или нажмите F8, наведя на него курсор"
    TRACKING = "цель захвачена"
    LOST = "цель пропала — ищу"
    NO_PINK = "в кадре нет розовых объектов"


@dataclass
class Blob:
    bbox: tuple[int, int, int, int]          # x, y, w, h (frame px)
    center: tuple[float, float]
    click: tuple[float, float]
    area: float                              # silhouette area, frame px
    fresh: float                             # share of recently changed pixels
    changed_now: float                       # share of pixels that changed in this very frame
    hue: float                               # circular mean hue (0..179)
    contour: np.ndarray                      # frame px, int32 (N,1,2)
    similarity: float = 0.0
    worm_score: float = 0.0


@dataclass
class Appearance:
    area: float
    hue: float
    contour: np.ndarray
    source: str                              # "motion" | "sample"
    jumps: int = 0                           # confirmed jumps since learned

    @classmethod
    def from_blob(cls, b: Blob, source: str) -> "Appearance":
        return cls(b.area, b.hue, b.contour.copy(), source)

    def similarity(self, b: Blob) -> float:
        lr = math.log(max(b.area, 1.0) / max(self.area, 1.0))
        size = math.exp(-(lr * lr) / (2 * 0.35 * 0.35))
        try:
            d = cv2.matchShapes(self.contour, b.contour, cv2.CONTOURS_MATCH_I1, 0.0)
            shape = math.exp(-3.0 * d)
        except cv2.error:
            shape = 0.5
        dh = abs(((b.hue - self.hue + 90.0) % 180.0) - 90.0)
        hue = math.exp(-((dh / 15.0) ** 2))
        return 0.5 * size + 0.3 * shape + 0.2 * hue

    def blend(self, b: Blob) -> None:
        """Slowly follow animation (motion-learned models only)."""
        self.area += 0.1 * (b.area - self.area)


@dataclass
class FindResult:
    target: Blob | None
    state: TargetState
    jumped: bool = False
    acquired: bool = False
    blobs: int = 0
    ms: float = 0.0
    frame_size: tuple[int, int] = (0, 0)
    message: str = ""

    def __post_init__(self) -> None:
        if not self.message:
            self.message = self.state.value


@dataclass
class SampleResult:
    ok: bool
    message: str
    blob: Blob | None = None
    hue_ranges: list[list[int]] | None = field(default=None)


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], grow: float = 0.0) -> int:
    ax, ay, aw, ah = a
    gx, gy = aw * grow, ah * grow
    x0, y0 = max(ax - gx, b[0]), max(ay - gy, b[1])
    x1, y1 = min(ax + aw + gx, b[0] + b[2]), min(ay + ah + gy, b[1] + b[3])
    if x1 <= x0 or y1 <= y0:
        return 0
    return int((x1 - x0) * (y1 - y0))


class TargetFinder:
    IGNORE_TTL_S = 5.0
    RELOCK_WINDOW_S = 1.0     # after a map reset the learned target may be re-found without motion

    def __init__(self, det_cfg: DetectorConfig | None = None, cfg: TargetConfig | None = None):
        self.cfg = cfg or TargetConfig()
        self.detector = WormDetector(det_cfg or DetectorConfig())
        self._k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.model: Appearance | None = None
        self.prev: Blob | None = None
        self.prev_t = -1e9
        self.last_move_t = -1e9          # last acquisition or jump
        self._ignored: list[tuple[tuple[int, int, int, int], float]] = []
        self._relock_until = -1e9
        self.reset_map()

    # ------------------------------------------------------------------ state
    @property
    def det_cfg(self) -> DetectorConfig:
        return self.detector.cfg

    def update_config(self, det_cfg: DetectorConfig, cfg: TargetConfig) -> None:
        self.detector.update_config(det_cfg)
        self.cfg = cfg

    def reset_map(self) -> None:
        """Forget the motion history (new region / window moved / resized)."""
        self._prev_pink: np.ndarray | None = None
        self._changed_t: np.ndarray | None = None
        self._t0 = None
        self._scale_used = 0
        if self.model is not None:
            self._relock_until = float("inf")     # armed on the next frame

    def reset_target(self) -> None:
        """Forget the learned target completely."""
        self.model = None
        self.prev = None
        self.prev_t = -1e9
        self._ignored.clear()
        self._relock_until = -1e9

    def mark_unresponsive(self, now: float) -> None:
        """The current target does not react to clicks: it is not the worm."""
        if self.prev is not None:
            self._ignored.append((self.prev.bbox, now + self.IGNORE_TTL_S))
        if self.model is not None and self.model.source == "motion" and self.model.jumps == 0:
            self.model = None             # learned the wrong object
        self.prev = None
        self.prev_t = -1e9

    @property
    def ignored(self) -> list[tuple[int, int, int, int]]:
        return [b for b, _ in self._ignored]

    # ------------------------------------------------------------------ blobs
    def _scale(self, w: int, h: int) -> int:
        s = self.det_cfg.downscale
        if self.det_cfg.expected_area and self.det_cfg.expected_area < 300 * s:
            s = 1
        if min(w, h) < 64:
            s = 1
        return max(1, s)

    def _blobs(self, frame: np.ndarray, now: float, update_motion: bool = True) -> list[Blob] | None:
        fh, fw = frame.shape[:2]
        s = self._scale(fw, fh)
        small = cv2.resize(frame, (fw // s, fh // s), interpolation=cv2.INTER_NEAREST) if s > 1 else frame
        if small.ndim == 3 and small.shape[2] == 4:
            small = cv2.cvtColor(small, cv2.COLOR_BGRA2BGR)
        elif not small.flags["C_CONTIGUOUS"]:
            small = np.ascontiguousarray(small)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        pink = self.detector.pink_mask(hsv)

        # ---- motion history
        if self._t0 is None:
            self._t0 = now
        tnow = np.float32(now - self._t0)
        if (self._changed_t is None or self._changed_t.shape != pink.shape or self._scale_used != s):
            self._changed_t = np.full(pink.shape, -1e6, np.float32)   # existing content = old
            self._prev_pink = pink.copy()
            self._scale_used = s
            if self._relock_until == float("inf"):
                self._relock_until = now + self.RELOCK_WINDOW_S
        elif update_motion:
            changed = cv2.compare(pink, self._prev_pink, cv2.CMP_NE)
            self._changed_t[changed > 0] = tnow
            np.copyto(self._prev_pink, pink)
        fresh_lim = tnow - np.float32(self.cfg.fresh_window_ms / 1000.0)

        if cv2.countNonZero(pink) == 0:
            return None
        joined = cv2.dilate(pink, self._k3)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(joined, connectivity=8)
        s2 = s * s
        d = self.det_cfg
        min_c = max(2.0, d.min_area * 0.35 / s2)
        max_c = min(d.max_area, self.cfg.max_target_area) * 2.0 / s2
        min_len = max(2.0, math.sqrt(d.min_area) * 0.8 / s)
        order = [i for i in range(1, n)
                 if min_c <= stats[i, cv2.CC_STAT_AREA] <= max_c
                 and max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) >= min_len]
        order.sort(key=lambda i: -stats[i, cv2.CC_STAT_AREA])
        out: list[Blob] = []
        for i in order[:48]:
            bx, by, bw, bh = (int(v) for v in stats[i, :4])
            comp = cv2.compare(labels[by:by + bh, bx:bx + bw], i, cv2.CMP_EQ)
            real = cv2.bitwise_and(comp, pink[by:by + bh, bx:bx + bw])
            n_real = cv2.countNonZero(real)
            if n_real == 0:
                continue
            ch = self._changed_t[by:by + bh, bx:bx + bw]
            fresh = float(np.count_nonzero((ch > fresh_lim) & (real > 0))) / n_real
            changed_now = float(np.count_nonzero((ch == tnow) & (real > 0))) / n_real if update_motion else 0.0
            contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            sil = np.zeros((bh + 2, bw + 2), np.uint8)
            cv2.drawContours(sil, [cnt + 1], -1, 255, cv2.FILLED)
            area = float(cv2.countNonZero(sil)) * s2
            if area < d.min_area or area > min(d.max_area, self.cfg.max_target_area):
                continue
            dist = cv2.distanceTransform(sil, cv2.DIST_L2, 3)
            _, _, _, (mx, my) = cv2.minMaxLoc(dist)
            click = ((bx + mx - 1) * s + s / 2.0, (by + my - 1) * s + s / 2.0)
            hues = hsv[by:by + bh, bx:bx + bw, 0][real > 0].astype(np.float32) * np.float32(math.pi / 90.0)
            hue = (math.degrees(math.atan2(float(np.sin(hues).mean()), float(np.cos(hues).mean()))) / 2.0) % 180.0
            gcnt = ((cnt + np.array([bx, by], np.int32)) * s + s // 2).astype(np.int32)
            cx, cy = cents[i]
            out.append(Blob((bx * s, by * s, bw * s, bh * s), (float(cx) * s + s / 2.0, float(cy) * s + s / 2.0),
                            click, area, fresh, changed_now, hue, gcnt))
        return out

    def _worm_score(self, frame: np.ndarray, b: Blob) -> float:
        """Classic worm-likeness (eyes, shades, shape) - only for ranking."""
        fh, fw = frame.shape[:2]
        x, y, w, h = b.bbox
        m = max(4, int(0.2 * max(w, h)))
        x0, y0, x1, y1 = max(0, x - m), max(0, y - m), min(fw, x + w + m), min(fh, y + h + m)
        try:
            c = self.detector._analyze(frame[y0:y1, x0:x1], x0, y0, est_area=b.area)
        except cv2.error:
            return 0.0
        return float(c.score) if c is not None else 0.0

    # ---------------------------------------------------------------- process
    def process(self, frame: np.ndarray, now: float | None = None) -> FindResult:
        t0 = time.perf_counter()
        now = time.perf_counter() if now is None else now
        fh, fw = frame.shape[:2]
        self._ignored = [(b, t) for b, t in self._ignored if t > now]
        blobs = self._blobs(frame, now)
        res = self._decide(frame, blobs or [], now, blobs is None)
        res.ms = (time.perf_counter() - t0) * 1000.0
        res.frame_size = (fw, fh)
        return res

    def _decide(self, frame: np.ndarray, blobs: list[Blob], now: float, no_pink: bool) -> FindResult:
        cfg = self.cfg
        model = self.model
        lost_s = cfg.lost_timeout_ms / 1000.0
        prev = self.prev if (self.prev is not None and now - self.prev_t <= max(lost_s, 0.05)) else None

        # 1) the target is still where it was (or moved a little): follow it
        same, best_ov = None, 0
        if prev is not None:
            for b in blobs:
                ov = _overlap(prev.bbox, b.bbox, grow=0.25)
                ratio = b.area / max(1.0, prev.area)
                if ov > best_ov and 0.4 <= ratio <= 2.5:
                    same, best_ov = b, ov

        # 2) new moving objects elsewhere
        sample = model is not None and model.source == "sample"
        if cfg.mode == "sample" and not sample:
            if same is None:
                return FindResult(None, TargetState.NEED_SAMPLE, blobs=len(blobs))
        relock = model is not None and now <= self._relock_until
        new: list[Blob] = []
        for b in blobs:
            if b is same or (prev is not None and _overlap(prev.bbox, b.bbox) > 0):
                continue
            if not sample and any(_overlap(ib, b.bbox) > 0.5 * b.bbox[2] * b.bbox[3] for ib in self.ignored):
                continue
            moving = b.fresh >= cfg.min_fresh
            if model is not None:
                b.similarity = model.similarity(b)
                if sample:
                    ok = b.similarity >= cfg.sample_similarity
                else:
                    ok = (moving and b.similarity >= cfg.min_similarity) or (relock and b.similarity >= 0.75)
            else:
                ok = moving
            if ok:
                new.append(b)

        target: Blob | None = None
        jumped = acquired = False
        if same is not None:
            target = same
            dx, dy = same.center[0] - prev.center[0], same.center[1] - prev.center[1]
            moved = math.hypot(dx, dy) > 0.5 * max(prev.bbox[2], prev.bbox[3]) and same.fresh >= cfg.min_fresh
            if moved or same.changed_now >= 0.6:
                jumped = True                  # short jump overlapping the old place
            if model is not None:
                same.similarity = model.similarity(same)
        elif new:
            if model is not None:
                target = max(new, key=lambda b: b.similarity + 0.2 * b.fresh)
            elif len(new) == 1:
                target = new[0]
            else:
                for b in new:
                    b.worm_score = self._worm_score(frame, b)
                target = max(new, key=lambda b: (b.worm_score, b.fresh, -b.area))
            acquired = True
            jumped = self.prev is not None
            if self.model is None:
                self.model = Appearance.from_blob(target, "motion")
                target.similarity = 1.0

        if target is not None:
            if self.model is not None:
                if jumped:
                    self.model.jumps += 1
                if self.model.source == "motion":
                    self.model.blend(target)
            if jumped or acquired:
                self.last_move_t = now
                self._relock_until = -1e9
            self.prev, self.prev_t = target, now
            return FindResult(target, TargetState.TRACKING, jumped, acquired, len(blobs))
        if no_pink:
            return FindResult(None, TargetState.NO_PINK)
        if self.model is None:
            st = TargetState.NEED_SAMPLE if cfg.mode == "sample" else TargetState.WAITING
            return FindResult(None, st, blobs=len(blobs))
        return FindResult(None, TargetState.LOST, blobs=len(blobs))

    # ----------------------------------------------------------------- sample
    def learn_sample(self, frame: np.ndarray, box: tuple[int, int, int, int] | None = None,
                     point: tuple[int, int] | None = None, now: float | None = None) -> SampleResult:
        """Learn the target from a user-selected box (or the point under the cursor).

        Coordinates are frame pixels. The colours of the object calibrate
        the pink segmentation; its appearance becomes the target model.
        """
        now = time.perf_counter() if now is None else now
        fh, fw = frame.shape[:2]
        if box is not None:
            x, y, w, h = (int(round(v)) for v in box)
            cx, cy = x + w / 2.0, y + h / 2.0
            half = int(max(w, h) * 0.75) + 24
        elif point is not None:
            cx, cy = float(point[0]), float(point[1])
            half = 160
            x, y, w, h = int(cx) - 2, int(cy) - 2, 5, 5
        else:
            return SampleResult(False, "не задана область")
        if not (0 <= cx < fw and 0 <= cy < fh):
            return SampleResult(False, "точка вне области захвата")
        px0, py0 = max(0, int(cx) - half), max(0, int(cy) - half)
        px1, py1 = min(fw, int(cx) + half), min(fh, int(cy) + half)
        calib = calibrate_patch(frame[py0:py1, px0:px1], (int(cx) - px0, int(cy) - py0))
        if calib.ok:
            apply_calibration(self.det_cfg, calib)
            self.det_cfg.max_area = max(self.det_cfg.max_area, int(calib.area * 3.5))
            self.detector.update_config(self.det_cfg)
        blobs = self._blobs(frame, now, update_motion=False) or []
        sel = (x, y, max(1, w), max(1, h))
        best, best_key = None, None
        for b in blobs:
            ov = _overlap(sel, b.bbox)
            inside = b.bbox[0] <= cx <= b.bbox[0] + b.bbox[2] and b.bbox[1] <= cy <= b.bbox[1] + b.bbox[3]
            dist = math.hypot(b.center[0] - cx, b.center[1] - cy)
            if ov <= 0 and not inside and dist > 40:
                continue
            key = (ov, inside, -dist)
            if best_key is None or key > best_key:
                best, best_key = b, key
        if best is None:
            return SampleResult(False, calib.message if not calib.ok else "в выделенной области нет розового объекта")
        self.model = Appearance.from_blob(best, "sample")
        self.cfg.mode = "sample"
        self._ignored.clear()
        self.prev, self.prev_t = best, now
        self.last_move_t = now
        best.similarity = 1.0
        msg = f"образец запомнен: ≈ {best.area:.0f} px"
        return SampleResult(True, msg, best, calib.hue_ranges if calib.ok else None)
