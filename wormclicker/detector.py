"""Classical computer-vision detector of the pink pixel worm.

Pipeline (rotation and scale invariant, no single-template matching):

1. Coarse stage on a nearest-neighbour downscaled frame: HSV colour
   segmentation of "worm pink" (hue LUT handles wrap-around), small dilation,
   connected components -> list of candidate regions. Pixel art keeps exact
   colours under nearest sampling, so the coarse pass is cheap and lossless
   for colour classification.
2. Fine stage on the full-resolution ROI of every candidate:
   * silhouette = pink | eye-white, morphologically closed (fills pupils);
   * colour evidence: share of pink pixels, presence of several pink shades
     (light / mid / dark), compact pixel-art palette;
   * eyes: light blobs inside the silhouette that contain dark pupils;
   * rotation invariant geometry: minAreaRect aspect ratio, extent,
     Hu-moment distance to the reference silhouette (cv2.matchShapes);
   * optional size prior from calibration.
   The features are combined into a confidence score in [0, 1].
3. Click point: the interior point of the silhouette (distance transform
   maximum closest to the centroid) - stays inside the body for any rotation
   and for curved animation frames.

Every rejected candidate keeps a human readable reason for the debug view.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import DetectorConfig
from .sprite import SPRITE_GRID, reference_contour

SPRITE_CELLS = sum(1 for row in SPRITE_GRID for ch in row if ch != ".")


@dataclass
class Candidate:
    bbox: tuple[int, int, int, int]              # x, y, w, h (frame coords)
    center: tuple[float, float]
    click: tuple[float, float]
    area: int
    angle: float
    length: float                                # major axis, px
    width: float                                 # minor axis, px
    score: float
    features: dict[str, float] = field(default_factory=dict)
    reject: str | None = None
    contour: np.ndarray | None = None            # frame coords, for debug drawing

    @property
    def accepted(self) -> bool:
        return self.reject is None


@dataclass
class Detection:
    candidates: list[Candidate]
    best: Candidate | None
    reason: str
    ms: float
    frame_size: tuple[int, int]                  # w, h
    tiny_rejected: int = 0

    @property
    def accepted(self) -> list[Candidate]:
        return [c for c in self.candidates if c.accepted]


def _trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
    """1 inside [b, c], linear ramps to 0 at a and d."""
    if x <= a or x >= d:
        return 0.0
    if x < b:
        return (x - a) / (b - a)
    if x <= c:
        return 1.0
    return (d - x) / (d - c)


class WormDetector:
    def __init__(self, cfg: DetectorConfig | None = None):
        self.cfg = cfg or DetectorConfig()
        self._ref_contour = reference_contour()
        self._k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._kernels: dict[int, np.ndarray] = {}
        self.update_config(self.cfg)

    # ------------------------------------------------------------------ setup
    def update_config(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        self._reject_cache: dict[tuple, Candidate] = {}
        lut = np.zeros(256, np.uint8)
        for lo, hi in cfg.hue_ranges:
            if lo <= hi:
                lut[lo:hi + 1] = 255
            else:  # wrapped range, e.g. [170, 5]
                lut[lo:180] = 255
                lut[0:hi + 1] = 255
        self._hue_lut = lut

    def _kernel(self, k: int) -> np.ndarray:
        ker = self._kernels.get(k)
        if ker is None:
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            self._kernels[k] = ker
        return ker

    # ----------------------------------------------------------------- masks
    def pink_mask(self, hsv: np.ndarray) -> np.ndarray:
        c = self.cfg
        h, s, v = cv2.split(hsv)
        hue_ok = cv2.LUT(h, self._hue_lut)
        sat = cv2.threshold(s, c.min_saturation - 1, 255, cv2.THRESH_BINARY)[1]
        val = cv2.threshold(v, c.min_value - 1, 255, cv2.THRESH_BINARY)[1]
        strong = cv2.bitwise_and(sat, val)
        lsat = cv2.threshold(s, c.light_min_saturation - 1, 255, cv2.THRESH_BINARY)[1]
        lval = cv2.threshold(v, c.light_min_value - 1, 255, cv2.THRESH_BINARY)[1]
        light = cv2.bitwise_and(lsat, lval)
        return cv2.bitwise_and(hue_ok, cv2.bitwise_or(strong, light))

    # ----------------------------------------------------------------- detect
    def detect(self, frame: np.ndarray, hint: tuple[int, int, int, int] | None = None) -> Detection:
        """Detect the worm in a BGR or BGRA frame.

        ``hint`` (x, y, w, h) restricts the search to a region first (short
        term tracking optimisation); when nothing is accepted there, the full
        frame is scanned.
        """
        t0 = time.perf_counter()
        fh, fw = frame.shape[:2]
        if hint is not None:
            x, y, w, h = hint
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(fw, int(x + w)), min(fh, int(y + h))
            if x1 - x0 >= 16 and y1 - y0 >= 16:
                det = self._detect_region(frame, (x0, y0, x1, y1))
                if det.best is not None:
                    det.ms = (time.perf_counter() - t0) * 1000
                    return det
        det = self._detect_region(frame, (0, 0, fw, fh))
        det.ms = (time.perf_counter() - t0) * 1000
        return det

    def _detect_region(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> Detection:
        c = self.cfg
        x0, y0, x1, y1 = box
        fh, fw = frame.shape[:2]
        view = frame[y0:y1, x0:x1]
        h, w = view.shape[:2]
        s = c.downscale
        # tiny worms need full resolution in the coarse pass
        if c.expected_area and c.expected_area < 300 * s:
            s = 1
        if min(w, h) < 64:
            s = 1
        if s > 1:
            small = cv2.resize(view, (w // s, h // s), interpolation=cv2.INTER_NEAREST)
        else:
            small = view
        if small.shape[2] == 4:
            small = cv2.cvtColor(small, cv2.COLOR_BGRA2BGR)
        elif not small.flags["C_CONTIGUOUS"]:
            small = np.ascontiguousarray(small)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        pink = self.pink_mask(hsv)
        if cv2.countNonZero(pink) == 0:
            return Detection([], None, "нет розовых пикселей в кадре", 0.0, (fw, fh))
        joined = cv2.dilate(pink, self._k3)
        n, _labels, stats, _cent = cv2.connectedComponentsWithStats(joined, connectivity=8)
        s2 = s * s
        min_coarse = max(2, c.min_area * 0.35 / s2)
        max_coarse = c.max_area * 3.0 / s2
        min_len = math.sqrt(c.min_area) * 1.2 / s     # shortest plausible worm, coarse px
        comps = []
        tiny = 0
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_coarse or max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) < min_len:
                tiny += 1
                continue
            comps.append((area, i))
        cands: list[Candidate] = []
        new_cache: dict[tuple, Candidate] = {}
        comps.sort(reverse=True)
        for area, i in comps[: c.max_candidates]:
            bx, by, bw, bh = (int(v) for v in stats[i, :4])
            gx, gy, gw, gh = bx * s + x0, by * s + y0, bw * s, bh * s
            if area > max_coarse:
                cands.append(Candidate((gx, gy, gw, gh), (gx + gw / 2, gy + gh / 2),
                                       (gx + gw / 2, gy + gh / 2), int(area * s2), 0.0,
                                       float(max(gw, gh)), float(min(gw, gh)), 0.0,
                                       reject="слишком большая область"))
                continue
            # static page elements (text, buttons) are rejected every frame:
            # reuse the verdict while their coarse footprint is unchanged
            key = (gx, gy, gw, gh, int(area), s)
            cached = self._reject_cache.get(key)
            if cached is not None:
                new_cache[key] = cached
                cands.append(cached)
                continue
            margin = max(4, int(0.2 * max(gw, gh))) + s
            rx0, ry0 = max(0, gx - margin), max(0, gy - margin)
            rx1, ry1 = min(fw, gx + gw + margin), min(fh, gy + gh + margin)
            cand = self._analyze(frame[ry0:ry1, rx0:rx1], rx0, ry0, est_area=area * s2)
            if cand is not None:
                cands.append(cand)
                if not cand.accepted:
                    new_cache[key] = cand
        if box == (0, 0, fw, fh):
            self._reject_cache = new_cache
        cands.sort(key=lambda cd: (cd.accepted, cd.score), reverse=True)
        best = cands[0] if cands and cands[0].accepted else None
        if best is not None:
            reason = "ok"
        elif cands:
            reason = f"кандидаты отклонены: {cands[0].reject}"
        elif tiny:
            reason = f"только мелкие розовые точки ({tiny})"
        else:
            reason = "подходящих объектов нет"
        return Detection(cands, best, reason, 0.0, (fw, fh), tiny)

    # ---------------------------------------------------------------- analyze
    def _analyze(self, roi: np.ndarray, ox: int, oy: int, est_area: float) -> Candidate | None:
        c = self.cfg
        if roi.size == 0:
            return None
        bgr = roi[..., :3]
        if not bgr.flags["C_CONTIGUOUS"]:
            bgr = np.ascontiguousarray(bgr)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        _h, sat, val = cv2.split(hsv)
        pink = self.pink_mask(hsv)
        white = cv2.bitwise_and(
            cv2.threshold(sat, c.eye_max_saturation, 255, cv2.THRESH_BINARY_INV)[1],
            cv2.threshold(val, c.eye_min_value - 1, 255, cv2.THRESH_BINARY)[1])
        dark = cv2.threshold(val, c.pupil_max_value, 255, cv2.THRESH_BINARY_INV)[1]

        k = int(round(math.sqrt(max(est_area, 1.0)) / 10.0))
        k = max(3, min(11, k | 1))
        body = cv2.bitwise_or(pink, white)
        closed = cv2.morphologyEx(body, cv2.MORPH_CLOSE, self._kernel(k))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(closed, connectivity=8)
        if n <= 1:
            return None
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = int(stats[idx, cv2.CC_STAT_AREA])
        bx, by, bw, bh = (int(v) for v in stats[idx, :4])
        gb = (bx + ox, by + oy, bw, bh)
        center = (float(cents[idx][0]) + ox, float(cents[idx][1]) + oy)

        def reject(reason: str, score: float = 0.0, **kw) -> Candidate:
            return Candidate(gb, center, center, area, kw.get("angle", 0.0),
                             kw.get("length", float(max(bw, bh))), kw.get("width", float(min(bw, bh))),
                             score, kw.get("features", {}), reject=reason, contour=kw.get("contour"))

        if area < c.min_area:
            return reject(f"слишком мал ({area} px)")
        if area > c.max_area:
            return reject(f"слишком велик ({area} px)")

        comp = cv2.compare(labels, idx, cv2.CMP_EQ)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)
        # filled silhouette: pupils and other interior holes belong to the worm
        sil = np.zeros_like(comp)
        cv2.drawContours(sil, [cnt], -1, 255, thickness=cv2.FILLED)
        area = max(area, cv2.countNonZero(sil))
        pink_in = cv2.bitwise_and(pink, sil)
        n_pink = cv2.countNonZero(pink_in)
        pink_frac = n_pink / area
        (rcx, rcy), (rw, rh), ang = cv2.minAreaRect(cnt)
        major, minor = max(rw, rh), max(1.0, min(rw, rh))
        if rw < rh:
            ang += 90.0
        aspect = major / minor
        extent = area / max(1.0, rw * rh)
        gcnt = cnt + np.array([ox, oy], dtype=cnt.dtype)
        geo = dict(angle=float(ang % 180.0), length=float(major), width=float(minor), contour=gcnt)

        if pink_frac < 0.3:
            return reject(f"мало розового ({pink_frac:.0%})", features={"pink_frac": pink_frac}, **geo)
        if minor < 3:
            return reject("слишком тонкий объект", **geo)

        # --- colour evidence --------------------------------------------------
        sat_p = sat[pink_in > 0]
        val_p = val[pink_in > 0]
        light = np.count_nonzero((sat_p < 120) & (val_p >= 180))
        darkp = np.count_nonzero(val_p < 160)
        mid = n_pink - light - darkp
        classes = sum(1 for cnt_ in (light, darkp, mid) if cnt_ >= 0.04 * n_pink)
        diversity = (classes - 1) / 2.0
        q = (bgr[sil > 0] >> 4).astype(np.int32)
        codes = (q[:, 0] << 8) | (q[:, 1] << 4) | q[:, 2]
        hist = np.bincount(codes, minlength=4096)
        top = np.sort(hist)[-8:].sum()
        palette = float(top) / max(1, codes.size)
        color = min(1.0, pink_frac / 0.55) * (0.3 + 0.45 * diversity + 0.25 * palette)

        # --- eyes ---------------------------------------------------------------
        white_in = cv2.bitwise_and(white, sil)
        dark_in = cv2.bitwise_and(dark, sil)
        # depth of every silhouette pixel: pupils lie inside the head, while
        # background pulled in by the closing is near the silhouette border
        dist = cv2.distanceTransform(sil, cv2.DIST_L2, 3)
        eyes_with_pupil = 0
        white_blobs = 0
        eye_pts: list[tuple[float, float, float]] = []   # x, y, white area
        en, elabels, estats, ecents = cv2.connectedComponentsWithStats(white_in, connectivity=8)
        rh_, rw_ = sil.shape[:2]
        for j in range(1, en):
            ea = estats[j, cv2.CC_STAT_AREA]
            if ea < 3 or ea > 0.25 * area:
                continue
            white_blobs += 1
            if self._is_eye(j, estats[j], elabels, dark_in, pink, dist):
                eyes_with_pupil += 1
                eye_pts.append((float(ecents[j][0]), float(ecents[j][1]), float(ea)))
        paired = False
        pair_anchor = None
        if len(eye_pts) >= 2:
            srt = sorted(eye_pts, key=lambda e: -e[2])
            for a_i in range(len(srt)):
                for b_i in range(a_i + 1, len(srt)):
                    ea_, eb_ = srt[a_i], srt[b_i]
                    e_ps = math.sqrt(ea_[2] / 4.0)
                    if ea_[2] <= 2.5 * eb_[2] and math.hypot(ea_[0] - eb_[0], ea_[1] - eb_[1]) < 6 * e_ps:
                        if not paired:
                            pair_anchor = ((ea_[0] + eb_[0]) / 2, (ea_[1] + eb_[1]) / 2, e_ps)
                        paired = True
        if paired:
            eyes = 1.0
        elif eyes_with_pupil >= 1:
            eyes = 0.4
        elif white_blobs:
            eyes = 0.1
        else:
            eyes = 0.0
        if eyes_with_pupil > 3:  # many "eyes": dotted texture / text, not a worm
            eyes *= 0.4
        anchor = pair_anchor
        if anchor is None and eye_pts:
            # keep the two largest eyes that are close to each other
            eye_pts.sort(key=lambda e: -e[2])
            e0 = eye_pts[0]
            eps = math.sqrt(e0[2] / 4.0)            # eye = 4 white sprite px
            pair = [e0] + [e for e in eye_pts[1:2] if math.hypot(e[0] - e0[0], e[1] - e0[1]) < 6 * eps]
            anchor = (sum(e[0] for e in pair) / len(pair), sum(e[1] for e in pair) / len(pair), eps)

        # --- geometry ---------------------------------------------------------
        aspect_s = _trapezoid(aspect, 1.3, 1.9, 5.5, 9.0)
        extent_s = _trapezoid(extent, 0.35, 0.5, 0.9, 0.985)
        hu_d = cv2.matchShapes(cnt, self._ref_contour, cv2.CONTOURS_MATCH_I1, 0.0)
        hu_s = math.exp(-hu_d * 4.0)
        shape = 0.4 * aspect_s + 0.3 * hu_s + 0.3 * extent_s

        if c.expected_area > 0:
            lr = math.log(area / c.expected_area)
            size_s = math.exp(-(lr * lr) / (2 * 0.6 * 0.6))
        else:
            size_s = 1.0

        score = 0.27 * color + 0.35 * eyes + 0.28 * shape + 0.10 * size_s
        feats = {
            "pink_frac": round(pink_frac, 3), "shades": classes, "palette": round(palette, 3),
            "eyes": eyes_with_pupil, "white_blobs": white_blobs, "aspect": round(aspect, 2),
            "extent": round(extent, 3), "hu": round(hu_d, 3), "color": round(color, 3),
            "shape": round(shape, 3), "size": round(size_s, 3),
        }
        geo["features"] = feats
        if score < c.confidence_threshold:
            return reject(f"низкая уверенность {score:.2f}", score=score, **geo)

        # --- click point --------------------------------------------------------
        # depth inside the real (unfilled) body: never click into enclosed background
        dist = cv2.distanceTransform(cv2.bitwise_and(comp, sil), cv2.DIST_L2, 3)
        maxd = float(dist.max())
        mx, my = center[0] - ox, center[1] - oy
        min_depth = 0.8 * maxd
        merged = False
        if anchor is not None:
            ax, ay, eps = anchor
            worm_len = 22.0 * eps
            # silhouette glued to other pink objects (text, glow, buttons)?
            # silhouette much larger than a worm with eyes of this size
            too_big = area > 1.6 * SPRITE_CELLS * eps * eps
            merged = (too_big or hu_d > 1.0 or extent < 0.5 or aspect < 1.5 or aspect > 7.0
                      or math.hypot(mx - ax, my - ay) > 0.6 * worm_len)
            if merged:
                mx, my = ax, ay          # click between the eyes: always on the head
                min_depth = 0.0
            else:
                mx, my = 0.5 * (mx + ax), 0.5 * (my + ay)   # between head and body centre
                min_depth = min(min_depth, 1.6 * eps)
        feats["merged"] = int(merged)
        if merged:
            body_px = cv2.bitwise_or(pink_in, white_in)
            ys, xs = np.nonzero(body_px)
        else:
            ys, xs = np.nonzero(dist >= min_depth)
        j = int(np.argmin((xs - mx) ** 2 + (ys - my) ** 2))
        click = (float(xs[j]) + ox, float(ys[j]) + oy)
        feats["inner_r"] = round(float(dist[ys[j], xs[j]]), 1)
        return Candidate(gb, center, click, area, geo["angle"], geo["length"], geo["width"],
                         float(score), feats, None, gcnt)

    def _is_eye(self, j: int, st: np.ndarray, elabels: np.ndarray, dark_in: np.ndarray,
                pink: np.ndarray, dist: np.ndarray) -> bool:
        """Eye = light rectangle containing a dark pupil, surrounded by pink."""
        ex, ey, ew, eh, ea = (int(v) for v in st[:5])
        eps = math.sqrt(ea / 4.0)                     # eye white = 4 sprite pixels
        pad = max(3, int(round(eps * 1.6)) + 2)
        H, W = dark_in.shape[:2]
        x0, y0 = max(0, ex - pad), max(0, ey - pad)
        x1, y1 = min(W, ex + ew + pad), min(H, ey + eh + pad)
        blob = cv2.compare(elabels[y0:y1, x0:x1], j, cv2.CMP_EQ)
        dark = dark_in[y0:y1, x0:x1]
        union = cv2.bitwise_or(blob, dark)
        n, lab, stt, _ = cv2.connectedComponentsWithStats(union, connectivity=8)
        bl = lab[blob > 0]
        if bl.size == 0:
            return False
        k = int(np.bincount(bl).argmax())
        if k == 0:
            return False
        bx, by, bw, bh, barea = (int(v) for v in stt[k, :5])
        # the pupil must be enclosed by the eye, not a gap leading outside
        if bx == 0 and x0 > 0 or by == 0 and y0 > 0 or bx + bw == x1 - x0 and x1 < W \
                or by + bh == y1 - y0 and y1 < H:
            return False
        comp = cv2.compare(lab, k, cv2.CMP_EQ)
        pupil = barea - cv2.countNonZero(cv2.bitwise_and(comp, blob))
        r = pupil / max(1, ea)
        if not (0.12 <= r <= 1.3):
            return False
        pts = cv2.findNonZero(comp)
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        if barea / max(1.0, (rw + 1) * (rh + 1)) < 0.55:
            return False
        ring = cv2.subtract(cv2.dilate(comp, self._kernel(max(3, int(eps) | 1))), comp)
        n_ring = cv2.countNonZero(ring)
        if n_ring == 0:
            return False
        if cv2.countNonZero(cv2.bitwise_and(ring, pink[y0:y1, x0:x1])) < 0.5 * n_ring:
            return False
        # pupils lie inside the head, not at the silhouette border
        depth = float(cv2.mean(dist[y0:y1, x0:x1], mask=comp)[0])
        return depth >= 0.6 * eps
