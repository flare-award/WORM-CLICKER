"""Application settings: defaults, JSON load/save.

User settings are stored in ``config.json`` next to the program (or the
``.exe``). Missing keys fall back to the defaults below, unknown keys are
ignored, so old config files keep working after updates.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class DetectorConfig:
    # Hue ranges (OpenCV scale 0..179) treated as "worm pink". Wraps around red.
    hue_ranges: list[list[int]] = field(default_factory=lambda: [[140, 179], [0, 10]])
    min_saturation: int = 60
    min_value: int = 70
    # very light pink (low saturation, high brightness)
    light_min_saturation: int = 25
    light_min_value: int = 160
    # eye whites / pupils
    eye_max_saturation: int = 70
    eye_min_value: int = 180
    pupil_max_value: int = 80
    # silhouette area limits in screen pixels
    min_area: int = 120
    max_area: int = 60000
    # 0 = unknown; set by calibration, used as a soft size prior
    expected_area: float = 0.0
    confidence_threshold: float = 0.58
    # coarse search stride (1 = full resolution, 2 = every 2nd pixel ...)
    downscale: int = 2
    max_candidates: int = 12


@dataclass
class TrackerConfig:
    # After a click the worm must disappear from that spot. While it is still
    # visible there the frame is considered stale and no click is sent.
    stale_timeout_ms: float = 250.0
    # Target counted as "lost" after this time without a confident detection.
    lost_timeout_ms: float = 300.0
    # Number of consecutive frames with a stable detection before clicking.
    stable_frames: int = 1
    # 0 = unlimited. Optional safety cap of clicks per second.
    max_cps: float = 0.0
    # A candidate clicked this many times without a jump is ignored for a while
    # (static picture of a worm in the page UI, etc.).
    blacklist_after: int = 4
    blacklist_ttl_s: float = 15.0


@dataclass
class AppConfig:
    hotkey_toggle: str = "F6"
    hotkey_stop: str = "F7"
    hotkey_calibrate: str = "F8"
    # "window" = foreground browser window client area, "monitor:N", "all"
    capture_area: str = "window"
    # "auto" (dxcam -> mss), "dxcam", "mss"
    capture_backend: str = "auto"
    # Pause when the selected window loses focus (window mode only).
    require_focus: bool = True
    # Test mode: everything works, but mouse clicks are not sent.
    dry_run: bool = False
    debug_view: bool = False
    # Sleep when nothing was found for a while to save CPU (ms, 0 = never).
    idle_sleep_ms: float = 2.0
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AppConfig":
        cfg = cls()
        if not isinstance(data, dict):
            return cfg
        _merge_into(cfg, data)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        d, t = self.detector, self.tracker
        d.downscale = int(min(4, max(1, d.downscale)))
        d.min_area = int(max(10, d.min_area))
        d.max_area = int(max(d.min_area + 1, d.max_area))
        d.confidence_threshold = float(min(0.99, max(0.05, d.confidence_threshold)))
        d.max_candidates = int(min(64, max(1, d.max_candidates)))
        for name in ("min_saturation", "min_value", "light_min_saturation", "light_min_value",
                     "eye_max_saturation", "eye_min_value", "pupil_max_value"):
            setattr(d, name, int(min(255, max(0, getattr(d, name)))))
        ranges = []
        for r in d.hue_ranges or []:
            try:
                lo, hi = int(r[0]), int(r[1])
            except (TypeError, ValueError, IndexError):
                continue
            ranges.append([min(179, max(0, lo)), min(179, max(0, hi))])
        d.hue_ranges = ranges or DetectorConfig().hue_ranges
        t.stable_frames = int(min(10, max(1, t.stable_frames)))
        t.stale_timeout_ms = float(min(5000, max(10, t.stale_timeout_ms)))
        t.lost_timeout_ms = float(min(10000, max(10, t.lost_timeout_ms)))
        t.max_cps = float(max(0.0, t.max_cps))
        t.blacklist_after = int(max(0, t.blacklist_after))
        self.idle_sleep_ms = float(min(100, max(0, self.idle_sleep_ms)))
        if self.capture_backend not in ("auto", "dxcam", "mss"):
            self.capture_backend = "auto"


def _merge_into(obj: Any, data: dict[str, Any]) -> None:
    for f in fields(obj):
        if f.name not in data:
            continue
        cur = getattr(obj, f.name)
        val = data[f.name]
        if hasattr(cur, "__dataclass_fields__"):
            if isinstance(val, dict):
                _merge_into(cur, val)
            continue
        try:
            if isinstance(cur, bool):
                val = bool(val)
            elif isinstance(cur, int):
                val = int(val)
            elif isinstance(cur, float):
                val = float(val)
            elif isinstance(cur, str):
                val = str(val)
            elif isinstance(cur, list) and not isinstance(val, list):
                continue
        except (TypeError, ValueError):
            log.warning("config: bad value for %s: %r", f.name, val)
            continue
        setattr(obj, f.name, val)


def app_dir() -> Path:
    """Folder of the executable (frozen) or of the repository."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def default_config_path() -> Path:
    env = os.environ.get("WORMCLICKER_CONFIG")
    return Path(env) if env else app_dir() / "config.json"


def load_config(path: Path | None = None) -> AppConfig:
    path = path or default_config_path()
    candidates = [path]
    if path == default_config_path():
        candidates.append(app_dir() / "config.default.json")
    for p in candidates:
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    return AppConfig.from_dict(json.load(fh))
            except (OSError, ValueError) as exc:
                log.warning("cannot read config %s: %s", p, exc)
    return AppConfig()


def save_config(cfg: AppConfig, path: Path | None = None) -> None:
    path = path or default_config_path()
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cfg.to_dict(), fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except OSError as exc:
        log.warning("cannot save config %s: %s", path, exc)
