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
class TargetConfig:
    # "motion": the target is the first small pink object that MOVES (static
    #           page elements are never targets); "sample": the object the
    #           user outlined on the preview / captured with F8.
    mode: str = "motion"
    # a pixel counts as "recently changed" for this long
    fresh_window_ms: float = 400.0
    # share of recently changed pixels needed to call an object "moving"
    min_fresh: float = 0.5
    # appearance similarity needed to re-acquire the learned target
    min_similarity: float = 0.45
    sample_similarity: float = 0.6
    # biggest object (screen px) that may become a target ("small" object)
    max_target_area: int = 30000
    # target counted as lost after this time without seeing it
    lost_timeout_ms: float = 300.0
    # target that does not react to clicks for this long is ignored for a
    # few seconds (it is not the worm). 0 = never. Disabled in test mode.
    unlock_after_ms: float = 1500.0


CONFIG_VERSION = 2


@dataclass
class AppConfig:
    config_version: int = CONFIG_VERSION
    hotkey_toggle: str = "F6"
    hotkey_stop: str = "F7"
    hotkey_calibrate: str = "F8"          # capture the target under the cursor
    # "window" = foreground browser window client area, "monitor:N", "all"
    capture_area: str = "window"
    # "auto" (dxcam -> mss), "dxcam", "mss"
    capture_backend: str = "auto"
    # Pause when the selected window loses focus (window mode only).
    require_focus: bool = True
    # Test mode: everything works, but mouse clicks are not sent.
    dry_run: bool = False
    # Pause between clicks in ms. 0 = maximum speed (a click on every
    # captured frame while the target is visible).
    click_interval_ms: float = 0.0
    # Live preview inside the main window.
    debug_view: bool = True
    preview_fps: int = 60
    # Sleep when nothing was found for a while to save CPU (ms, 0 = never).
    idle_sleep_ms: float = 2.0
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    target: TargetConfig = field(default_factory=TargetConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AppConfig":
        cfg = cls()
        if not isinstance(data, dict):
            return cfg
        _merge_into(cfg, data)
        if int(data.get("config_version", 1) or 1) < 2:
            # v1 -> v2: the preview became part of the main window and is on
            # by default; the old "wait for the jump" settings are gone
            cfg.debug_view = True
        cfg.config_version = CONFIG_VERSION
        cfg.validate()
        return cfg

    def validate(self) -> None:
        d, t = self.detector, self.target
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
        if t.mode not in ("motion", "sample"):
            t.mode = "motion"
        t.fresh_window_ms = float(min(5000, max(50, t.fresh_window_ms)))
        t.min_fresh = float(min(1.0, max(0.05, t.min_fresh)))
        t.min_similarity = float(min(0.99, max(0.05, t.min_similarity)))
        t.sample_similarity = float(min(0.99, max(0.05, t.sample_similarity)))
        t.max_target_area = int(max(d.min_area + 1, t.max_target_area))
        t.lost_timeout_ms = float(min(10000, max(10, t.lost_timeout_ms)))
        t.unlock_after_ms = float(min(60000, max(0, t.unlock_after_ms)))
        self.click_interval_ms = float(min(10000, max(0.0, self.click_interval_ms)))
        self.preview_fps = int(min(144, max(10, self.preview_fps)))
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
