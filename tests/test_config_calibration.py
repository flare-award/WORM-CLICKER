import json

import cv2
import numpy as np

from wormclicker.calibration import apply_calibration, calibrate_patch, _circular_range
from wormclicker.config import AppConfig, DetectorConfig, load_config, save_config
from wormclicker.detector import WormDetector
from wormclicker.sprite import make_background, render_worm


def test_config_roundtrip(tmp_path):
    cfg = AppConfig(hotkey_toggle="Ctrl+F9", dry_run=True)
    cfg.detector.confidence_threshold = 0.7
    cfg.tracker.max_cps = 25
    p = tmp_path / "c.json"
    save_config(cfg, p)
    back = load_config(p)
    assert back == cfg


def test_config_tolerates_garbage(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"hotkey_toggle": "F10", "unknown": 1,
                             "detector": {"downscale": 99, "min_area": "abc", "hue_ranges": [[300, -5], "x"]},
                             "tracker": {"stable_frames": 0}}), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.hotkey_toggle == "F10"
    assert cfg.detector.downscale == 4
    assert cfg.detector.min_area == DetectorConfig().min_area
    assert cfg.detector.hue_ranges == [[179, 0]]
    assert cfg.tracker.stable_frames == 1
    p.write_text("{broken", encoding="utf-8")
    assert load_config(p) == AppConfig()


def test_default_config_file_matches_defaults():
    from pathlib import Path

    p = Path(__file__).resolve().parent.parent / "config.default.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert AppConfig.from_dict(data) == AppConfig()
    assert data["hotkey_toggle"] == "F6" and data["hotkey_stop"] == "F7"
    assert data["dry_run"] is False


def test_circular_range_wraps():
    h = np.array([175, 178, 2, 4, 176, 1], np.uint8)
    r = _circular_range(h, margin=2)
    assert len(r) == 2 and r[0][1] == 179 and r[1][0] == 0


def _recolor_purple(frame, mask):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.int16)
    h[mask] = (h[mask] - 30) % 180          # pink (~166) -> purple (~136)
    hsv[..., 0] = h.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_calibration_adapts_to_different_palette():
    f = make_background(800, 600)
    m = render_worm(f, (400, 300), 25, 4)
    purple = _recolor_purple(f, m)
    cfg = DetectorConfig(hue_ranges=[[150, 179], [0, 8]])
    det = WormDetector(cfg)
    assert det.detect(purple).best is None        # not pink enough for the defaults
    patch = purple[300 - 160:300 + 160, 400 - 160:400 + 160]
    res = calibrate_patch(patch, (160, 160))
    assert res.ok, res.message
    apply_calibration(cfg, res)
    det.update_config(cfg)
    d = det.detect(purple)
    assert d.best is not None and m[int(d.best.click[1]), int(d.best.click[0])]
    assert cfg.expected_area > 0


def test_calibration_fails_on_empty_area():
    res = calibrate_patch(make_background(320, 320), (160, 160))
    assert not res.ok
