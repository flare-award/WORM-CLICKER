"""Detector tests on synthetic frames (no real game frames are available)."""

import numpy as np
import pytest

from wormclicker.config import DetectorConfig
from wormclicker.detector import WormDetector
from wormclicker.sprite import make_background, render_worm
from wormclicker.synthetic import scene, scene_with_distractors


@pytest.fixture(scope="module")
def det():
    return WormDetector(DetectorConfig())


def hit(det_result, mask):
    b = det_result.best
    return b is not None and bool(mask[int(round(b.click[1])), int(round(b.click[0]))])


def test_initial_position(det):
    f, m = scene(worm_center=(640, 400), angle=0, pixel_size=4)
    d = det.detect(f)
    assert hit(d, m)
    assert d.best.score > 0.8
    assert d.best.features["eyes"] == 2


@pytest.mark.parametrize("angle", [0, 15, 30, 45, 90, 135, 180, 225, 270, 315, 360, 7.5, 61, 199.3])
@pytest.mark.parametrize("smooth", [False, True])
def test_rotations(det, angle, smooth):
    f, m = scene(worm_center=(800, 450), angle=angle, pixel_size=4, smooth=smooth)
    assert hit(det.detect(f), m)


@pytest.mark.parametrize("ps", [2, 3, 4, 5, 6, 8, 10])
def test_scales(det, ps):
    for angle in (0, 37, 90, 151):
        f, m = scene(worm_center=(700, 500), angle=angle, pixel_size=ps)
        assert hit(det.detect(f), m), (ps, angle)


def test_random_positions_rotations_scales(det):
    rng = np.random.default_rng(42)
    bad = 0
    n = 120
    for i in range(n):
        ps = int(rng.integers(2, 8))
        ang = float(rng.uniform(0, 360))
        c = (float(rng.uniform(60, 1220)), float(rng.uniform(260, 660)))
        f, m = scene(worm_center=c, angle=ang, pixel_size=ps, smooth=bool(rng.random() < 0.5), seed=i,
                     wiggle=float(rng.uniform(0, 6)))
        bad += not hit(det.detect(f), m)
    assert bad == 0


def test_animation_wiggle(det):
    for phase in np.linspace(0, 2 * np.pi, 9):
        f, m = scene(worm_center=(500, 500), angle=33, pixel_size=4, wiggle=float(phase))
        assert hit(det.detect(f), m)


@pytest.mark.parametrize("center", [(30, 30), (1250, 30), (30, 690), (1250, 690), (640, 8)])
def test_corners_and_borders(det, center):
    f = make_background(1280, 720)
    m = render_worm(f, center, 20, 4)
    d = det.detect(f)
    # partially visible worm in a corner: either a correct click or no click at all
    assert d.best is None or hit(d, m)


def test_whole_frame_any_region(det):
    for c in [(100, 300), (1180, 300), (640, 650), (200, 650), (1100, 600)]:
        f, m = scene(worm_center=c, angle=45, pixel_size=4)
        assert hit(det.detect(f), m), c


def test_no_worm_no_detection(det):
    f = make_background(1280, 720)
    d = det.detect(f)
    assert d.best is None
    assert d.reason


def test_distractors_are_rejected(det):
    for seed in range(60):
        d = det.detect(scene_with_distractors(seed=seed))
        assert d.best is None, (seed, d.best and d.best.features)


def test_worm_among_distractors(det):
    for seed, c in enumerate([(300, 400), (900, 500), (640, 300), (1100, 650)]):
        f, m = scene(worm_center=c, angle=seed * 40, pixel_size=4, seed=seed)
        d = det.detect(f)
        assert hit(d, m)
        assert d.best is d.accepted[0]


def test_jump_redetected_at_new_position(det):
    f1, m1 = scene(worm_center=(300, 600), angle=10)
    f2, m2 = scene(worm_center=(1100, 350), angle=130)
    d1, d2 = det.detect(f1), det.detect(f2)
    assert hit(d1, m1) and hit(d2, m2)
    assert abs(d2.best.click[0] - d1.best.click[0]) > 500


def test_disappearance(det):
    f, m = scene(worm_center=(600, 500))
    assert det.detect(f).best is not None
    empty = scene_with_distractors()
    assert det.detect(empty).best is None


def test_bgra_input(det):
    import cv2

    f, m = scene(worm_center=(640, 500), angle=77)
    d = det.detect(cv2.cvtColor(f, cv2.COLOR_BGR2BGRA))
    assert hit(d, m)


def test_hint_falls_back_to_full_frame(det):
    f, m = scene(worm_center=(1000, 600), angle=12)
    d = det.detect(f, hint=(0, 0, 200, 200))   # wrong hint
    assert hit(d, m)
    d = det.detect(f, hint=(900, 520, 200, 160))
    assert hit(d, m)


@pytest.mark.parametrize("downscale", [1, 2, 3])
def test_downscale_settings(downscale):
    d = WormDetector(DetectorConfig(downscale=downscale))
    f, m = scene(worm_center=(640, 500), angle=200, pixel_size=4)
    assert hit(d.detect(f), m)


def test_rejection_reasons_present(det):
    d = det.detect(scene_with_distractors(seed=3))
    assert d.candidates
    assert all(c.reject for c in d.candidates)


def test_click_point_inside_for_merged_with_text(det):
    """Worm touching pink UI text: click must still be on the worm (or no click)."""
    for ang in (0, 48, 120, 305):
        f, m = scene(worm_center=(274, 103), angle=ang, pixel_size=6)
        d = det.detect(f)
        assert d.best is None or hit(d, m)


def test_two_frames_cache_does_not_hide_worm(det):
    empty = scene_with_distractors(seed=5)
    det.detect(empty)
    f, m = scene(worm_center=(700, 500), seed=5)
    assert hit(det.detect(f), m)
