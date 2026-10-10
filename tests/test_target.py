"""TargetFinder: only the moving small pink object becomes the target."""

import numpy as np
import pytest

from wormclicker.config import DetectorConfig, TargetConfig
from wormclicker.sprite import render_worm
from wormclicker.synthetic import scene_with_distractors
from wormclicker.target import TargetFinder, TargetState

W, H = 1280, 720


@pytest.fixture(scope="module")
def page():
    bg = scene_with_distractors(W, H, 0)        # pink text, pink button, particles, glow, icon
    render_worm(bg, (1100, 620), 30, 4)         # a static picture of the worm (decoy)
    return bg


def frame(page, center, angle, ps=4, wiggle=0.0, smooth=False):
    f = page.copy()
    m = render_worm(f, center, angle, ps, smooth, wiggle)
    return f, m


def inside(res, mask):
    t = res.target
    return t is not None and bool(mask[int(t.click[1]), int(t.click[0])])


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self, dt=0.016):
        self.t += dt
        return self.t


def test_static_worm_and_decoys_are_ignored(page):
    tf, now = TargetFinder(), Clock()
    f, _ = frame(page, (400, 400), 10)
    for _ in range(30):
        r = tf.process(f, now())
    assert r.target is None and r.state == TargetState.WAITING
    assert r.blobs >= 2                         # candidates exist, but none moves


def test_worm_acquired_after_it_moves(page):
    tf, now = TargetFinder(), Clock()
    f, _ = frame(page, (400, 400), 10)
    tf.process(f, now())
    f2, m2 = frame(page, (700, 500), 77)        # the user clicked it - it jumped
    r = tf.process(f2, now())
    assert r.acquired and inside(r, m2)
    assert tf.model is not None and tf.model.source == "motion"


def test_target_followed_while_it_stays(page):
    tf, now = TargetFinder(), Clock()
    tf.process(page, now())
    f, m = frame(page, (640, 450), 33)
    tf.process(f, now())
    for _ in range(20):                          # clicked again and again: still the target
        r = tf.process(f, now())
        assert inside(r, m) and not r.acquired and not r.jumped


@pytest.mark.parametrize("angle", [0, 30, 45, 90, 135, 180, 225, 270, 315, 17.5, 199.3])
@pytest.mark.parametrize("ps", [2, 4, 7])
def test_jumps_at_any_rotation_and_scale(page, angle, ps):
    tf, now = TargetFinder(), Clock()
    tf.process(page, now())
    f, m = frame(page, (300, 420), 0, ps)
    assert inside(tf.process(f, now()), m)
    f2, m2 = frame(page, (820, 520), angle, ps)
    r = tf.process(f2, now())
    assert r.jumped and inside(r, m2)


def test_many_random_jumps_never_pick_decoys(page):
    tf, now = TargetFinder(), Clock()
    tf.process(page, now())
    rng = np.random.default_rng(5)
    bad = 0
    for i in range(150):
        c = (float(rng.uniform(80, 1000)), float(rng.uniform(290, 560)))
        f, m = frame(page, c, float(rng.uniform(0, 360)), wiggle=float(rng.uniform(0, 6)), smooth=i % 3 == 0)
        r = tf.process(f, now())
        bad += not inside(r, m)
        now(0.3)                                 # sometimes the worm stays still for a while
        bad += not inside(tf.process(f, now()), m)
    assert bad == 0
    assert tf.model.jumps >= 149


def test_disappearance_and_reappearance(page):
    tf, now = TargetFinder(), Clock()
    tf.process(page, now())
    f, m = frame(page, (500, 400), 0)
    assert inside(tf.process(f, now()), m)
    r = tf.process(page, now())                  # hidden
    assert r.target is None and r.state == TargetState.LOST
    f2, m2 = frame(page, (900, 450), 120)
    r = tf.process(f2, now(0.4))
    assert r.acquired and inside(r, m2)


def test_crawl_animation_in_place_is_followed(page):
    tf, now = TargetFinder(), Clock()
    tf.process(page, now())
    for k, ph in enumerate(np.linspace(0, 2 * np.pi, 12)):
        f, m = frame(page, (600, 450), 40, wiggle=float(ph))
        r = tf.process(f, now())
        if k:
            assert inside(r, m)


def test_unresponsive_target_is_ignored_and_forgotten(page):
    tf, now = TargetFinder(), Clock()
    tf.process(page, now())
    f, m = frame(page, (500, 400), 0)
    tf.process(f, now())
    tf.mark_unresponsive(now())
    assert tf.model is None                      # it never jumped -> wrong object
    for _ in range(5):
        assert tf.process(f, now()).target is None
    f2, m2 = frame(page, (800, 420), 66)         # the real worm moves
    assert inside(tf.process(f2, now()), m2)


def test_map_reset_relocks_learned_target_without_motion(page):
    tf, now = TargetFinder(), Clock()
    tf.process(page, now())
    f, m = frame(page, (500, 400), 0)
    tf.process(f, now())
    f2, m2 = frame(page, (750, 500), 90)
    assert tf.process(f2, now()).jumped
    tf.reset_map()                                # e.g. the browser window was moved
    tf.prev = None
    r = tf.process(f2, now())                     # nothing "moved", but it is the learned worm
    assert inside(r, m2)


def test_sample_mode_learns_from_box_and_finds_static_worm(page):
    tf, now = TargetFinder(DetectorConfig(), TargetConfig(mode="sample")), Clock()
    f, m = frame(page, (450, 420), 20)
    r = tf.process(f, now())
    assert r.target is None and r.state == TargetState.NEED_SAMPLE
    res = tf.learn_sample(f, box=(390, 380, 120, 80), now=now())
    assert res.ok, res.message
    assert tf.model.source == "sample"
    assert inside(tf.process(f, now()), m)
    tf.prev = None
    f2, m2 = frame(page, (850, 500), 140)
    tf.reset_map()                                # no motion information at all
    r = tf.process(f2, now())
    assert inside(r, m2)


def test_sample_from_cursor_point(page):
    tf, now = TargetFinder(), Clock()
    f, m = frame(page, (450, 420), 20)
    ys, xs = np.nonzero(m)
    res = tf.learn_sample(f, point=(int(xs[len(xs) // 2]), int(ys[len(ys) // 2])), now=now())
    assert res.ok, res.message
    assert tf.cfg.mode == "sample"


def test_sample_on_empty_area_fails(page):
    tf = TargetFinder()
    res = tf.learn_sample(page, box=(600, 500, 40, 40))
    assert not res.ok and tf.model is None


def test_bgra_frames_and_timing(page):
    import cv2
    import time

    tf, now = TargetFinder(), Clock()
    tf.process(cv2.cvtColor(page, cv2.COLOR_BGR2BGRA), now())
    f, m = frame(page, (640, 400), 50)
    t0 = time.perf_counter()
    r = tf.process(cv2.cvtColor(f, cv2.COLOR_BGR2BGRA), now())
    assert inside(r, m)
    assert (time.perf_counter() - t0) < 0.5
