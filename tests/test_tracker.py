from wormclicker.config import TrackerConfig
from wormclicker.geometry import Rect
from wormclicker.stats import Stats
from wormclicker.tracker import Tracker, Target


def T(x, y, angle=10.0, area=2000, length=90.0, score=0.9):
    return Target(x, y, x, y, length, angle, area, score, Rect(int(x - 45), int(y - 20), 90, 40))


def make(**kw):
    st = Stats()
    return Tracker(TrackerConfig(**kw), st), st


def test_click_when_target_visible():
    tr, _ = make()
    d = tr.update([T(100, 100)], 0.0)
    assert d.click and d.target.x == 100


def test_no_click_without_target():
    tr, _ = make()
    for i in range(10):
        assert not tr.update([], i * 0.01).click


def test_stale_frame_not_clicked_again():
    tr, st = make(stale_timeout_ms=250)
    d = tr.update([T(100, 100)], 0.0)
    tr.on_click(d.target, 0.0)
    for t in (0.005, 0.02, 0.1, 0.2):
        d = tr.update([T(101, 100)], t)       # still at the old spot: stale capture
        assert not d.click
        assert d.state == "await_jump"
    assert st.snapshot().stale_frames == 4


def test_jump_confirmed_and_new_target_clicked():
    tr, st = make()
    d = tr.update([T(100, 100)], 0.0)
    tr.on_click(d.target, 0.0)
    d = tr.update([T(800, 500)], 0.03)
    assert d.click and d.target.x == 800
    s = st.snapshot()
    assert s.jumps == 1 and 29 < s.jump_ms < 31


def test_old_spot_present_together_with_other_candidate_is_stale():
    tr, _ = make()
    d = tr.update([T(100, 100)], 0.0)
    tr.on_click(d.target, 0.0)
    # stale frame: old worm still there, and some other (lower score) candidate
    d = tr.update([T(100, 100), T(600, 600, score=0.7)], 0.01)
    assert not d.click


def test_retry_after_stale_timeout():
    tr, st = make(stale_timeout_ms=100)
    d = tr.update([T(100, 100)], 0.0)
    tr.on_click(d.target, 0.0)
    assert not tr.update([T(100, 100)], 0.05).click
    d = tr.update([T(100, 100)], 0.15)
    assert d.click
    assert st.snapshot().no_jump == 1


def test_nearby_jump_with_new_pose_is_new_target():
    tr, st = make()
    d = tr.update([T(100, 100, angle=10)], 0.0)
    tr.on_click(d.target, 0.0)
    d = tr.update([T(110, 100, angle=80)], 0.02)
    assert d.click
    assert st.snapshot().jumps == 1


def test_disappearance_then_reappearance():
    tr, st = make(lost_timeout_ms=100)
    d = tr.update([T(100, 100)], 0.0)
    tr.on_click(d.target, 0.0)
    for t in (0.05, 0.1, 0.2, 0.3):
        assert not tr.update([], t).click
    assert st.snapshot().lost == 1
    d = tr.update([T(500, 300)], 0.4)
    assert d.click
    s = st.snapshot()
    assert s.reacquired == 1 and s.jumps == 1


def test_static_decoy_blacklisted():
    tr, st = make(stale_timeout_ms=50, blacklist_after=3, blacklist_ttl_s=10)
    t = 0.0
    clicks = 0
    for _ in range(20):
        d = tr.update([T(300, 300)], t)
        if d.click:
            clicks += 1
            tr.on_click(d.target, t)
        t += 0.02
    assert clicks <= 4
    assert st.snapshot().blacklisted == 1
    # the real worm elsewhere is still clicked
    assert tr.update([T(300, 300), T(900, 600, score=0.8)], t).target.x == 900
    # blacklist expires
    tr2_t = t + 11
    assert tr.update([T(300, 300)], tr2_t).click


def test_max_cps_limit():
    tr, _ = make(max_cps=10)
    d = tr.update([T(100, 100)], 0.0)
    tr.on_click(d.target, 0.0)
    assert not tr.update([T(900, 900)], 0.05).click
    assert tr.update([T(900, 900)], 0.11).click


def test_stable_frames():
    tr, _ = make(stable_frames=3)
    assert not tr.update([T(100, 100)], 0.0).click
    assert tr.hint() is not None
    assert not tr.update([T(101, 100)], 0.01).click
    assert tr.update([T(101, 101)], 0.02).click


def test_unstable_target_resets_streak():
    tr, _ = make(stable_frames=2)
    assert not tr.update([T(100, 100)], 0.0).click
    assert not tr.update([T(400, 400)], 0.01).click   # moved: streak restarts
    assert tr.update([T(401, 400)], 0.02).click
