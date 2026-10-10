import re
import sys
from pathlib import Path

import pytest

from wormclicker.mouse import DryRunMouse, create_mouse
from wormclicker.sprite import GRID_W, PALETTE_RGB, SPRITE_GRID
from wormclicker.stats import RateMeter, Stats


def test_rate_meter():
    r = RateMeter(1.0)
    for i in range(10):
        r.add(i * 0.05)
    assert r.rate(0.5) == 10
    assert r.rate(1.3) == 4         # only events newer than 0.3


def test_stats_snapshot_is_copy():
    t = [0.0]
    s = Stats(clock=lambda: t[0])
    s.session_started()
    s.click(0.1, 5.0, 0.9)
    s.jump(30.0)
    s.incr("lost")
    t[0] = 0.5
    snap = s.snapshot()
    assert snap.clicks == 1 and snap.jumps == 1 and snap.lost == 1
    assert snap.avg_cps == pytest.approx(2.0)
    s.click(0.1, 5.0, 0.9)
    assert snap.clicks == 1


def test_dry_run_mouse_records():
    m = DryRunMouse()
    m.click(-100, 20)
    assert m.clicks[0][:2] == (-100, 20)


def test_create_mouse_dry_run_never_real():
    assert isinstance(create_mouse(True), DryRunMouse)
    if sys.platform != "win32":
        assert isinstance(create_mouse(False), DryRunMouse)


def test_sprite_grid_consistent():
    assert all(len(r) == GRID_W for r in SPRITE_GRID)
    assert sum(r.count("W") for r in SPRITE_GRID) == 8      # two eyes, 4 white px each
    assert sum(r.count("K") for r in SPRITE_GRID) == 4      # two pupils


def test_practice_page_uses_same_sprite():
    html = (Path(__file__).resolve().parent.parent / "practice" / "index.html").read_text(encoding="utf-8")
    rows = re.findall(r'"([.DMPLWK]{%d})"' % GRID_W, html)
    assert tuple(rows[:len(SPRITE_GRID)]) == SPRITE_GRID
    for ch, (r, g, b) in PALETTE_RGB.items():
        assert f"#{r:02x}{g:02x}{b:02x}" in html.lower(), ch
