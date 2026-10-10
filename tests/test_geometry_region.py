from types import SimpleNamespace

from wormclicker.geometry import (Rect, bounding_rect, clip_to_monitors, frame_to_screen,
                                  monitor_at_point, monitor_index_containing, normalize_absolute)
from wormclicker.region import RegionProvider

LEFT = Rect(-1920, 0, 1920, 1080)       # secondary monitor left of primary
PRIMARY = Rect(0, 0, 2560, 1440)        # primary, different resolution / DPI
ABOVE = Rect(0, -1080, 1920, 1080)
MONS = [PRIMARY, LEFT, ABOVE]


def test_rect_basics():
    r = Rect(-100, -50, 200, 100)
    assert r.right == 100 and r.bottom == 50
    assert r.contains(-100, -50) and not r.contains(100, 0)
    assert r.intersect(Rect(0, 0, 500, 500)) == Rect(0, 0, 100, 50)
    assert r.intersect(Rect(1000, 1000, 5, 5)) is None


def test_virtual_desktop_with_negative_coordinates():
    vs = bounding_rect(MONS)
    assert vs == Rect(-1920, -1080, 4480, 2520)
    assert monitor_at_point(-5, 10, MONS) == 1
    assert monitor_at_point(5, -10, MONS) == 2
    assert monitor_index_containing(Rect(-1000, 100, 300, 300), MONS) == 1
    assert monitor_index_containing(Rect(-100, 100, 300, 300), MONS) is None   # spans two monitors


def test_clip_window_partially_offscreen():
    win = Rect(-2100, 500, 800, 600)
    assert clip_to_monitors(win, MONS) == Rect(-1920, 500, 620, 600)


def test_frame_to_screen_negative_and_clamped():
    region = Rect(-1920, -1080 + 100, 800, 600)
    assert frame_to_screen(region, 10.4, 20.6) == (-1910, -959)
    assert frame_to_screen(region, 799.9, 599.9) == (-1121, -381)    # clamped inside region


def test_normalize_absolute():
    vs = bounding_rect(MONS)
    assert normalize_absolute(vs.left, vs.top, vs) == (0, 0)
    assert normalize_absolute(vs.right - 1, vs.bottom - 1, vs) == (65535, 65535)


class FakeWin:
    IS_WINDOWS = True

    def __init__(self):
        self.fg = 0
        self.pids = {}
        self.rects = {}
        self.usable = {}

    def foreground_window(self):
        return self.fg

    def root_window(self, h):
        return h

    def window_pid(self, h):
        return self.pids.get(h, 0)

    def current_pid(self):
        return 1

    def window_title(self, h):
        return f"win{h}"

    def is_window_usable(self, h):
        return self.usable.get(h, (True, ""))

    def client_rect_on_screen(self, h):
        return self.rects.get(h)


def make_window_provider():
    w = FakeWin()
    t = [0.0]
    rp = RegionProvider("window", lambda: MONS, require_focus=True, win=w, clock=lambda: t[0])
    return rp, w, t


def test_window_mode_waits_while_own_window_focused():
    rp, w, t = make_window_provider()
    w.fg, w.pids[5] = 5, 1          # our own UI is in front
    rect, why = rp.get()
    assert rect is None and "браузер" in why


def test_window_mode_tracks_move_resize_and_focus():
    rp, w, t = make_window_provider()
    w.fg, w.pids[7] = 7, 999
    w.rects[7] = Rect(-1800, 100, 1200, 800)
    rect, _ = rp.get()
    assert rect == Rect(-1800, 100, 1200, 800) and rp.hwnd == 7
    w.rects[7] = Rect(100, 100, 1000, 700)       # moved to the primary monitor
    t[0] = 0.1
    assert rp.get()[0] == Rect(-1800, 100, 1200, 800)   # cached for 250 ms
    t[0] = 0.5
    assert rp.get()[0] == Rect(100, 100, 1000, 700)
    w.fg = 8                                     # user switched to another app
    rect, why = rp.get()
    assert rect is None and "фокус" in why
    rp.require_focus = False
    assert rp.get()[0] is not None
    w.usable[7] = (False, "окно браузера свернуто")
    assert rp.get() == (None, "окно браузера свернуто")


def test_monitor_modes():
    rp = RegionProvider("monitor:1", lambda: MONS)
    assert rp.get()[0] == LEFT
    rp.area = "all"
    assert rp.get()[0] == bounding_rect(MONS)
    rp.area = "monitor:9"
    assert rp.get()[0] is None


def test_window_mode_not_on_other_os():
    rp = RegionProvider("window", lambda: MONS, win=SimpleNamespace(IS_WINDOWS=False))
    rect, why = rp.get()
    assert rect is None and "Windows" in why
