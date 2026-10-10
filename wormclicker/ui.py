"""Main window: controls on the left, live preview on the right.

The UI thread never captures or analyses frames: it reads stats snapshots
4 times per second and shows the newest preview frame published by the
worker (up to 60-120 fps, binary PPM -> Tk photo, no PNG encoding), so it
can not slow down the click loop and the click loop can not freeze it.
"""

from __future__ import annotations

import copy
import logging
import queue
import time
import tkinter as tk
from tkinter import font as tkfont

from . import __version__, winapi
from .automation import AutomationEngine, Status
from .capture import list_monitors
from .config import AppConfig, DetectorConfig, save_config
from .geometry import Rect
from .hotkeys import HotkeyError, HotkeyManager, hotkey_from_tk, parse_hotkey
from .formatting import format_stats, interval_text, interval_to_pos, pos_to_interval  # noqa: F401
from .target import SampleResult
from .widgets import (ACCENT, ACCENT_SOFT, AMBER, BG, BLUE, BORDER, DIM, GREEN, MUTED, PREVIEW_BG, RED,
                      SURFACE, SURFACE2, SURFACE3, TEXT, Card, Dropdown, LabeledToggle, NumEntry, RoundButton,
                      Scale, Segmented, Slider, StatusPill, Toggle, mix, px, round_rect, sprite_photo)

log = logging.getLogger(__name__)

STATUS_COLORS = {
    Status.IDLE: "#8d95aa",
    Status.PAUSED: AMBER,
    Status.SEARCHING: BLUE,
    Status.ACTIVE: GREEN,
    Status.ERROR: RED,
}
INTERVAL_PRESETS = [(0, "Макс"), (1, "1"), (5, "5"), (10, "10"), (25, "25"), (50, "50"), (100, "100"), (250, "250")]
TILES = [
    ("cps", "кликов / с"),
    ("jumps_ps", "≈ попаданий / с"),
    ("fps", "кадров / с"),
    ("clicks", "кликов всего"),
    ("jumps", "≈ попаданий"),
    ("pipeline_ms", "кадр → клик, мс"),
]
MODE_HELP = {
    "motion": "Всё неподвижное игнорируется: текст, кнопки, картинки. Кликните по червю один раз — "
              "он прыгнет, программа поймёт, что это он, и начнёт кликать.",
    "sample": "Обведите червя мышью прямо на превью справа (или наведите на него курсор в игре и "
              "нажмите F8). Программа запомнит, как он выглядит, и будет искать похожий объект, "
              "даже если он не двигается.",
}


class ScrollColumn(tk.Frame):
    """Vertical scroll area with a slim custom scrollbar."""

    def __init__(self, master: tk.Misc, width: int):
        super().__init__(master, bg=BG)
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0, bd=0, width=width)
        self.bar = tk.Canvas(self, bg=BG, highlightthickness=0, bd=0, width=px(8))
        self.inner = tk.Frame(self.canvas, bg=BG)
        self._win = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.bar.pack(side="right", fill="y", padx=(px(4), 0))
        self.inner.bind("<Configure>", lambda e: self._update())
        self.canvas.bind("<Configure>", self._resize)
        self.bar.bind("<B1-Motion>", self._drag)
        self.bar.bind("<Button-1>", self._drag)
        for w in (self.canvas, self.inner):
            w.bind("<Enter>", lambda e: self._wheel_bind(True))
            w.bind("<Leave>", lambda e: self._wheel_bind(False))

    def _wheel_bind(self, on: bool) -> None:
        if on:
            self.bind_all("<MouseWheel>", self._wheel)
            self.bind_all("<Button-4>", lambda e: self._scroll(-1))
            self.bind_all("<Button-5>", lambda e: self._scroll(1))
        else:
            self.unbind_all("<MouseWheel>")
            self.unbind_all("<Button-4>")
            self.unbind_all("<Button-5>")

    def _resize(self, e) -> None:
        self.canvas.itemconfigure(self._win, width=e.width)
        self._update()

    def _update(self) -> None:
        h = self.inner.winfo_reqheight()
        self.canvas.configure(scrollregion=(0, 0, self.canvas.winfo_width(), h))
        self._draw_bar()

    def _scroll(self, units: int) -> None:
        top, bottom = self.canvas.yview()
        if bottom - top < 1.0:
            self.canvas.yview_scroll(units, "units")
            self._draw_bar()

    def _wheel(self, e) -> None:
        self._scroll(int(-e.delta / 120) or (-1 if e.delta > 0 else 1))

    def _drag(self, e) -> None:
        top, bottom = self.canvas.yview()
        span = bottom - top
        if span >= 1.0:
            return
        self.canvas.yview_moveto(max(0.0, e.y / max(1, self.bar.winfo_height()) - span / 2))
        self._draw_bar()

    def _draw_bar(self) -> None:
        self.bar.delete("all")
        top, bottom = self.canvas.yview()
        if bottom - top >= 0.999:
            return
        h = self.bar.winfo_height()
        round_rect(self.bar, 1, top * h + 1, px(7), bottom * h - 1, px(3), fill=SURFACE3, outline="")


class App:
    PREVIEW_POLL_MS = 4

    def __init__(self, root: tk.Tk, engine: AutomationEngine, hotkeys: HotkeyManager, cfg: AppConfig):
        self.root = root
        self.engine = engine
        self.hotkeys = hotkeys
        self.cfg = copy.deepcopy(cfg)
        self._events: queue.SimpleQueue = queue.SimpleQueue()
        self._apply_job: str | None = None
        self._assigning: str | None = None
        self._closing = False
        self._last_seq = -1
        self._frame_times: list[float] = []
        self._img_geom = (0, 0, 0, 0)          # x0, y0, w, h of the image on the preview canvas
        self._df = None
        self._sel_start: tuple[int, int] | None = None
        self._calib_job: str | None = None

        Scale.init(root)
        self._fonts()
        engine.on_sample = lambda res: self._events.put(("sample", res))
        root.title("Worm Clicker")
        root.configure(bg=BG)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = min(px(1240), int(sw * 0.94)), min(px(780), int(sh * 0.88))
        root.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")
        root.minsize(min(px(980), w), min(px(600), h))
        try:
            self._icon = sprite_photo(root, 2)
            root.iconphoto(True, self._icon)
        except tk.TclError:
            pass
        self._build()
        self._register_hotkeys()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(300, self._post_map)
        root.after(100, self._tick)
        root.after(self.PREVIEW_POLL_MS, self._preview_tick)

    # ================================================================ fonts
    def _fonts(self) -> None:
        fams = set(tkfont.families(self.root))
        fam = next((f for f in ("Segoe UI", "Inter", "Helvetica Neue", "DejaVu Sans", "Arial") if f in fams),
                   "TkDefaultFont")
        mono = next((f for f in ("Cascadia Mono", "Consolas", "JetBrains Mono", "DejaVu Sans Mono", "Courier New")
                     if f in fams), "TkFixedFont")
        self.f_body = (fam, 10)
        self.f_small = (fam, 9)
        self.f_label = (fam, 10, "bold")
        self.f_h2 = (fam, 11, "bold")
        self.f_title = (fam, 15, "bold")
        self.f_tile = (fam, 17, "bold")
        self.f_button = (fam, 12, "bold")
        self.f_big = (fam, 14, "bold")
        self.f_mono = (mono, 10, "bold")
        self.f_mono_small = (mono, 9)

    def _label(self, parent, text="", font=None, fg=TEXT, **kw) -> tk.Label:
        return tk.Label(parent, text=text, font=font or self.f_body, fg=fg, bg=parent.cget("bg"),
                        anchor="w", justify="left", **kw)

    # ================================================================ build
    def _build(self) -> None:
        r = self.root
        r.columnconfigure(1, weight=1)
        r.rowconfigure(1, weight=1)

        # ---- header
        head = tk.Frame(r, bg=BG, padx=px(18), pady=px(12))
        head.grid(row=0, column=0, columnspan=2, sticky="ew")
        self._logo = sprite_photo(r, max(2, px(2)))
        tk.Label(head, image=self._logo, bg=BG).pack(side="left", padx=(0, px(12)))
        titles = tk.Frame(head, bg=BG)
        titles.pack(side="left")
        self._label(titles, "Worm Clicker", self.f_title).pack(anchor="w")
        self._label(titles, f"охота на розового червя · v{__version__}", self.f_small, DIM).pack(anchor="w")
        self.pill = StatusPill(head, font=self.f_label)
        self.pill.pack(side="right")

        # ---- left column
        left = ScrollColumn(r, px(392))
        left.grid(row=1, column=0, sticky="nsw", padx=(px(18), px(8)), pady=(0, px(8)))
        col = left.inner
        col.columnconfigure(0, weight=1)
        self._build_control(col).grid(row=0, column=0, sticky="ew")
        self._build_stats(col).grid(row=1, column=0, sticky="ew", pady=(px(12), 0))
        self.tab_var = tk.StringVar(value="clicks")
        Segmented(col, [("clicks", "Клики"), ("target", "Цель"), ("capture", "Захват"), ("keys", "Клавиши")],
                  self.tab_var, command=self._show_tab, font=self.f_label).grid(
            row=2, column=0, sticky="ew", pady=(px(12), 0))
        self.tab_card = Card(col)
        self.tab_card.grid(row=3, column=0, sticky="ew", pady=(px(8), px(4)))
        self.tab_card.columnconfigure(0, weight=1)
        self.tabs = {
            "clicks": self._build_clicks(self.tab_card),
            "target": self._build_target(self.tab_card),
            "capture": self._build_capture(self.tab_card),
            "keys": self._build_keys(self.tab_card),
        }
        self._show_tab("clicks")

        # ---- preview
        self._build_preview(r).grid(row=1, column=1, sticky="nsew", padx=(px(8), px(18)), pady=(0, px(8)))

        # ---- footer
        foot = tk.Frame(r, bg=BG, padx=px(18), pady=px(6))
        foot.grid(row=2, column=0, columnspan=2, sticky="ew")
        self._label(foot, "≈ попадания — оценка по прыжкам червя после кликов; точное число показывает игра.",
                    self.f_small, DIM).pack(side="left")
        self.keys_hint = self._label(foot, "", self.f_small, DIM)
        self.keys_hint.pack(side="right")
        self._update_keys_hint()

    # ---------------------------------------------------------------- control
    def _build_control(self, parent) -> tk.Frame:
        c = Card(parent)
        c.columnconfigure(0, weight=1)
        self.start_btn = RoundButton(c, "▶   Старт", self._toggle, kind="primary", height=52, width=340,
                                     font=self.f_button, hint=self.cfg.hotkey_toggle, radius=12)
        self.start_btn.grid(row=0, column=0, sticky="ew")
        self.dry_var = tk.BooleanVar(value=self.cfg.dry_run)
        LabeledToggle(c, "Тест без кликов", self.dry_var, self._dry_changed,
                      hint="всё работает, но мышь не нажимается", font=self.f_body,
                      hint_font=self.f_small).grid(row=1, column=0, sticky="w", pady=(px(12), 0))
        self.msg = self._label(c, self.engine.message, self.f_small, MUTED, wraplength=px(340))
        self.msg.grid(row=2, column=0, sticky="ew", pady=(px(10), 0))
        return c

    def _build_stats(self, parent) -> tk.Frame:
        c = Card(parent, pady=12)
        for i in range(3):
            c.columnconfigure(i, weight=1, uniform="tile")
        self.tile_vals: dict[str, tk.Label] = {}
        for i, (key, caption) in enumerate(TILES):
            t = tk.Frame(c, bg=SURFACE2, padx=px(10), pady=px(8))
            t.grid(row=i // 3, column=i % 3, sticky="nsew", padx=(0 if i % 3 == 0 else px(6), 0),
                   pady=(0 if i < 3 else px(6), 0))
            v = tk.Label(t, text="—", font=self.f_tile, bg=SURFACE2,
                         fg=ACCENT if key in ("cps", "jumps_ps") else TEXT, anchor="w")
            v.pack(anchor="w")
            tk.Label(t, text=caption, font=self.f_small, bg=SURFACE2, fg=DIM, anchor="w").pack(anchor="w")
            self.tile_vals[key] = v
        row = tk.Frame(c, bg=SURFACE)
        row.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(px(10), 0))
        self.detail = self._label(row, "", self.f_small, DIM, wraplength=px(290))
        self.detail.pack(side="left", fill="x", expand=True)
        reset = tk.Label(row, text="сбросить", font=self.f_small, bg=SURFACE, fg=ACCENT, cursor="hand2")
        reset.pack(side="right", anchor="n")
        reset.bind("<Button-1>", lambda e: self.engine.stats.reset())
        return c

    # ---------------------------------------------------------------- tabs
    def _show_tab(self, key: str) -> None:
        if self.tab_var.get() != key:
            self.tab_var.set(key)             # keeps the switcher highlight in sync
        for k, f in self.tabs.items():
            if k == key:
                f.grid(row=0, column=0, sticky="ew")
            else:
                f.grid_forget()

    def _section(self, parent, row: int, title: str, sub: str | None = None, pad_top: int = 0) -> None:
        self._label(parent, title, self.f_label).grid(row=row, column=0, columnspan=3, sticky="w",
                                                      pady=(px(pad_top), 0))
        if sub:
            self._label(parent, sub, self.f_small, DIM, wraplength=px(330)).grid(
                row=row + 1, column=0, columnspan=3, sticky="w", pady=(px(2), 0))

    def _build_clicks(self, parent) -> tk.Frame:
        f = tk.Frame(parent, bg=SURFACE)
        f.columnconfigure(0, weight=1)
        top = tk.Frame(f, bg=SURFACE)
        top.grid(row=0, column=0, sticky="ew")
        self._label(top, "Скорость кликов", self.f_label).pack(side="left")
        self.iv_value = self._label(top, "", self.f_big, ACCENT)
        self.iv_value.pack(side="right")
        self.iv_sub = self._label(f, "", self.f_small, MUTED)
        self.iv_sub.grid(row=1, column=0, sticky="e")
        self.iv_slider = Slider(f, interval_to_pos(self.cfg.click_interval_ms), self._slider_moved)
        self.iv_slider.grid(row=2, column=0, sticky="ew", pady=(px(8), px(4)))
        chips = tk.Frame(f, bg=SURFACE)
        chips.grid(row=3, column=0, sticky="ew")
        self.iv_chips: dict[float, tk.Label] = {}
        for i, (ms, text) in enumerate(INTERVAL_PRESETS):
            chips.columnconfigure(i, weight=1, uniform="chip")
            lb = tk.Label(chips, text=text, font=self.f_small, bg=SURFACE2, fg=MUTED, pady=px(4), cursor="hand2")
            lb.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else px(3), 0))
            lb.bind("<Button-1>", lambda e, v=ms: self._set_interval(v))
            self.iv_chips[float(ms)] = lb
        own = tk.Frame(f, bg=SURFACE)
        own.grid(row=4, column=0, sticky="w", pady=(px(12), 0))
        self._label(own, "Свой интервал:", self.f_body, MUTED).pack(side="left")
        self.iv_entry = NumEntry(own, self.cfg.click_interval_ms, self._set_interval, width=7, font=self.f_mono,
                                 minimum=0, maximum=10000)
        self.iv_entry.pack(side="left", padx=px(8), ipady=px(3))
        self._label(own, "мс  (0 = максимум)", self.f_body, MUTED).pack(side="left")
        self._label(f, "Пауза между кликами. «Максимум» — клик по цели на каждом захваченном кадре, без "
                       "ожидания прыжка. Если игра не успевает засчитывать клики, увеличьте интервал.",
                    self.f_small, DIM, wraplength=px(330)).grid(row=5, column=0, sticky="w", pady=(px(12), 0))
        self._show_interval()
        return f

    def _build_target(self, parent) -> tk.Frame:
        f = tk.Frame(parent, bg=SURFACE)
        f.columnconfigure(0, weight=1)
        self._label(f, "Как находить цель", self.f_label).grid(row=0, column=0, sticky="w")
        self.mode_var = tk.StringVar(value=self.cfg.target.mode)
        Segmented(f, [("motion", "По движению"), ("sample", "По образцу")], self.mode_var,
                  command=self._mode_changed, font=self.f_body).grid(row=1, column=0, sticky="ew", pady=(px(8), 0))
        self.mode_help = self._label(f, MODE_HELP[self.cfg.target.mode], self.f_small, MUTED, wraplength=px(330))
        self.mode_help.grid(row=2, column=0, sticky="w", pady=(px(8), 0))

        info = tk.Frame(f, bg=SURFACE2, padx=px(12), pady=px(8))
        info.grid(row=3, column=0, sticky="ew", pady=(px(12), 0))
        self._label(info, "Цель", self.f_small, DIM).pack(anchor="w")
        self.target_info = self._label(info, "не выбрана", self.f_body, TEXT, wraplength=px(300))
        self.target_info.pack(anchor="w")

        btns = tk.Frame(f, bg=SURFACE)
        btns.grid(row=4, column=0, sticky="ew", pady=(px(10), 0))
        btns.columnconfigure(0, weight=1, uniform="b")
        btns.columnconfigure(1, weight=1, uniform="b")
        RoundButton(btns, "Сбросить цель", self._reset_target, height=34, width=150, font=self.f_body).grid(
            row=0, column=0, sticky="ew", padx=(0, px(4)))
        self.cursor_btn = RoundButton(btns, "Под курсором через 3 с", self._cursor_delayed, kind="accent-outline",
                                      height=34, width=150, font=self.f_body)
        self.cursor_btn.grid(row=0, column=1, sticky="ew", padx=(px(4), 0))
        self.sample_msg = self._label(f, "", self.f_small, MUTED, wraplength=px(330))
        self.sample_msg.grid(row=5, column=0, sticky="w", pady=(px(8), 0))

        size = tk.Frame(f, bg=SURFACE)
        size.grid(row=6, column=0, sticky="w", pady=(px(10), 0))
        self._label(size, "Размер цели, px:", self.f_body, MUTED).pack(side="left")
        self.min_area = NumEntry(size, self.cfg.detector.min_area, lambda v: self._set_area("min", v), width=6,
                                 font=self.f_mono, minimum=10, maximum=500000, integer=True)
        self.min_area.pack(side="left", padx=(px(8), px(4)), ipady=px(3))
        self._label(size, "—", self.f_body, MUTED).pack(side="left")
        self.max_area = NumEntry(size, self.cfg.target.max_target_area, lambda v: self._set_area("max", v), width=7,
                                 font=self.f_mono, minimum=20, maximum=1000000, integer=True)
        self.max_area.pack(side="left", padx=(px(4), 0), ipady=px(3))
        return f

    def _build_capture(self, parent) -> tk.Frame:
        f = tk.Frame(parent, bg=SURFACE)
        f.columnconfigure(0, weight=1)
        self._label(f, "Где искать", self.f_label).grid(row=0, column=0, sticky="w")
        self.area_var = tk.StringVar(value=self.cfg.capture_area)
        self.area_dd = Dropdown(f, self._area_options(), self.area_var, lambda k: self._schedule_apply(0),
                                font=self.f_body)
        self.area_dd.grid(row=1, column=0, sticky="ew", pady=(px(6), 0))
        self._label(f, "Способ захвата экрана", self.f_label).grid(row=2, column=0, sticky="w", pady=(px(14), 0))
        self.backend_var = tk.StringVar(value=self.cfg.capture_backend)
        Segmented(f, [("auto", "Авто"), ("dxcam", "DXGI"), ("mss", "GDI")], self.backend_var,
                  command=lambda k: self._schedule_apply(0), font=self.f_body).grid(
            row=3, column=0, sticky="ew", pady=(px(6), 0))
        self._label(f, "Частота превью", self.f_label).grid(row=4, column=0, sticky="w", pady=(px(14), 0))
        self.fps_var = tk.StringVar(value=str(self.cfg.preview_fps if self.cfg.preview_fps in (30, 60, 90, 120)
                                              else 60))
        Segmented(f, [("30", "30"), ("60", "60"), ("90", "90"), ("120", "120 fps")], self.fps_var,
                  command=lambda k: self._schedule_apply(0), font=self.f_body).grid(
            row=5, column=0, sticky="ew", pady=(px(6), 0))
        self.focus_var = tk.BooleanVar(value=self.cfg.require_focus)
        LabeledToggle(f, "Пауза, когда браузер не в фокусе", self.focus_var, lambda: self._schedule_apply(0),
                      hint="клики не уйдут в другие программы", font=self.f_body, hint_font=self.f_small).grid(
            row=6, column=0, sticky="w", pady=(px(14), 0))
        self.capture_info = self._label(f, "", self.f_small, DIM, wraplength=px(330))
        self.capture_info.grid(row=7, column=0, sticky="w", pady=(px(12), 0))
        return f

    def _build_keys(self, parent) -> tk.Frame:
        f = tk.Frame(parent, bg=SURFACE)
        f.columnconfigure(0, weight=1)
        self.hk_vals = {"toggle": self.cfg.hotkey_toggle, "stop": self.cfg.hotkey_stop,
                        "calibrate": self.cfg.hotkey_calibrate}
        self.hk_chips: dict[str, tk.Label] = {}
        self.hk_err: dict[str, tk.Label] = {}
        rows = [("toggle", "Старт / стоп"), ("stop", "Аварийная остановка"), ("calibrate", "Захват цели под курсором")]
        for i, (act, title) in enumerate(rows):
            self._label(f, title, self.f_body).grid(row=2 * i, column=0, sticky="w", pady=(px(10) if i else 0, 0))
            chip = tk.Label(f, text=self.hk_vals[act], font=self.f_mono, bg=SURFACE3, fg=TEXT, padx=px(14),
                            pady=px(5), cursor="hand2", width=12)
            chip.grid(row=2 * i, column=1, sticky="e", pady=(px(10) if i else 0, 0))
            chip.bind("<Button-1>", lambda e, a=act: self._assign(a))
            self.hk_chips[act] = chip
            err = self._label(f, "", self.f_small, AMBER, wraplength=px(330))
            err.grid(row=2 * i + 1, column=0, columnspan=2, sticky="w")
            self.hk_err[act] = err
        self._label(f, "Нажмите на клавишу справа и затем новое сочетание (Esc — отмена). Клавиши работают "
                       "глобально — и когда активен браузер.", self.f_small, DIM, wraplength=px(330)).grid(
            row=7, column=0, columnspan=2, sticky="w", pady=(px(14), 0))
        return f

    # ---------------------------------------------------------------- preview
    def _build_preview(self, parent) -> tk.Frame:
        c = Card(parent, padx=14, pady=12)
        c.columnconfigure(0, weight=1)
        c.rowconfigure(1, weight=1)
        head = tk.Frame(c, bg=SURFACE)
        head.grid(row=0, column=0, sticky="ew", pady=(0, px(10)))
        self._label(head, "Превью", self.f_h2).pack(side="left")
        self.debug_var = tk.BooleanVar(value=self.cfg.debug_view)
        Toggle(head, self.debug_var, self._debug_changed).pack(side="left", padx=(px(12), 0))
        self.fps_lbl = self._label(head, "", self.f_mono_small, DIM)
        self.fps_lbl.pack(side="right")
        self.cv = tk.Canvas(c, bg=PREVIEW_BG, highlightthickness=1, highlightbackground=BORDER, bd=0,
                            cursor="crosshair")
        self.cv.grid(row=1, column=0, sticky="nsew")
        # NB: no width/height here - an explicit size would freeze the image at that size
        # and the loaded frames would be cropped to it (that was a 1x1 black preview).
        self._photo = tk.PhotoImage(master=self.root)
        self._img_item = self.cv.create_image(0, 0, image=self._photo, anchor="nw", state="hidden")
        self._ph_text = self.cv.create_text(0, 0, text="", fill=MUTED, font=self.f_body, justify="center")
        self._hud_bg = self.cv.create_polygon(0, 0, 0, 0, fill="", outline="")
        self._hud_dot = self.cv.create_oval(0, 0, 0, 0, fill="", outline="")
        self._hud_text = self.cv.create_text(0, 0, text="", anchor="w", fill=TEXT, font=self.f_small)
        self._tag_bg = self.cv.create_rectangle(0, 0, 0, 0, fill=ACCENT, outline="", state="hidden")
        self._tag_text = self.cv.create_text(0, 0, text="", anchor="sw", fill="#ffffff", font=self.f_small,
                                             state="hidden")
        self._sel_rect = self.cv.create_rectangle(0, 0, 0, 0, outline=ACCENT, width=2, dash=(6, 3), state="hidden")
        self.cv.bind("<Configure>", self._preview_resized)
        self.cv.bind("<ButtonPress-1>", self._sel_press)
        self.cv.bind("<B1-Motion>", self._sel_move)
        self.cv.bind("<ButtonRelease-1>", self._sel_release)
        foot = tk.Frame(c, bg=SURFACE)
        foot.grid(row=2, column=0, sticky="ew", pady=(px(10), 0))
        self.preview_info = self._label(foot, "", self.f_small, MUTED)
        self.preview_info.pack(side="left")
        self._label(foot, "обведите червя мышью — образец цели", self.f_small, DIM).pack(side="right")
        return c

    def _preview_resized(self, e) -> None:
        self.engine.preview_box = (max(64, e.width - px(4)), max(64, e.height - px(4)))
        self.cv.coords(self._ph_text, e.width / 2, e.height / 2)
        self.cv.itemconfigure(self._ph_text, width=max(100, e.width - px(80)))
        self._last_seq = -1

    def _preview_tick(self) -> None:
        if self._closing:
            return
        try:
            df = self.engine.debug_frame
            if not self.debug_var.get():
                if self._last_seq != -2:
                    self._last_seq = -2
                    self._show_placeholder("Превью выключено. Включите переключатель выше — это не замедляет клики.")
            elif df is not None and df.seq != self._last_seq:
                self._last_seq = df.seq
                self._show_frame(df)
        except Exception:  # pragma: no cover - never kill the UI loop
            log.exception("preview update failed")
        self.root.after(self.PREVIEW_POLL_MS, self._preview_tick)

    def _show_placeholder(self, text: str) -> None:
        self._df = None
        self.cv.itemconfigure(self._img_item, state="hidden")
        self.cv.itemconfigure(self._ph_text, text=text, state="normal")
        for it in (self._tag_bg, self._tag_text):
            self.cv.itemconfigure(it, state="hidden")
        self._hud("", None)

    def _show_frame(self, df) -> None:
        from .debug_view import render, target_box_on_preview, to_ppm

        rgb = render(df)
        if rgb is None:
            self._show_placeholder(df.message or "нет изображения")
            return
        self._df = df
        ih, iw = rgb.shape[:2]
        self._photo.configure(data=to_ppm(rgb), format="PPM")
        cw, ch = self.cv.winfo_width(), self.cv.winfo_height()
        x0, y0 = (cw - iw) // 2, (ch - ih) // 2
        self._img_geom = (x0, y0, iw, ih)
        self.cv.coords(self._img_item, x0, y0)
        self.cv.itemconfigure(self._img_item, state="normal")
        self.cv.itemconfigure(self._ph_text, state="hidden")
        box = target_box_on_preview(df)
        if box is not None:
            bx, by = x0 + box[0], y0 + box[1]
            label = "ЦЕЛЬ" if df.target.similarity <= 0 else f"ЦЕЛЬ  {df.target.similarity:.2f}"
            self.cv.itemconfigure(self._tag_text, text=label, state="normal")
            self.cv.coords(self._tag_text, bx - px(2), by - px(10))
            tb = self.cv.bbox(self._tag_text)
            if tb:
                self.cv.coords(self._tag_bg, tb[0] - px(5), tb[1] - px(2), tb[2] + px(5), tb[3] + px(1))
            self.cv.itemconfigure(self._tag_bg, state="normal")
            self.cv.tag_raise(self._tag_bg)
            self.cv.tag_raise(self._tag_text)
        else:
            for it in (self._tag_bg, self._tag_text):
                self.cv.itemconfigure(it, state="hidden")
        color = GREEN if df.target is not None else (BLUE if df.state in ("WAITING", "NEED_SAMPLE") else AMBER)
        self._hud(df.message, color, x0, y0)
        self.cv.tag_raise(self._sel_rect)
        now = time.perf_counter()
        self._frame_times.append(now)
        while self._frame_times and self._frame_times[0] < now - 1.0:
            self._frame_times.pop(0)

    def _hud(self, text: str, color: str | None, x0: int = 0, y0: int = 0) -> None:
        if not text or color is None:
            for it in (self._hud_bg, self._hud_dot, self._hud_text):
                self.cv.itemconfigure(it, state="hidden")
            return
        x, y = max(x0, 0) + px(10), max(y0, 0) + px(10)
        self.cv.itemconfigure(self._hud_text, text=text, state="normal")
        self.cv.coords(self._hud_text, x + px(22), y + px(13))
        tb = self.cv.bbox(self._hud_text)
        w = (tb[2] - tb[0]) if tb else px(100)
        self.cv.delete(self._hud_bg)
        self._hud_bg = round_rect(self.cv, x, y, x + w + px(34), y + px(26), px(13),
                                  fill=mix(PREVIEW_BG, "#000000", 0.2), outline=mix(PREVIEW_BG, color, 0.6))
        r = px(4)
        self.cv.coords(self._hud_dot, x + px(13) - r, y + px(13) - r, x + px(13) + r, y + px(13) + r)
        self.cv.itemconfigure(self._hud_dot, fill=color, state="normal")
        for it in (self._hud_dot, self._hud_text):
            self.cv.tag_raise(it)

    # ---- outline a sample with the mouse
    def _sel_press(self, e) -> None:
        if self._df is None:
            return
        self._sel_start = (e.x, e.y)
        self.cv.coords(self._sel_rect, e.x, e.y, e.x, e.y)
        self.cv.itemconfigure(self._sel_rect, state="normal")

    def _sel_move(self, e) -> None:
        if self._sel_start is None:
            return
        x0, y0 = self._sel_start
        self.cv.coords(self._sel_rect, x0, y0, e.x, e.y)

    def _sel_release(self, e) -> None:
        start, self._sel_start = self._sel_start, None
        self.root.after(600, lambda: self.cv.itemconfigure(self._sel_rect, state="hidden"))
        df = self._df
        if start is None or df is None or df.rect is None:
            return
        x0, y0 = min(start[0], e.x), min(start[1], e.y)
        x1, y1 = max(start[0], e.x), max(start[1], e.y)
        ix, iy, iw, ih = self._img_geom
        if x1 - x0 < 3 and y1 - y0 < 3:          # a click: small box around the point
            x0, y0, x1, y1 = x0 - 3, y0 - 3, x1 + 3, y1 + 3
        s = df.scale
        fx0, fy0 = (x0 - ix) / s, (y0 - iy) / s
        fx1, fy1 = (x1 - ix) / s, (y1 - iy) / s
        if fx1 < 0 or fy1 < 0 or fx0 > iw / s or fy0 > ih / s:
            return
        box = (df.rect.left + int(fx0), df.rect.top + int(fy0), max(2, int(fx1 - fx0)), max(2, int(fy1 - fy0)))
        self.sample_msg.configure(text="запоминаю образец…", fg=MUTED)
        self.engine.request_sample(box)

    # ================================================================ actions
    def _toggle(self) -> None:
        self.engine.toggle()
        self._refresh_status()

    def _dry_changed(self) -> None:
        if self.engine.active:
            self.engine.stop("режим изменён — запустите снова")
        self._schedule_apply(0)

    def _debug_changed(self) -> None:
        self._last_seq = -1
        self._schedule_apply(0)
        self.engine.set_preview(bool(self.debug_var.get()))

    def _slider_moved(self, pos: float, final: bool) -> None:
        self.cfg.click_interval_ms = pos_to_interval(pos)
        self._show_interval(move_slider=False)
        if final:
            self._schedule_apply(0)
        else:
            self._schedule_apply(150)

    def _set_interval(self, ms: float) -> None:
        self.cfg.click_interval_ms = float(max(0.0, ms))
        self._show_interval()
        self._schedule_apply(0)

    def _show_interval(self, move_slider: bool = True) -> None:
        ms = self.cfg.click_interval_ms
        big, sub = interval_text(ms)
        self.iv_value.configure(text=big)
        self.iv_sub.configure(text=sub)
        if move_slider:
            self.iv_slider.set(interval_to_pos(ms))
        try:
            focused = self.root.focus_get() is self.iv_entry
        except (KeyError, tk.TclError):
            focused = False
        if not focused:
            self.iv_entry.set_value(ms)
        for v, lb in self.iv_chips.items():
            on = abs(v - ms) < 1e-6
            lb.configure(bg=ACCENT_SOFT if on else SURFACE2, fg=ACCENT if on else MUTED)

    def _mode_changed(self, mode: str) -> None:
        self.cfg.target.mode = mode
        self.mode_help.configure(text=MODE_HELP[mode])
        self.sample_msg.configure(text="")
        self._schedule_apply(0)

    def _reset_target(self) -> None:
        base = DetectorConfig()
        d = self.cfg.detector
        d.hue_ranges, d.expected_area, d.min_saturation = base.hue_ranges, 0.0, base.min_saturation
        d.min_area, d.max_area = base.min_area, base.max_area
        self.min_area.set_value(d.min_area)
        self.engine.reset_target()
        self.sample_msg.configure(text="цель сброшена, цвета — по умолчанию", fg=MUTED)
        self._schedule_apply(0)

    def _cursor_delayed(self) -> None:
        if self._calib_job is not None:
            self.root.after_cancel(self._calib_job)
        self.sample_msg.configure(text="наведите курсор на червя в игре… 3 с", fg=AMBER)
        self._calib_job = self.root.after(3000, self._cursor_now)

    def _cursor_now(self) -> None:
        self._calib_job = None
        self.engine.request_cursor_capture()

    def _set_area(self, which: str, v: float) -> None:
        if which == "min":
            self.cfg.detector.min_area = int(v)
        else:
            self.cfg.target.max_target_area = int(v)
            self.cfg.detector.max_area = max(self.cfg.detector.max_area, int(v))
        self._schedule_apply(0)

    def handle_action(self, action: str) -> None:
        """Hotkey actions (called from the hotkey thread - no Tk calls here)."""
        if action == "toggle":
            self.engine.toggle()
        elif action == "stop":
            self.engine.emergency_stop()
        elif action == "calibrate":
            self.engine.request_cursor_capture()

    # ================================================================ hotkeys
    def _register_hotkeys(self) -> None:
        errors = self.hotkeys.set_bindings(dict(self.hk_vals))
        for a, lb in self.hk_err.items():
            lb.configure(text=("⚠ " + errors[a]) if a in errors else "")
        if not self.hotkeys.available:
            # local fallback: keys work while this window has focus
            for a, text in self.hk_vals.items():
                try:
                    hk = parse_hotkey(text)
                except HotkeyError:
                    continue
                key = hk.text.split("+")[-1]
                if key.startswith("F") and key[1:].isdigit() and "+" not in hk.text:
                    self.root.bind(f"<{key}>", lambda e, act=a: self.handle_action(act))

    def _assign(self, action: str) -> None:
        if self._assigning is not None:
            self._end_assign()
        self._assigning = action
        self.hk_chips[action].configure(text="нажмите…", bg=ACCENT_SOFT, fg=ACCENT)
        self.hk_err[action].configure(text="Нажмите новую клавишу или сочетание (Esc — отмена)", fg=MUTED)
        self.root.bind("<KeyPress>", self._on_key_assign)
        self.root.focus_force()

    def _end_assign(self) -> None:
        act, self._assigning = self._assigning, None
        self.root.unbind("<KeyPress>")
        if act is not None:
            self.hk_chips[act].configure(text=self.hk_vals[act], bg=SURFACE3, fg=TEXT)

    def _on_key_assign(self, ev) -> None:
        act = self._assigning
        if act is None:
            return
        if ev.keysym == "Escape":
            self.hk_err[act].configure(text="")
            self._end_assign()
            return
        text = hotkey_from_tk(ev.keysym, ev.state)
        if text is None:
            return  # modifier alone - wait for the main key
        if text in {v for a, v in self.hk_vals.items() if a != act}:
            self.hk_err[act].configure(text=f"⚠ {text} уже используется", fg=AMBER)
            return
        self.hk_vals[act] = text
        self.hk_err[act].configure(text="", fg=AMBER)
        self._end_assign()
        self._register_hotkeys()
        self._update_keys_hint()
        self.start_btn.update_button(hint=self.hk_vals["toggle"])
        self._schedule_apply(0)

    def _update_keys_hint(self) -> None:
        v = self.hk_vals
        self.keys_hint.configure(text=f"{v['toggle']} — старт/стоп   ·   {v['stop']} — стоп   ·   "
                                      f"{v['calibrate']} — цель под курсором")

    # ================================================================ config
    def _area_options(self) -> list[tuple[str, str]]:
        opts = [("window", "Окно браузера (активное)")]
        try:
            mons = list_monitors()
        except Exception:
            mons = []
        for i, m in enumerate(mons):
            opts.append((f"monitor:{i}", f"Монитор {i + 1} — {m.width}×{m.height} ({m.left}, {m.top})"))
        opts.append(("all", "Все мониторы"))
        if self.cfg.capture_area not in {k for k, _ in opts}:
            opts.append((self.cfg.capture_area, self.cfg.capture_area))
        return opts

    def _schedule_apply(self, delay: int = 300) -> None:
        if self._closing:
            return
        if self._apply_job is not None:
            self.root.after_cancel(self._apply_job)
        self._apply_job = self.root.after(delay, self._apply)

    def _read_config(self) -> AppConfig:
        cfg = copy.deepcopy(self.cfg)
        cfg.hotkey_toggle = self.hk_vals["toggle"]
        cfg.hotkey_stop = self.hk_vals["stop"]
        cfg.hotkey_calibrate = self.hk_vals["calibrate"]
        cfg.capture_area = self.area_var.get()
        cfg.capture_backend = self.backend_var.get()
        cfg.require_focus = bool(self.focus_var.get())
        cfg.dry_run = bool(self.dry_var.get())
        cfg.debug_view = bool(self.debug_var.get())
        cfg.preview_fps = int(self.fps_var.get())
        cfg.target.mode = self.mode_var.get()
        cfg.validate()
        return cfg

    def _apply(self) -> None:
        self._apply_job = None
        cfg = self._read_config()
        self.cfg = copy.deepcopy(cfg)
        self.engine.apply_config(cfg)
        save_config(cfg)

    # ================================================================ periodic
    def _post_map(self) -> None:
        winapi.exclude_from_capture(self.root.winfo_id())

    def _own_rects(self) -> list[Rect]:
        try:
            if not self.root.winfo_viewable():
                return []
            return [Rect(self.root.winfo_rootx() - 8, self.root.winfo_rooty() - 40,
                         self.root.winfo_width() + 16, self.root.winfo_height() + 48)]
        except tk.TclError:
            return []

    def _refresh_status(self) -> None:
        e = self.engine
        text = e.status.value + (" · тест" if e.active and self.cfg.dry_run else "")
        self.pill.set(text, STATUS_COLORS.get(e.status, MUTED))
        self.msg.configure(text=e.message, fg=RED if e.status == Status.ERROR else MUTED)
        if e.active:
            self.start_btn.update_button(text="■   Стоп", kind="danger")
        else:
            self.start_btn.update_button(text="▶   Старт", kind="primary")

    def _tick(self) -> None:
        if self._closing:
            return
        try:
            while True:
                kind, payload = self._events.get_nowait()
                if kind == "sample":
                    self._on_sample(payload)
        except queue.Empty:
            pass
        try:
            self._refresh_status()
            out = format_stats(self.engine.stats.snapshot())
            for k, lb in self.tile_vals.items():
                if lb.cget("text") != out[k]:
                    lb.configure(text=out[k])
            self.detail.configure(text=out["detail"])
            self._refresh_info()
            self.engine.set_exclusions(self._own_rects())
        except tk.TclError:  # pragma: no cover - window is closing
            return
        self.root.after(250, self._tick)

    def _refresh_info(self) -> None:
        e = self.engine
        rect = e.region_rect
        reg = f"{rect.width}×{rect.height} в ({rect.left}, {rect.top})" if rect else "—"
        title = e.region.title[:48] if e.region.title else ""
        self.capture_info.configure(text=f"Захват: {e.capture_name} · область {reg}" +
                                    (f"\nОкно: {title}" if title and self.cfg.capture_area == "window" else ""))
        n = len(self._frame_times)
        self.fps_lbl.configure(text=f"{n} fps" if self.debug_var.get() and n else "")
        self.preview_info.configure(text=f"{e.capture_name} · {reg}" if rect else (e.target_message or ""))
        m = e.finder.model
        if m is None:
            text = "не выбрана — " + ("жду движения" if self.cfg.target.mode == "motion" else "нужен образец")
        else:
            src = "по движению" if m.source == "motion" else "по образцу"
            text = f"запомнена {src} · ≈ {m.area:.0f} px · прыжков {m.jumps}"
        if self.target_info.cget("text") != text:
            self.target_info.configure(text=text)

    def _on_sample(self, res: SampleResult) -> None:
        self.sample_msg.configure(text=("✔ " if res.ok else "⚠ ") + res.message, fg=GREEN if res.ok else AMBER)
        if res.ok:
            # the finder calibrated the colours and switched to the sample mode
            self.cfg.detector = copy.deepcopy(self.engine.cfg.detector)
            self.cfg.target.mode = "sample"
            self.mode_var.set("sample")
            self.mode_help.configure(text=MODE_HELP["sample"])
            self.min_area.set_value(self.cfg.detector.min_area)
            self._schedule_apply(0)

    # ================================================================ close
    def close(self) -> None:
        if self._closing:
            return
        try:
            if self._apply_job is not None:
                self.root.after_cancel(self._apply_job)
            self._apply()
        except Exception:
            log.exception("save on close failed")
        self._closing = True
        self.engine.shutdown()
        self.hotkeys.stop()
        self.root.destroy()
