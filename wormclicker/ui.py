"""Compact dark Tkinter interface.

The UI thread never touches capture/detection: it only reads stats
snapshots and the latest debug frame a few times per second, so it can not
slow down the click loop, and the click loop can not freeze the UI.
"""

from __future__ import annotations

import logging
import queue
import tkinter as tk
from dataclasses import replace
from tkinter import ttk

from . import __version__, winapi
from .automation import AutomationEngine, Status
from .calibration import CalibrationResult
from .capture import list_monitors
from .config import AppConfig, DetectorConfig, save_config
from .geometry import Rect
from .hotkeys import HotkeyManager, hotkey_from_tk, parse_hotkey, HotkeyError
from .stats import StatsSnapshot

log = logging.getLogger(__name__)

BG = "#14161c"
PANEL = "#1d2029"
FG = "#e6e6ea"
MUTED = "#8b8fa3"
ACCENT = "#ff6fae"
STATUS_COLORS = {
    Status.IDLE: "#8b8fa3",
    Status.PAUSED: "#e0b84c",
    Status.SEARCHING: "#4cb3e0",
    Status.ACTIVE: "#5fd46a",
    Status.ERROR: "#f05a5a",
}


def format_stats(s: StatsSnapshot) -> dict[str, str]:
    """Pure formatting (unit-tested without a display)."""
    def ms(v: float) -> str:
        return f"{v:.1f}" if v > 0 else "—"

    hit_rate = f"{100.0 * s.jumps / s.clicks:.0f}%" if s.clicks else "—"
    return {
        "clicks": str(s.clicks),
        "jumps": f"{s.jumps} ({hit_rate})",
        "cps": f"{s.cps:.1f}",
        "jumps_ps": f"{s.jumps_ps:.1f}",
        "avg_cps": f"{s.avg_cps:.1f}",
        "fps": f"{s.fps:.0f}",
        "capture_ms": ms(s.capture_ms),
        "detect_ms": ms(s.detect_ms),
        "pipeline_ms": ms(s.pipeline_ms),
        "jump_ms": ms(s.jump_ms),
        "lost": str(s.lost),
        "reacquired": f"{s.reacquired} (ср. {ms(s.reacquire_ms)} мс)",
        "no_jump": str(s.no_jump),
        "stale": str(s.stale_frames),
        "blacklisted": str(s.blacklisted),
        "errors": str(s.errors),
        "session": f"{int(s.session_s // 60)}:{int(s.session_s % 60):02d}",
    }


STAT_ROWS = [
    ("clicks", "Клики отправлено"),
    ("jumps", "Прыжки после клика (≈ попадания)"),
    ("cps", "Кликов/с сейчас"),
    ("jumps_ps", "Прыжков/с сейчас"),
    ("avg_cps", "Средний кликов/с"),
    ("fps", "Кадров/с обработано"),
    ("capture_ms", "Захват кадра, мс"),
    ("detect_ms", "Детекция, мс"),
    ("pipeline_ms", "Кадр → клик, мс"),
    ("jump_ms", "Клик → новая позиция, мс"),
    ("lost", "Потерь цели"),
    ("reacquired", "Повторных обнаружений"),
    ("no_jump", "Клик без прыжка (промах?)"),
    ("stale", "Устаревших кадров пропущено"),
    ("blacklisted", "Статичных ложных целей"),
    ("errors", "Ошибок захвата"),
    ("session", "Время сессии"),
]


class App:
    def __init__(self, root: tk.Tk, engine: AutomationEngine, hotkeys: HotkeyManager, cfg: AppConfig):
        self.root = root
        self.engine = engine
        self.hotkeys = hotkeys
        self.cfg = cfg
        self._events: queue.SimpleQueue = queue.SimpleQueue()
        self._apply_job: str | None = None
        self._assigning: str | None = None
        self._debug_win: tk.Toplevel | None = None
        self._debug_img: tk.PhotoImage | None = None
        self._last_debug_t = 0.0
        self._closing = False

        engine.on_calibration = lambda res: self._events.put(("calib", res))
        root.title(f"Worm Clicker {__version__}")
        root.configure(bg=BG)
        root.minsize(440, 600)
        self._style()
        self._build()
        self._register_hotkeys()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(300, self._post_map)
        root.after(250, self._tick)

    # ------------------------------------------------------------------ style
    def _style(self) -> None:
        st = ttk.Style(self.root)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", background=BG, foreground=FG, fieldbackground=PANEL, bordercolor="#2a2e3a",
                     lightcolor=PANEL, darkcolor=PANEL, troughcolor=PANEL, font=("Segoe UI", 9))
        st.configure("TFrame", background=BG)
        st.configure("Panel.TFrame", background=PANEL)
        st.configure("TLabel", background=BG, foreground=FG)
        st.configure("Panel.TLabel", background=PANEL, foreground=FG)
        st.configure("Muted.TLabel", background=PANEL, foreground=MUTED)
        st.configure("Value.TLabel", background=PANEL, foreground=FG, font=("Consolas", 10, "bold"))
        st.configure("Title.TLabel", background=BG, foreground=ACCENT, font=("Segoe UI", 14, "bold"))
        st.configure("Status.TLabel", background=BG, foreground=FG, font=("Segoe UI", 11, "bold"))
        st.configure("Msg.TLabel", background=BG, foreground=MUTED)
        st.configure("TCheckbutton", background=PANEL, foreground=FG)
        st.map("TCheckbutton", background=[("active", PANEL)])
        st.configure("TButton", background="#2a2e3a", foreground=FG, padding=(8, 3))
        st.map("TButton", background=[("active", "#353a49")])
        st.configure("Start.TButton", background="#2f7d3a", foreground="white", font=("Segoe UI", 11, "bold"),
                     padding=(10, 8))
        st.map("Start.TButton", background=[("active", "#3a9447")])
        st.configure("Stop.TButton", background="#a33a3a", foreground="white", font=("Segoe UI", 11, "bold"),
                     padding=(10, 8))
        st.map("Stop.TButton", background=[("active", "#bd4646")])
        st.configure("TLabelframe", background=PANEL, bordercolor="#2a2e3a")
        st.configure("TLabelframe.Label", background=PANEL, foreground=ACCENT, font=("Segoe UI", 9, "bold"))
        st.configure("TCombobox", fieldbackground=PANEL, background="#2a2e3a", foreground=FG,
                     arrowcolor=FG)
        st.configure("TEntry", fieldbackground=PANEL, foreground=FG, insertcolor=FG)
        st.configure("TSpinbox", fieldbackground=PANEL, foreground=FG, arrowcolor=FG)
        st.configure("Horizontal.TScale", background=PANEL)
        self.root.option_add("*TCombobox*Listbox.background", PANEL)
        self.root.option_add("*TCombobox*Listbox.foreground", FG)

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        r = self.root
        top = ttk.Frame(r, padding=(10, 8, 10, 4))
        top.pack(fill="x")
        ttk.Label(top, text="WORM CLICKER", style="Title.TLabel").pack(side="left")
        self.status_dot = tk.Canvas(top, width=14, height=14, bg=BG, highlightthickness=0)
        self.status_dot.pack(side="right", padx=(4, 0))
        self._dot = self.status_dot.create_oval(2, 2, 12, 12, fill=MUTED, outline="")
        self.status_var = tk.StringVar(value=Status.IDLE.value)
        ttk.Label(top, textvariable=self.status_var, style="Status.TLabel").pack(side="right")
        self.msg_var = tk.StringVar(value=self.engine.message)
        ttk.Label(r, textvariable=self.msg_var, style="Msg.TLabel", wraplength=420,
                  padding=(10, 0, 10, 4)).pack(fill="x")

        ctl = ttk.Frame(r, padding=(10, 2))
        ctl.pack(fill="x")
        self.start_btn = ttk.Button(ctl, text="▶  Старт (F6)", style="Start.TButton", command=self._toggle)
        self.start_btn.pack(side="left", fill="x", expand=True)
        self.dry_var = tk.BooleanVar(value=self.cfg.dry_run)
        cb = tk.Checkbutton(ctl, text="Тест без кликов", variable=self.dry_var, command=self._dry_changed,
                            bg=BG, fg=FG, selectcolor=PANEL, activebackground=BG, activeforeground=FG)
        cb.pack(side="left", padx=(8, 0))

        nb = ttk.Notebook(r)
        nb.pack(fill="both", expand=True, padx=10, pady=8)
        nb.add(self._build_stats(nb), text="Статистика")
        nb.add(self._build_control(nb), text="Управление")
        nb.add(self._build_detector(nb), text="Детектор")

        foot = ttk.Label(r, style="Msg.TLabel", wraplength=420, padding=(10, 0, 10, 8),
                         text="«Попадания» оцениваются по прыжку червя после клика и являются "
                              "приблизительными. Точное значение — счётчик самой игры.")
        foot.pack(fill="x")

    def _build_stats(self, parent) -> ttk.Frame:
        f = ttk.Frame(parent, style="Panel.TFrame", padding=10)
        self.stat_vars: dict[str, tk.StringVar] = {}
        for i, (key, title) in enumerate(STAT_ROWS):
            ttk.Label(f, text=title, style="Muted.TLabel").grid(row=i, column=0, sticky="w", pady=1)
            v = tk.StringVar(value="—")
            self.stat_vars[key] = v
            ttk.Label(f, textvariable=v, style="Value.TLabel").grid(row=i, column=1, sticky="e", pady=1)
        f.columnconfigure(0, weight=1)
        n = len(STAT_ROWS)
        self.info_var = tk.StringVar(value="")
        ttk.Label(f, textvariable=self.info_var, style="Muted.TLabel", wraplength=400).grid(
            row=n, column=0, columnspan=2, sticky="w", pady=(6, 0))
        btns = ttk.Frame(f, style="Panel.TFrame")
        btns.grid(row=n + 1, column=0, columnspan=2, sticky="we", pady=(6, 0))
        self.debug_var = tk.BooleanVar(value=self.cfg.debug_view)
        ttk.Checkbutton(btns, text="Визуальная отладка", variable=self.debug_var,
                        command=self._debug_changed).pack(side="left")
        ttk.Button(btns, text="Сбросить", command=self.engine.stats.reset).pack(side="right")
        return f

    def _build_control(self, parent) -> ttk.Frame:
        f = ttk.Frame(parent, style="Panel.TFrame", padding=10)
        self.hk_vars: dict[str, tk.StringVar] = {}
        self.hk_err: dict[str, tk.StringVar] = {}
        rows = [("toggle", "Старт / стоп", self.cfg.hotkey_toggle),
                ("stop", "Аварийная остановка", self.cfg.hotkey_stop),
                ("calibrate", "Калибровка по курсору", self.cfg.hotkey_calibrate)]
        for i, (act, title, val) in enumerate(rows):
            ttk.Label(f, text=title, style="Panel.TLabel").grid(row=2 * i, column=0, sticky="w")
            v = tk.StringVar(value=val)
            self.hk_vars[act] = v
            e = ttk.Entry(f, textvariable=v, width=14, state="readonly")
            e.grid(row=2 * i, column=1, padx=4)
            ttk.Button(f, text="Назначить", command=lambda a=act: self._assign(a)).grid(row=2 * i, column=2)
            ev = tk.StringVar(value="")
            self.hk_err[act] = ev
            ttk.Label(f, textvariable=ev, style="Muted.TLabel", wraplength=380).grid(
                row=2 * i + 1, column=0, columnspan=3, sticky="w")
        row = 7
        ttk.Label(f, text="Область поиска", style="Panel.TLabel").grid(row=row, column=0, sticky="w", pady=(8, 0))
        self.area_var = tk.StringVar()
        self.area_box = ttk.Combobox(f, textvariable=self.area_var, state="readonly", width=30)
        self.area_box.grid(row=row, column=1, columnspan=2, sticky="we", pady=(8, 0))
        self._fill_areas()
        self.area_box.bind("<<ComboboxSelected>>", lambda e: self._schedule_apply())
        row += 1
        ttk.Label(f, text="Захват экрана", style="Panel.TLabel").grid(row=row, column=0, sticky="w", pady=(4, 0))
        self.backend_var = tk.StringVar(value=self.cfg.capture_backend)
        bb = ttk.Combobox(f, textvariable=self.backend_var, state="readonly", width=12,
                          values=["auto", "dxcam", "mss"])
        bb.grid(row=row, column=1, sticky="w", pady=(4, 0))
        bb.bind("<<ComboboxSelected>>", lambda e: self._schedule_apply())
        row += 1
        self.focus_var = tk.BooleanVar(value=self.cfg.require_focus)
        ttk.Checkbutton(f, text="Пауза, если окно браузера не в фокусе", variable=self.focus_var,
                        command=self._schedule_apply).grid(row=row, column=0, columnspan=3, sticky="w", pady=(6, 0))
        row += 1
        ttk.Label(f, style="Muted.TLabel", wraplength=400, text=(
            "Режим «окно браузера»: откройте игру, нажмите F6 в окне браузера — будет сканироваться "
            "его клиентская область. Перемещение и изменение размера окна отслеживаются.")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(8, 0))
        f.columnconfigure(1, weight=1)
        return f

    def _build_detector(self, parent) -> ttk.Frame:
        f = ttk.Frame(parent, style="Panel.TFrame", padding=10)
        d, t = self.cfg.detector, self.cfg.tracker
        self.det_vars: dict[str, tk.Variable] = {}

        def add(row, key, title, var, widget):
            ttk.Label(f, text=title, style="Panel.TLabel").grid(row=row, column=0, sticky="w", pady=2)
            widget.grid(row=row, column=1, sticky="we", pady=2)
            self.det_vars[key] = var
            var.trace_add("write", lambda *_: self._schedule_apply())

        conf = tk.DoubleVar(value=d.confidence_threshold)
        self.conf_lbl = tk.StringVar(value=f"{d.confidence_threshold:.2f}")
        add(0, "conf", "Порог уверенности", conf,
            ttk.Scale(f, from_=0.3, to=0.95, variable=conf, orient="horizontal",
                      command=lambda v: self.conf_lbl.set(f"{float(v):.2f}")))
        ttk.Label(f, textvariable=self.conf_lbl, style="Value.TLabel").grid(row=0, column=2, padx=4)

        def spin(frm, to, inc, var, width=8):
            return ttk.Spinbox(f, from_=frm, to=to, increment=inc, textvariable=var, width=width)

        rows = [
            (1, "min_area", "Мин. площадь червя, px", tk.IntVar(value=d.min_area), (10, 100000, 10)),
            (2, "max_area", "Макс. площадь червя, px", tk.IntVar(value=d.max_area), (100, 1000000, 100)),
            (3, "downscale", "Шаг грубого поиска (1–4)", tk.IntVar(value=d.downscale), (1, 4, 1)),
            (4, "stable", "Кадров подтверждения", tk.IntVar(value=t.stable_frames), (1, 10, 1)),
            (5, "stale", "Ожидание прыжка, мс", tk.IntVar(value=int(t.stale_timeout_ms)), (10, 5000, 10)),
            (6, "max_cps", "Лимит кликов/с (0 = нет)", tk.DoubleVar(value=t.max_cps), (0, 1000, 1)),
            (7, "idle", "Пауза поиска без цели, мс", tk.DoubleVar(value=self.cfg.idle_sleep_ms), (0, 100, 1)),
        ]
        for row, key, title, var, (lo, hi, inc) in rows:
            add(row, key, title, var, spin(lo, hi, inc, var))

        b = ttk.Frame(f, style="Panel.TFrame")
        b.grid(row=8, column=0, columnspan=3, sticky="we", pady=(10, 0))
        ttk.Button(b, text="Калибровать по курсору (3 с)", command=self._calibrate_delayed).pack(side="left")
        ttk.Button(b, text="Сбросить калибровку", command=self._reset_calibration).pack(side="right")
        self.calib_var = tk.StringVar(value=self._calib_text())
        ttk.Label(f, textvariable=self.calib_var, style="Muted.TLabel", wraplength=400).grid(
            row=9, column=0, columnspan=3, sticky="w", pady=(6, 0))
        f.columnconfigure(1, weight=1)
        return f

    def _calib_text(self) -> str:
        d = self.cfg.detector
        if d.expected_area > 0:
            return f"Калибровка: оттенки {d.hue_ranges}, площадь ≈ {d.expected_area:.0f} px"
        return ("Калибровка не выполнялась (используются стандартные цвета). Наведите курсор на "
                "червя и нажмите F8, чтобы подстроить цвета и размер под вашу игру.")

    # -------------------------------------------------------------- areas
    def _fill_areas(self) -> None:
        self._areas = [("window", "Окно браузера (активное при старте)"), ("all", "Все мониторы")]
        try:
            mons = list_monitors()
        except Exception:
            mons = []
        for i, m in enumerate(mons):
            self._areas.append((f"monitor:{i}", f"Монитор {i + 1}: {m.width}x{m.height} ({m.left},{m.top})"))
        self.area_box["values"] = [t for _, t in self._areas]
        cur = next((t for k, t in self._areas if k == self.cfg.capture_area), self._areas[0][1])
        self.area_var.set(cur)

    def _area_key(self) -> str:
        txt = self.area_var.get()
        return next((k for k, t in self._areas if t == txt), "window")

    # -------------------------------------------------------------- actions
    def _toggle(self) -> None:
        self.engine.toggle()
        self._refresh_status()

    def _dry_changed(self) -> None:
        if self.engine.active:
            self.engine.stop("режим изменён — запустите снова")
        self._schedule_apply(0)

    def _debug_changed(self) -> None:
        self._schedule_apply(0)
        if self.debug_var.get():
            self._open_debug()
        elif self._debug_win is not None:
            self._debug_win.destroy()
            self._debug_win = None

    def _calibrate_delayed(self) -> None:
        self.calib_var.set("Наведите курсор на червя… калибровка через 3 с")
        self.root.after(3000, self.engine.request_calibration)

    def _reset_calibration(self) -> None:
        base = DetectorConfig()
        d = self.cfg.detector
        d.hue_ranges, d.expected_area = base.hue_ranges, 0.0
        d.min_area, d.max_area, d.min_saturation = base.min_area, base.max_area, base.min_saturation
        self.det_vars["min_area"].set(d.min_area)
        self.det_vars["max_area"].set(d.max_area)
        self._schedule_apply(0)
        self.calib_var.set(self._calib_text())

    def handle_action(self, action: str) -> None:
        """Hotkey actions (called from the hotkey thread)."""
        if action == "toggle":
            self.engine.toggle()
        elif action == "stop":
            self.engine.emergency_stop()
        elif action == "calibrate":
            self.engine.request_calibration()

    # -------------------------------------------------------------- hotkeys
    def _register_hotkeys(self) -> None:
        bindings = {a: v.get() for a, v in self.hk_vars.items()}
        errors = self.hotkeys.set_bindings(bindings)
        for a, v in self.hk_err.items():
            v.set(("⚠ " + errors[a]) if a in errors else "")
        if not self.hotkeys.available:
            # local fallback: keys work while this window has focus
            for a, text in bindings.items():
                try:
                    hk = parse_hotkey(text)
                except HotkeyError:
                    continue
                key = hk.text.split("+")[-1]
                if key.startswith("F") and key[1:].isdigit() and "+" not in hk.text:
                    self.root.bind(f"<{key}>", lambda e, act=a: self.handle_action(act))

    def _assign(self, action: str) -> None:
        self._assigning = action
        self.hk_err[action].set("Нажмите новую клавишу (Esc — отмена)…")
        self.root.bind("<KeyPress>", self._on_key_assign)
        self.root.focus_force()

    def _on_key_assign(self, ev) -> None:
        act = self._assigning
        if act is None:
            return
        if ev.keysym == "Escape":
            self.hk_err[act].set("")
        else:
            text = hotkey_from_tk(ev.keysym, ev.state)
            if text is None:
                return  # modifier alone - wait for the main key
            others = {a: v.get() for a, v in self.hk_vars.items() if a != act}
            if text in others.values():
                self.hk_err[act].set(f"⚠ {text} уже используется")
                return
            self.hk_vars[act].set(text)
        self._assigning = None
        self.root.unbind("<KeyPress>")
        self._register_hotkeys()
        self._schedule_apply(0)
        self.start_btn.configure(text=self._start_text())

    def _start_text(self) -> str:
        key = self.hk_vars["toggle"].get()
        return f"■  Стоп ({key})" if self.engine.active else f"▶  Старт ({key})"

    # -------------------------------------------------------------- config
    def _schedule_apply(self, delay: int = 300) -> None:
        if self._apply_job is not None:
            self.root.after_cancel(self._apply_job)
        self._apply_job = self.root.after(delay, self._apply)

    def _read_config(self) -> AppConfig:
        cfg = replace(self.cfg, detector=replace(self.cfg.detector), tracker=replace(self.cfg.tracker))
        g = self.det_vars

        def num(key, cast, default):
            try:
                return cast(g[key].get())
            except (tk.TclError, ValueError):
                return default

        cfg.detector.confidence_threshold = round(num("conf", float, cfg.detector.confidence_threshold), 3)
        cfg.detector.min_area = num("min_area", int, cfg.detector.min_area)
        cfg.detector.max_area = num("max_area", int, cfg.detector.max_area)
        cfg.detector.downscale = num("downscale", int, cfg.detector.downscale)
        cfg.tracker.stable_frames = num("stable", int, cfg.tracker.stable_frames)
        cfg.tracker.stale_timeout_ms = num("stale", float, cfg.tracker.stale_timeout_ms)
        cfg.tracker.max_cps = num("max_cps", float, cfg.tracker.max_cps)
        cfg.idle_sleep_ms = num("idle", float, cfg.idle_sleep_ms)
        cfg.hotkey_toggle = self.hk_vars["toggle"].get()
        cfg.hotkey_stop = self.hk_vars["stop"].get()
        cfg.hotkey_calibrate = self.hk_vars["calibrate"].get()
        cfg.capture_area = self._area_key()
        cfg.capture_backend = self.backend_var.get()
        cfg.require_focus = bool(self.focus_var.get())
        cfg.dry_run = bool(self.dry_var.get())
        cfg.debug_view = bool(self.debug_var.get())
        cfg.validate()
        return cfg

    def _apply(self) -> None:
        self._apply_job = None
        cfg = self._read_config()
        self.cfg = cfg
        self.engine.apply_config(cfg)
        save_config(cfg)

    # -------------------------------------------------------------- periodic
    def _post_map(self) -> None:
        winapi.exclude_from_capture(self.root.winfo_id())
        if self.debug_var.get():
            self._open_debug()

    def _own_rects(self) -> list[Rect]:
        out = []
        for w in (self.root, self._debug_win):
            if w is None:
                continue
            try:
                if w.winfo_viewable():
                    out.append(Rect(w.winfo_rootx() - 8, w.winfo_rooty() - 40,
                                    w.winfo_width() + 16, w.winfo_height() + 48))
            except tk.TclError:
                pass
        return out

    def _refresh_status(self) -> None:
        e = self.engine
        self.status_var.set(e.status.value + (" (тест)" if e.active and self.cfg.dry_run else ""))
        self.status_dot.itemconfigure(self._dot, fill=STATUS_COLORS.get(e.status, MUTED))
        self.msg_var.set(e.message)
        self.start_btn.configure(text=self._start_text(), style="Stop.TButton" if e.active else "Start.TButton")

    def _tick(self) -> None:
        if self._closing:
            return
        try:
            while True:
                kind, payload = self._events.get_nowait()
                if kind == "calib":
                    self._on_calibration(payload)
        except queue.Empty:
            pass
        self._refresh_status()
        snap = self.engine.stats.snapshot()
        for k, v in format_stats(snap).items():
            var = self.stat_vars.get(k)
            if var is not None and var.get() != v:
                var.set(v)
        rect = self.engine.region_rect
        reg = f"{rect.width}x{rect.height} @ ({rect.left},{rect.top})" if rect else "—"
        mode = "ТЕСТ (клики не отправляются)" if self.cfg.dry_run else "реальные клики"
        title = f" · окно: {self.engine.region.title[:40]}" if self.engine.region.title else ""
        self.info_var.set(f"Захват: {self.engine.capture_name} · область: {reg}{title} · {mode}")
        self.engine.set_exclusions(self._own_rects())
        if self._debug_win is not None:
            self._update_debug()
        self.root.after(250, self._tick)

    def _on_calibration(self, res: CalibrationResult) -> None:
        self.calib_var.set(("✔ " if res.ok else "⚠ ") + res.message)
        if res.ok:
            d = self.engine.cfg.detector
            self.cfg.detector = d
            self.det_vars["min_area"].set(d.min_area)
            self.det_vars["max_area"].set(d.max_area)
            self._schedule_apply(0)

    # -------------------------------------------------------------- debug
    def _open_debug(self) -> None:
        if self._debug_win is not None:
            return
        w = tk.Toplevel(self.root)
        w.title("Worm Clicker — отладка")
        w.configure(bg=BG)
        self._debug_label = tk.Label(w, bg=BG, fg=MUTED, text="нет кадров — запустите поиск (F6)")
        self._debug_label.pack(padx=6, pady=6)
        self._debug_text = tk.Label(w, bg=BG, fg=FG, justify="left", anchor="w", font=("Consolas", 9))
        self._debug_text.pack(fill="x", padx=6, pady=(0, 6))
        w.protocol("WM_DELETE_WINDOW", lambda: (self.debug_var.set(False), self._debug_changed()))
        self._debug_win = w
        w.after(200, lambda: winapi.exclude_from_capture(w.winfo_id()))

    def _update_debug(self) -> None:
        df = self.engine.debug_frame
        if df is None or df.t == self._last_debug_t:
            return
        self._last_debug_t = df.t
        from .debug_view import annotate, to_png_base64

        try:
            img, lines = annotate(df)
            self._debug_img = tk.PhotoImage(data=to_png_base64(img))
            self._debug_label.configure(image=self._debug_img, text="")
            self._debug_text.configure(text="\n".join(lines[:10]))
        except Exception as exc:  # pragma: no cover
            self._debug_text.configure(text=f"ошибка отладки: {exc}")

    # -------------------------------------------------------------- close
    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        try:
            self._apply()
        except Exception:
            log.exception("save on close failed")
        self.engine.shutdown()
        self.hotkeys.stop()
        self.root.destroy()
