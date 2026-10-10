"""Small custom Tk widgets for the dark UI.

Plain ``tk`` widgets and canvases only: every colour is set explicitly, so
there is no theme-dependent "white text on a light-grey field" anywhere
(the classic problem of ttk entries/comboboxes in the readonly state).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from typing import Callable, Sequence

# ---------------------------------------------------------------- palette
BG = "#0d0f14"
SURFACE = "#151821"
SURFACE2 = "#1c202b"
SURFACE3 = "#262b39"
HOVER = "#2e3445"
BORDER = "#252a37"
TEXT = "#eef0f6"
MUTED = "#a6adbf"
DIM = "#727a8f"
ACCENT = "#ff5fa2"
ACCENT_DEEP = "#e2407f"
ACCENT_SOFT = "#3b1a2c"
GREEN = "#3ddc97"
AMBER = "#ffb547"
BLUE = "#5aa9ff"
RED = "#ff5c6c"
RED_DEEP = "#e43b50"
PREVIEW_BG = "#07080b"


def mix(c1: str, c2: str, t: float) -> str:
    """Blend two #rrggbb colours (t=0 -> c1, t=1 -> c2)."""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(round(x + (y - x) * t)):02x}" for x, y in zip(a, b, strict=True))


def round_rect(cv: tk.Canvas, x0: float, y0: float, x1: float, y1: float, r: float, **kw) -> int:
    r = max(0.0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1, x1 - r, y1,
           x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
    return cv.create_polygon(pts, smooth=True, splinesteps=12, **kw)


class Scale:
    """Pixel scaling for high-DPI screens (Tk scales fonts, not pixel sizes)."""

    k = 1.0

    @classmethod
    def init(cls, root: tk.Misc) -> None:
        try:
            cls.k = max(1.0, min(3.0, float(root.winfo_fpixels("1i")) / 96.0))
        except tk.TclError:
            cls.k = 1.0


def px(v: float) -> int:
    return int(round(v * Scale.k))


# ---------------------------------------------------------------- button
class RoundButton(tk.Canvas):
    KINDS = {
        # normal, hover, pressed, text, hint
        "primary": (ACCENT_DEEP, "#ee4f8e", "#c8306c", "#ffffff", "#ffd0e3"),
        "danger": (RED_DEEP, "#ef4c60", "#c42b3f", "#ffffff", "#ffd5da"),
        "secondary": (SURFACE3, HOVER, SURFACE2, TEXT, MUTED),
        "accent-outline": (ACCENT_SOFT, "#4a2138", "#2e1522", "#ffc2dc", ACCENT),
    }

    def __init__(self, master: tk.Misc, text: str = "", command: Callable[[], None] | None = None,
                 kind: str = "secondary", height: int = 34, width: int = 120, font=None,
                 hint: str | None = None, radius: int = 10, bg: str | None = None):
        super().__init__(master, height=px(height), width=px(width), bg=bg or master.cget("bg"),
                         highlightthickness=0, bd=0, cursor="hand2", takefocus=0)
        self._text, self._kind, self._hint = text, kind, hint
        self._cmd = command
        self._font = font
        self._hint_font = None
        if hint is not None or font is not None:
            base = tkfont.Font(font=font) if font is not None else tkfont.nametofont("TkDefaultFont")
            self._hint_font = (base.actual("family"), max(8, int(abs(base.actual("size")) * 0.72)), "bold")
        self._r = px(radius)
        self._hover = self._pressed = False
        self._enabled = True
        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<Enter>", lambda e: self._set(hover=True))
        self.bind("<Leave>", lambda e: self._set(hover=False, pressed=False))
        self.bind("<ButtonPress-1>", lambda e: self._set(pressed=True))
        self.bind("<ButtonRelease-1>", self._release)

    def _set(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self, "_" + k, v)
        self._draw()

    def _release(self, ev) -> None:
        was = self._pressed
        self._set(pressed=False)
        inside = 0 <= ev.x < self.winfo_width() and 0 <= ev.y < self.winfo_height()
        if was and inside and self._enabled and self._cmd is not None:
            self._cmd()

    def update_button(self, text: str | None = None, kind: str | None = None, hint: str | None = None) -> None:
        if text is not None:
            self._text = text
        if kind is not None:
            self._kind = kind
        if hint is not None:
            self._hint = hint
        self._draw()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self.configure(cursor="hand2" if enabled else "arrow")
        self._draw()

    def _draw(self) -> None:
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 4 or h < 4:
            return
        normal, hover, pressed, fg, hint_fg = self.KINDS.get(self._kind, self.KINDS["secondary"])
        fill = pressed if self._pressed else hover if self._hover else normal
        if not self._enabled:
            fill, fg, hint_fg = SURFACE2, DIM, DIM
        round_rect(self, 0, 0, w - 1, h - 1, self._r, fill=fill, outline="")
        if self._hint:
            self.create_text(w / 2, h / 2, text=self._text, fill=fg, font=self._font)
            self.create_text(w - px(16), h / 2, text=self._hint, fill=hint_fg, anchor="e", font=self._hint_font)
        else:
            self.create_text(w / 2, h / 2, text=self._text, fill=fg, font=self._font)


# ---------------------------------------------------------------- toggle
class Toggle(tk.Canvas):
    def __init__(self, master: tk.Misc, variable: tk.BooleanVar, command: Callable[[], None] | None = None,
                 bg: str | None = None):
        super().__init__(master, width=px(40), height=px(22), bg=bg or master.cget("bg"),
                         highlightthickness=0, bd=0, cursor="hand2", takefocus=0)
        self.var = variable
        self._cmd = command
        self.bind("<Button-1>", self._click)
        variable.trace_add("write", lambda *_: self._draw())
        self._draw()

    def _click(self, _ev=None) -> None:
        self.var.set(not self.var.get())
        if self._cmd is not None:
            self._cmd()

    def _draw(self) -> None:
        self.delete("all")
        w, h = px(40), px(22)
        on = bool(self.var.get())
        round_rect(self, 1, 1, w - 1, h - 1, h / 2, fill=ACCENT_DEEP if on else SURFACE3, outline="")
        r = h / 2 - px(3)
        cx = w - h / 2 if on else h / 2
        self.create_oval(cx - r, h / 2 - r, cx + r, h / 2 + r, fill="#ffffff" if on else "#c3c8d6", outline="")


class LabeledToggle(tk.Frame):
    """Toggle + clickable caption (+ optional muted hint below)."""

    def __init__(self, master: tk.Misc, text: str, variable: tk.BooleanVar,
                 command: Callable[[], None] | None = None, hint: str | None = None, font=None, hint_font=None):
        bg = master.cget("bg")
        super().__init__(master, bg=bg)
        self.toggle = Toggle(self, variable, command, bg=bg)
        self.toggle.grid(row=0, column=0, rowspan=2 if hint else 1, sticky="w", padx=(0, px(10)))
        lbl = tk.Label(self, text=text, bg=bg, fg=TEXT, font=font, cursor="hand2", anchor="w")
        lbl.grid(row=0, column=1, sticky="w")
        lbl.bind("<Button-1>", self.toggle._click)
        if hint:
            tk.Label(self, text=hint, bg=bg, fg=DIM, font=hint_font, anchor="w", justify="left").grid(
                row=1, column=1, sticky="w")


# ---------------------------------------------------------------- segmented
class Segmented(tk.Frame):
    def __init__(self, master: tk.Misc, options: Sequence[tuple[str, str]], variable: tk.StringVar,
                 command: Callable[[str], None] | None = None, font=None, pad_y: int = 6):
        super().__init__(master, bg=SURFACE2, padx=px(3), pady=px(3))
        self.var = variable
        self._cmd = command
        self._labels: dict[str, tk.Label] = {}
        for i, (key, text) in enumerate(options):
            lb = tk.Label(self, text=text, font=font, bg=SURFACE2, fg=MUTED, padx=px(10), pady=px(pad_y),
                          cursor="hand2")
            lb.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else px(2), 0))
            self.columnconfigure(i, weight=1, uniform="seg")
            lb.bind("<Button-1>", lambda e, k=key: self._select(k))
            lb.bind("<Enter>", lambda e, k=key: self._hover(k, True))
            lb.bind("<Leave>", lambda e, k=key: self._hover(k, False))
            self._labels[key] = lb
        variable.trace_add("write", lambda *_: self._paint())
        self._paint()

    def _select(self, key: str) -> None:
        if self.var.get() != key:
            self.var.set(key)
            if self._cmd is not None:
                self._cmd(key)

    def _hover(self, key: str, on: bool) -> None:
        if key != self.var.get():
            self._labels[key].configure(fg=TEXT if on else MUTED, bg=SURFACE3 if on else SURFACE2)

    def _paint(self) -> None:
        cur = self.var.get()
        for key, lb in self._labels.items():
            if key == cur:
                lb.configure(bg=ACCENT_DEEP, fg="#ffffff")
            else:
                lb.configure(bg=SURFACE2, fg=MUTED)


# ---------------------------------------------------------------- dropdown
class Dropdown(tk.Frame):
    """Combobox replacement with a fully styled popup list."""

    def __init__(self, master: tk.Misc, options: Sequence[tuple[str, str]], variable: tk.StringVar,
                 command: Callable[[str], None] | None = None, font=None):
        super().__init__(master, bg=SURFACE2, highlightthickness=1, highlightbackground=BORDER,
                         highlightcolor=ACCENT, cursor="hand2")
        self.var = variable
        self._cmd = command
        self._font = font
        self._items = list(options)
        self._popup: tk.Toplevel | None = None
        self._label = tk.Label(self, bg=SURFACE2, fg=TEXT, font=font, anchor="w", padx=px(10), pady=px(6))
        self._label.pack(side="left", fill="x", expand=True)
        self._arrow = tk.Label(self, text="▾", bg=SURFACE2, fg=MUTED, font=font, padx=px(10))
        self._arrow.pack(side="right")
        for w in (self, self._label, self._arrow):
            w.bind("<Button-1>", self._open)
            w.bind("<Enter>", lambda e: self._paint(True))
            w.bind("<Leave>", lambda e: self._paint(False))
        variable.trace_add("write", lambda *_: self._refresh())
        self._refresh()

    def set_options(self, options: Sequence[tuple[str, str]]) -> None:
        self._items = list(options)
        self._refresh()

    def _refresh(self) -> None:
        cur = self.var.get()
        text = next((t for k, t in self._items if k == cur), cur)
        self._label.configure(text=text)

    def _paint(self, hover: bool) -> None:
        c = SURFACE3 if hover else SURFACE2
        for w in (self, self._label, self._arrow):
            w.configure(bg=c)

    def _open(self, _ev=None) -> None:
        if self._popup is not None:
            self._close()
            return
        top = tk.Toplevel(self)
        top.overrideredirect(True)
        top.configure(bg=BORDER)
        try:
            top.attributes("-topmost", True)
        except tk.TclError:
            pass
        inner = tk.Frame(top, bg=SURFACE2)
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        cur = self.var.get()
        for key, text in self._items:
            sel = key == cur
            lb = tk.Label(inner, text=("✓  " if sel else "    ") + text, bg=SURFACE2, fg=ACCENT if sel else TEXT,
                          font=self._font, anchor="w", padx=px(10), pady=px(6), cursor="hand2")
            lb.pack(fill="x")
            lb.bind("<Enter>", lambda e, w=lb: w.configure(bg=SURFACE3))
            lb.bind("<Leave>", lambda e, w=lb: w.configure(bg=SURFACE2))
            lb.bind("<ButtonRelease-1>", lambda e, k=key: self._choose(k))
        top.update_idletasks()
        x, y = self.winfo_rootx(), self.winfo_rooty() + self.winfo_height() + px(2)
        top.geometry(f"{max(self.winfo_width(), top.winfo_reqwidth())}x{top.winfo_reqheight()}+{x}+{y}")
        top.bind("<Escape>", lambda e: self._close())
        top.bind("<FocusOut>", lambda e: self.after(80, self._close))
        top.bind("<Button-1>", self._click_outside, add="+")
        self._popup = top
        try:
            top.focus_force()
            top.grab_set()
        except tk.TclError:
            pass

    def _click_outside(self, ev) -> None:
        top = self._popup
        if top is None:
            return
        x, y = ev.x_root, ev.y_root
        if not (top.winfo_rootx() <= x < top.winfo_rootx() + top.winfo_width()
                and top.winfo_rooty() <= y < top.winfo_rooty() + top.winfo_height()):
            self._close()

    def _choose(self, key: str) -> None:
        self._close()
        if self.var.get() != key:
            self.var.set(key)
            if self._cmd is not None:
                self._cmd(key)

    def _close(self) -> None:
        top, self._popup = self._popup, None
        if top is not None:
            try:
                top.grab_release()
                top.destroy()
            except tk.TclError:
                pass


# ---------------------------------------------------------------- slider
class Slider(tk.Canvas):
    """Horizontal slider 0..1 (the caller maps it to real values)."""

    def __init__(self, master: tk.Misc, value: float = 0.0, command: Callable[[float, bool], None] | None = None,
                 height: int = 24):
        super().__init__(master, height=px(height), bg=master.cget("bg"), highlightthickness=0, bd=0,
                         cursor="hand2", takefocus=0)
        self.value = float(value)
        self._cmd = command
        self._drag = False
        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<ButtonPress-1>", lambda e: self._at(e.x, False))
        self.bind("<B1-Motion>", lambda e: self._at(e.x, False))
        self.bind("<ButtonRelease-1>", lambda e: self._at(e.x, True))

    def _at(self, x: float, final: bool) -> None:
        w = self.winfo_width()
        pad = px(10)
        self.value = min(1.0, max(0.0, (x - pad) / max(1, w - 2 * pad)))
        self._draw()
        if self._cmd is not None:
            self._cmd(self.value, final)

    def set(self, value: float) -> None:
        self.value = min(1.0, max(0.0, float(value)))
        self._draw()

    def _draw(self) -> None:
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 10:
            return
        pad, th = px(10), px(4)
        cy = h / 2
        x = pad + self.value * (w - 2 * pad)
        round_rect(self, pad, cy - th / 2, w - pad, cy + th / 2, th / 2, fill=SURFACE3, outline="")
        round_rect(self, pad, cy - th / 2, max(pad + th, x), cy + th / 2, th / 2, fill=ACCENT_DEEP, outline="")
        r = px(8)
        self.create_oval(x - r, cy - r, x + r, cy + r, fill="#ffffff", outline=ACCENT_DEEP, width=px(2))


# ---------------------------------------------------------------- misc
class Card(tk.Frame):
    def __init__(self, master: tk.Misc, padx: int = 16, pady: int = 14):
        super().__init__(master, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER,
                         padx=px(padx), pady=px(pady))


class StatusPill(tk.Canvas):
    def __init__(self, master: tk.Misc, font=None):
        super().__init__(master, height=px(30), width=px(150), bg=master.cget("bg"), highlightthickness=0, bd=0)
        self._font = tkfont.Font(font=font) if font is not None else tkfont.nametofont("TkDefaultFont")
        self._state = ("", MUTED)

    def set(self, text: str, color: str) -> None:
        if self._state == (text, color):
            return
        self._state = (text, color)
        self.delete("all")
        tw = self._font.measure(text)
        w, h = tw + px(40), px(30)
        self.configure(width=w)
        round_rect(self, 1, 1, w - 1, h - 1, h / 2, fill=mix(BG, color, 0.16), outline=mix(BG, color, 0.45))
        r = px(4)
        self.create_oval(px(14) - r, h / 2 - r, px(14) + r, h / 2 + r, fill=color, outline="")
        self.create_text(px(26), h / 2, text=text, anchor="w", fill=mix(color, "#ffffff", 0.55), font=self._font)


class NumEntry(tk.Entry):
    """Dark numeric entry; calls ``on_commit(value)`` on Enter / focus out."""

    def __init__(self, master: tk.Misc, value: float, on_commit: Callable[[float], None], width: int = 7,
                 font=None, minimum: float = 0.0, maximum: float = 1e9, integer: bool = False):
        self.var = tk.StringVar(value=self._fmt(value, integer))
        super().__init__(master, textvariable=self.var, width=width, font=font, bg=SURFACE2, fg=TEXT,
                         insertbackground=TEXT, relief="flat", justify="center", highlightthickness=1,
                         highlightbackground=BORDER, highlightcolor=ACCENT, disabledbackground=SURFACE2,
                         disabledforeground=DIM, selectbackground=ACCENT_DEEP, selectforeground="#ffffff")
        self._commit_cb = on_commit
        self._min, self._max, self._int = minimum, maximum, integer
        self.bind("<Return>", lambda e: self.commit())
        self.bind("<FocusOut>", lambda e: self.commit())

    @staticmethod
    def _fmt(v: float, integer: bool) -> str:
        return str(int(round(v))) if integer or float(v).is_integer() else f"{v:g}"

    def set_value(self, v: float) -> None:
        self.var.set(self._fmt(v, self._int))

    def commit(self) -> None:
        try:
            v = float(self.var.get().replace(",", ".").strip())
        except ValueError:
            return
        v = min(self._max, max(self._min, v))
        if self._int:
            v = int(round(v))
        self.var.set(self._fmt(v, self._int))
        self._commit_cb(v)


def sprite_photo(master: tk.Misc, cell: int) -> tk.PhotoImage:
    """The worm sprite as a Tk image (logo / window icon)."""
    from .sprite import PALETTE_RGB, SPRITE_GRID

    gw, gh = len(SPRITE_GRID[0]), len(SPRITE_GRID)
    img = tk.PhotoImage(master=master, width=gw * cell, height=gh * cell)
    for y, row in enumerate(SPRITE_GRID):
        for x, ch in enumerate(row):
            if ch == ".":
                continue
            r, g, b = PALETTE_RGB[ch]
            img.put(f"#{r:02x}{g:02x}{b:02x}", to=(x * cell, y * cell, (x + 1) * cell, (y + 1) * cell))
    return img
