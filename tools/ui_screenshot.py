"""CI helper: start the real UI on a simulated game, grab the window, print it as base64 annotations.

Used only to visually check layout/readability on a Windows runner (screenshots are never saved by
the program itself). Output: ``::notice title=shot <name> i/n::<base64 chunk>`` lines.
"""

from __future__ import annotations

import base64
import hashlib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["WORMCLICKER_ALLOW_CAPTURE"] = "1"          # let mss see our own window

import cv2  # noqa: E402
import numpy as np  # noqa: E402

CHUNK = 4000            # GitHub cuts annotation messages at 4096 characters
MAX_CHARS = 20 * CHUNK  # 10 notices + 10 warnings per step (GitHub limit)
OUT = Path(os.environ.get("RUNNER_TEMP") or os.environ.get("TEMP") or "/tmp") / "wc_shots"
SHOTS = os.environ.get("SHOTS", "idle,running,target-tab,capture-tab").split(",")


def save(name: str, bgr: np.ndarray) -> None:
    """Encode so that the base64 text fits into the annotation budget of one CI step."""
    width = int(os.environ.get("SHOT_WIDTH", "1100"))
    h, w = bgr.shape[:2]
    if w > width:
        bgr = cv2.resize(bgr, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    for q in (85, 75, 65, 55, 45, 35):
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
        if len(base64.b64encode(buf.tobytes())) <= MAX_CHARS:
            break
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.jpg").write_bytes(buf.tobytes())
    report(f"{name}: {w}x{h}, jpeg q={q}, {len(buf)} bytes", "notice")


def emit(name: str) -> None:
    data = base64.b64encode((OUT / f"{name}.jpg").read_bytes()).decode()
    parts = [data[i:i + CHUNK] for i in range(0, len(data), CHUNK)]
    digest = hashlib.sha1(data.encode()).hexdigest()[:10]
    for i, p in enumerate(parts):
        level = "notice" if i < 10 else "warning"
        print(f"::{level} title=shot {name} {i + 1}/{len(parts)} {digest}::{p}", flush=True)


def main() -> int:
    import tkinter as tk

    import mss

    from tests.test_engine import build
    from wormclicker import winapi
    from wormclicker.hotkeys import HotkeyManager
    from wormclicker.ui import App
    from wormclicker.widgets import Scale

    winapi.set_dpi_awareness()
    winapi.set_timer_resolution(True)
    os.environ.setdefault("WORMCLICKER_CONFIG", str(Path(os.environ.get("TEMP", "/tmp")) / "wc_shot.json"))
    root = tk.Tk()
    eng, game, cap, mouse = build(preview=True)
    app = App(root, eng, HotkeyManager(lambda a: None), eng.cfg)
    root.attributes("-topmost", True)
    report(f"scale={Scale.k:.2f} screen={root.winfo_screenwidth()}x{root.winfo_screenheight()}", "notice")

    def pump(seconds: float) -> None:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            root.update()
            time.sleep(0.004)

    sct = getattr(mss, "MSS", None) or mss.mss
    grabber = sct()

    def shot(name: str) -> None:
        root.update()
        x, y = root.winfo_rootx(), root.winfo_rooty()
        w, h = root.winfo_width(), root.winfo_height()
        raw = np.asarray(grabber.grab({"left": x, "top": y, "width": w, "height": h}))
        if name in SHOTS:
            save(name, cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR))

    try:
        pump(2.0)
        n0 = eng.preview_frames
        pump(1.0)
        report(f"preview frames/s (engine) = {eng.preview_frames - n0}", "notice")
        shot("idle")
        app._toggle()
        pump(0.5)
        game.kick()
        pump(2.0)
        s = eng.stats.snapshot()
        report(f"clicking: clicks={s.clicks} hits={game.hits} decoy={game.decoy_clicks} cps={s.cps:.0f}", "notice")
        shot("running")
        app._show_tab("target")
        pump(0.4)
        shot("target-tab")
        app.handle_action("stop")
        app._show_tab("capture")
        pump(0.4)
        shot("capture-tab")
    finally:
        app.close()
        winapi.set_timer_resolution(False)
    return 0


def report(text: str, level: str = "error") -> None:
    msg = text.replace("%", "%25").replace("\r", "").replace("\n", "%0A")
    print(f"::{level} title=ui_screenshot::{msg}", flush=True)


if __name__ == "__main__":
    import traceback

    try:
        if len(sys.argv) == 3 and sys.argv[1] == "--emit":
            emit(sys.argv[2])
            raise SystemExit(0)
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        report(traceback.format_exc()[-6000:])
        raise SystemExit(1) from None
