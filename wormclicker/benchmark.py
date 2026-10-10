"""Performance measurements: ``python -m wormclicker --benchmark``.

1. Target finder speed on synthetic 1080p / 720p frames (CPU only): the
   worm jumps between frames, static decoys and page elements are present.
2. Real screen capture speed of the available backends (Windows / X11).
3. End-to-end offline loop with the simulator (capture -> find -> click),
   both with an instant game and with 30 ms render latency.

Nothing is clicked on the real screen.
"""

from __future__ import annotations

import statistics
import time

import numpy as np


def _ms(samples: list[float]) -> str:
    return f"медиана {statistics.median(samples) * 1000:.2f} мс, p95 {np.percentile(samples, 95) * 1000:.2f} мс"


def bench_detector(n: int = 60) -> list[str]:
    from .sprite import render_worm
    from .synthetic import scene_with_distractors
    from .target import TargetFinder

    out = []
    for res in ((1920, 1080), (1280, 720)):
        w, h = res
        bg = scene_with_distractors(w, h, 0)
        render_worm(bg, (w * 0.85, h * 0.85), 25.0, 4)            # static decoy worm
        rng = np.random.default_rng(1)
        frames, masks = [], []
        for _ in range(8):
            f = bg.copy()
            masks.append(render_worm(f, (float(rng.uniform(80, w * 0.7)), float(rng.uniform(280, h - 80))),
                                     float(rng.uniform(0, 360)), 4))
            frames.append(f)
        tf = TargetFinder()
        now = 0.0
        tf.process(bg, now)
        t, found = [], 0
        for i in range(n):
            now += 0.016
            k = i % len(frames)
            t0 = time.perf_counter()
            r = tf.process(frames[k], now)
            t.append(time.perf_counter() - t0)
            if r.target is not None and masks[k][int(r.target.click[1]), int(r.target.click[0])]:
                found += 1
        out.append(f"поиск цели {w}x{h}: {_ms(t)}, попадание точкой клика в червя {found}/{n} кадров")
    return out


def bench_capture(n: int = 40) -> list[str]:
    from .capture import DxcamCapture, MssCapture
    from .geometry import Rect

    out = []
    for cls in (DxcamCapture, MssCapture):
        try:
            cap = cls()
        except Exception as exc:
            out.append(f"захват {cls.name}: недоступен ({exc})")
            continue
        try:
            mons = cap.monitors()
            if not mons:
                out.append(f"захват {cls.name}: мониторы не найдены")
                continue
            m = mons[0]
            rect = Rect(m.left, m.top, min(m.width, 1600), min(m.height, 900))
            t, none = [], 0
            for _ in range(n):
                t0 = time.perf_counter()
                f = cap.grab(rect)
                t.append(time.perf_counter() - t0)
                none += f is None
            out.append(f"захват {cls.name} {rect.width}x{rect.height}: {_ms(t)}"
                       + (f", без нового кадра: {none}/{n}" if none else ""))
        except Exception as exc:
            out.append(f"захват {cls.name}: ошибка {exc}")
        finally:
            cap.close()
    return out


def bench_e2e(seconds: float = 3.0) -> list[str]:
    from .automation import AutomationEngine
    from .config import AppConfig
    from .geometry import Rect
    from .region import RegionProvider
    from .simulator import SimCapture, SimMouse, SimulatedGame

    out = []
    for latency, interval in ((0.0, 0.0), (0.03, 0.0), (0.0, 20.0)):
        game = SimulatedGame(1600, 900, render_latency_s=latency, seed=1, decoy=True)
        mons = [Rect(0, 0, 1600, 900)]
        cap = SimCapture(game, (0, 0), monitors=mons)
        cfg = AppConfig(capture_area="monitor:0", click_interval_ms=interval, debug_view=False)
        eng = AutomationEngine(cfg, capture_factory=lambda c=cap: c,
                               mouse_factory=lambda dry, g=game: SimMouse(g),
                               region_provider=RegionProvider("monitor:0", lambda m=mons: m))
        eng.start()
        time.sleep(0.2)
        game.kick()                       # the user's first click on the worm
        t0 = time.perf_counter()
        time.sleep(seconds)
        eng.stop()
        dt = time.perf_counter() - t0
        s = eng.stats.snapshot()
        eng.shutdown()
        mode = "максимум" if interval == 0 else f"интервал {interval:.0f} мс"
        out.append(f"симулятор ({mode}, задержка отрисовки {latency * 1000:.0f} мс): "
                   f"{game.hits / dt:.1f} попаданий/с, {s.clicks / dt:.1f} кликов/с, "
                   f"клики по статичной копии червя: {game.decoy_clicks}, "
                   f"анализ кадра {s.detect_ms:.1f} мс, кадр→клик {s.pipeline_ms:.1f} мс")
    return out


def run_benchmark() -> int:
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:  # redirected output on Windows defaults to cp1252/cp866
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    print("Worm Clicker — замеры производительности (реальные клики не отправляются)\n")
    for section in (bench_detector, bench_capture, bench_e2e):
        for line in section():
            print(" •", line, flush=True)
    print("\nВнимание: итоговая скорость в реальной игре ограничена ещё и браузером, частотой "
          "обновления монитора и самой игрой. Смотрите статистику в окне программы.")
    return 0
