"""Performance measurements: ``python -m wormclicker --benchmark``.

1. Detector speed on synthetic 1080p frames (CPU only).
2. Real screen capture speed of the available backends (Windows / X11).
3. End-to-end offline loop with the simulator (capture -> detect -> click),
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
    from .detector import WormDetector
    from .synthetic import scene

    out = []
    det = WormDetector()
    for res in ((1920, 1080), (1280, 720)):
        frames = [scene(res[0], res[1], worm_center=(300 + 37 * i % (res[0] - 600), 300 + 13 * i % (res[1] - 500)),
                        angle=17.0 * i, pixel_size=4, seed=i)[0] for i in range(8)]
        for i in range(3):
            det.detect(frames[i])
        t = []
        found = 0
        for i in range(n):
            t0 = time.perf_counter()
            d = det.detect(frames[i % len(frames)])
            t.append(time.perf_counter() - t0)
            found += d.best is not None
        out.append(f"детектор {res[0]}x{res[1]}: {_ms(t)}, найден в {found}/{n} кадров")
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
    for latency in (0.0, 0.03):
        game = SimulatedGame(1600, 900, render_latency_s=latency, seed=1)
        mons = [Rect(0, 0, 1600, 900)]
        cap = SimCapture(game, (0, 0), monitors=mons)
        eng = AutomationEngine(AppConfig(capture_area="monitor:0"), capture_factory=lambda c=cap: c,
                               mouse_factory=lambda dry, g=game: SimMouse(g),
                               region_provider=RegionProvider("monitor:0", lambda m=mons: m))
        eng.start()
        time.sleep(seconds)
        eng.stop()
        s = eng.stats.snapshot()
        eng.shutdown()
        acc = 100.0 * game.hits / max(1, game.hits + game.misses)
        out.append(f"симулятор (задержка отрисовки {latency * 1000:.0f} мс): {game.hits / seconds:.1f} попаданий/с, "
                   f"точность {acc:.1f}%, клики по старой позиции: {game.clicks_on_old_spot}, "
                   f"детекция {s.detect_ms:.1f} мс, кадр→клик {s.pipeline_ms:.1f} мс")
    return out


def run_benchmark() -> int:
    print("Worm Clicker — замеры производительности (реальные клики не отправляются)\n")
    for section in (bench_detector, bench_capture, bench_e2e):
        for line in section():
            print(" •", line, flush=True)
    print("\nВнимание: итоговая скорость в реальной игре ограничена ещё и браузером, частотой "
          "обновления монитора и самой игрой. Смотрите статистику в окне программы.")
    return 0
