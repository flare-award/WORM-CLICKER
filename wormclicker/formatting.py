"""Pure text formatting for the UI (no Tk import - testable everywhere)."""

from __future__ import annotations

import math

from .stats import StatsSnapshot


def format_stats(s: StatsSnapshot) -> dict[str, str]:
    """Pure formatting (unit-tested without a display)."""
    def ms(v: float) -> str:
        return f"{v:.1f}" if v > 0 else "—"

    rate = f" · {100.0 * s.jumps / s.clicks:.0f}%" if s.clicks else ""
    session = f"{int(s.session_s // 60)}:{int(s.session_s % 60):02d}"
    return {
        "cps": f"{s.cps:.0f}",
        "jumps_ps": f"{s.jumps_ps:.0f}",
        "fps": f"{s.fps:.0f}",
        "clicks": str(s.clicks),
        "jumps": f"{s.jumps}{rate}",
        "pipeline_ms": ms(s.pipeline_ms),
        "session": session,
        "detail": (f"захват {ms(s.capture_ms)} мс · анализ {ms(s.detect_ms)} мс · потерь цели {s.lost}"
                   f" · не-целей {s.ignored} · сессия {session}"),
    }


def interval_to_pos(ms: float) -> float:
    if ms <= 0:
        return 0.0
    return min(1.0, 0.05 + 0.95 * math.log10(max(1.0, ms)) / 3.0)


def pos_to_interval(p: float) -> float:
    if p < 0.04:
        return 0.0
    return float(max(1, int(round(10 ** (max(0.0, p - 0.05) / 0.95 * 3.0)))))


def interval_text(ms: float) -> tuple[str, str]:
    if ms <= 0:
        return "Максимум", "клик на каждом кадре, без пауз"
    return f"{ms:g} мс", f"≈ {1000.0 / ms:.0f} кликов в секунду"
