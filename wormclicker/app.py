"""Entry point: ``python -m wormclicker`` (or WormClicker.exe)."""

from __future__ import annotations

import argparse
import ctypes
import logging
import sys
from pathlib import Path

from . import __version__, winapi
from .config import load_config

log = logging.getLogger("wormclicker")


def _single_instance_guard():
    """Named mutex: a second copy would start a competing click loop."""
    if not winapi.IS_WINDOWS:
        return object()
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateMutexW.restype = ctypes.c_void_p
    handle = k32.CreateMutexW(None, False, "Local\\WormClickerSingleInstance")
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        return None
    return handle


def _message(title: str, text: str) -> None:
    if winapi.IS_WINDOWS:
        ctypes.windll.user32.MessageBoxW(None, text, title, 0x30)
    else:
        print(f"{title}: {text}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="wormclicker", description="Auto clicker for the pink worm game")
    p.add_argument("--dry-run", action="store_true", help="тестовый режим: клики не отправляются")
    p.add_argument("--config", type=Path, help="путь к config.json")
    p.add_argument("--benchmark", action="store_true", help="замеры производительности без интерфейса")
    p.add_argument("--debug", action="store_true", help="подробный лог в консоль")
    p.add_argument("--version", action="version", version=__version__)
    args = p.parse_args(argv)

    if sys.stderr is not None:   # pythonw / windowed exe have no console
        logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    else:
        logging.getLogger().addHandler(logging.NullHandler())
    dpi = winapi.set_dpi_awareness()
    log.info("Worm Clicker %s, DPI awareness: %s", __version__, dpi)

    if args.benchmark:
        from .benchmark import run_benchmark

        return run_benchmark()

    try:
        import tkinter as tk
    except ImportError:
        print("tkinter не найден: установите Python с python.org (Tcl/Tk входит в установщик)", file=sys.stderr)
        return 2

    guard = _single_instance_guard()
    if guard is None:
        _message("Worm Clicker", "Программа уже запущена.")
        return 1

    from .automation import AutomationEngine
    from .config import default_config_path
    from .hotkeys import create_hotkey_manager
    from .ui import App

    if args.config:
        import os

        os.environ["WORMCLICKER_CONFIG"] = str(args.config)
    cfg = load_config(args.config or default_config_path())
    if args.dry_run:
        cfg.dry_run = True

    engine = AutomationEngine(cfg)
    holder: dict[str, App] = {}
    hotkeys = create_hotkey_manager(lambda action: holder["app"].handle_action(action) if "app" in holder else None)
    hotkeys.start()
    root = tk.Tk()
    try:
        holder["app"] = App(root, engine, hotkeys, cfg)
        root.mainloop()
    finally:
        engine.shutdown()
        hotkeys.stop()
    return 0
