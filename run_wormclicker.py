"""Launcher used by PyInstaller and for double-click start (python run_wormclicker.py)."""

from wormclicker.app import main

if __name__ == "__main__":
    raise SystemExit(main())
