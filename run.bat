@echo off
rem Worm Clicker: double-click to start. First start installs dependencies.
cd /d "%~dp0"
call setup.bat || exit /b 1
rem arguments are passed through, e.g. run.bat --dry-run
start "" ".venv\Scripts\pythonw.exe" run_wormclicker.py %*
