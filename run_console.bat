@echo off
rem Console start with log output. Examples:  run_console.bat --benchmark   run_console.bat --dry-run --debug
cd /d "%~dp0"
call setup.bat || exit /b 1
".venv\Scripts\python.exe" run_wormclicker.py %*
pause
