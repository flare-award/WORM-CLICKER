@echo off
rem Creates .venv and installs dependencies (called by run.bat / run_console.bat).
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" exit /b 0
where py >nul 2>nul && (set "PY=py -3") || (set "PY=python")
echo [Worm Clicker] Creating virtual environment...
%PY% -m venv .venv || goto :nopython
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :piperror
exit /b 0

:nopython
echo Python 3.10+ not found. Install it from https://www.python.org/downloads/ (check "Add python.exe to PATH").
pause
exit /b 1

:piperror
echo Failed to install dependencies. Check the internet connection and try again.
rmdir /s /q .venv
pause
exit /b 1
