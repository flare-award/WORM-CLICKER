@echo off
rem Builds dist\WormClicker.exe (single file, Python is not needed on the target PC).
cd /d "%~dp0"
call setup.bat || exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements-dev.txt || exit /b 1
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --onefile --windowed --name WormClicker --collect-submodules dxcam run_wormclicker.py || exit /b 1
copy /y config.default.json dist\ >nul
xcopy /e /i /y practice dist\practice >nul
echo.
echo Done: dist\WormClicker.exe
pause
