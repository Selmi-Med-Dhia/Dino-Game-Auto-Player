@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo Python launcher ^(py^) was not found.
    echo Install Python 3.11+ from https://www.python.org/downloads/ and enable the launcher.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3 -m venv .venv
    if errorlevel 1 goto :error
)

echo Installing/updating dependencies...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :error

echo Starting Dino Auto-Player...
".venv\Scripts\python.exe" app.py
exit /b 0

:error
echo.
echo Setup failed. See the error above.
pause
exit /b 1
