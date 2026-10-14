@echo off
setlocal
cd /d "%~dp0"
echo [1/3] Creating virtual environment...
if not exist .venv py -3 -m venv .venv
if errorlevel 1 (
  echo Could not create the environment. Install Python 3.11 or newer and try again.
  pause
  exit /b 1
)
echo [2/3] Installing dependencies...
call .venv\Scripts\python.exe -m pip install --upgrade pip
call .venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed. Check your internet connection and Python installation.
  pause
  exit /b 1
)
echo [3/3] Setup complete. Run run_demo.bat to train and launch the demo.
pause
