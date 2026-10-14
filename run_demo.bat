@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe main.py train
if errorlevel 1 goto failed
.venv\Scripts\python.exe main.py demo
if errorlevel 1 goto failed
pause
exit /b 0
:failed
echo InfoTrustNet encountered an error. Read the output above.
pause
exit /b 1
