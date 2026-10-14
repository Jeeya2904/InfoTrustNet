@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup.bat first.
  pause
  exit /b 1
)
if not exist .venv\Scripts\streamlit.exe (
  echo Installing optional Streamlit web UI...
  .venv\Scripts\python.exe -m pip install -r requirements-app.txt
  if errorlevel 1 goto failed
)
if not exist artifacts\infotrustnet.joblib (
  echo Training model for first run...
  .venv\Scripts\python.exe main.py train
  if errorlevel 1 goto failed
)
.venv\Scripts\streamlit.exe run app.py
if errorlevel 1 goto failed
exit /b 0
:failed
echo The web app could not start. Read the output above.
pause
exit /b 1
