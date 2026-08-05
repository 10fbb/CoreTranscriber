@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Сначала запустите install_windows.bat
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" main.py

