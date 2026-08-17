@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Сначала запустите install_windows.bat
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
python -m pip install pyinstaller
pyinstaller --noconfirm CoreTranscriber.spec
if errorlevel 1 (
  echo Сборка не удалась.
  pause
  exit /b 1
)
echo Готово: dist\CoreTranscriber\CoreTranscriber.exe
pause
