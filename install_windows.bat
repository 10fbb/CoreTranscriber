@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Python не найден. Установите Python 3.11 с python.org и включите Add Python to PATH.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" py -3.11 -m venv .venv
if errorlevel 1 (
  echo Не удалось создать виртуальное окружение Python 3.11.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Установка не завершена. Проверьте интернет и повторите запуск.
  pause
  exit /b 1
)
echo.
echo Установка завершена. Запустите run_windows.bat
pause

