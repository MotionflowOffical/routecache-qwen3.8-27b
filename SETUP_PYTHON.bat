@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set NOPAUSE=0
if /I "%~1"=="--no-pause" set NOPAUSE=1
set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY where python >nul 2>nul && set PY=python
if not defined PY (
  echo Python 3 was not found. Trying WinGet installation...
  where winget >nul 2>nul || goto :nopy
  winget install --id Python.Python.3.12 -e --accept-package-agreements --accept-source-agreements
  if errorlevel 1 goto :nopy
  set "PATH=%LocalAppData%\Programs\Python\Python312;%LocalAppData%\Programs\Python\Python312\Scripts;%PATH%"
  where python >nul 2>nul && set PY=python
)
if not defined PY goto :nopy
if not exist .venv\Scripts\python.exe (
  %PY% -m venv .venv || goto :fail
)
.venv\Scripts\python.exe -m pip install --disable-pip-version-check --upgrade pip || goto :fail
.venv\Scripts\python.exe -m pip install --disable-pip-version-check -e . || goto :fail
exit /b 0
:nopy
echo ERROR: Python could not be installed automatically.
if "%NOPAUSE%"=="0" pause
exit /b 1
:fail
echo ERROR: Python environment setup failed.
if "%NOPAUSE%"=="0" pause
exit /b 1
