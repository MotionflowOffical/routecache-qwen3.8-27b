@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call SETUP_PYTHON.bat --no-pause
if errorlevel 1 goto :fail
if "%~1"=="" (
  .venv\Scripts\python.exe -m routecache_runtime install
) else (
  .venv\Scripts\python.exe -m routecache_runtime install "%~1"
)
if errorlevel 1 goto :fail
echo.
echo RouteCache runtime installed. Use RUN_CHAT.bat or RUN_API.bat.
pause
exit /b 0
:fail
echo.
echo Installation failed. Run DIAGNOSTICS.bat for details.
pause
exit /b 1
