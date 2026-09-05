@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo ========================================================================
echo RouteCache Qwen3.8-27B - one-click setup and run
echo ========================================================================
call SETUP_PYTHON.bat --no-pause
if errorlevel 1 goto :fail
.venv\Scripts\python.exe -m routecache_runtime install
if errorlevel 1 goto :fail
.venv\Scripts\python.exe -m routecache_runtime run --ui
exit /b %errorlevel%
:fail
echo.
echo Setup failed. Run DIAGNOSTICS.bat for details.
pause
exit /b 1
