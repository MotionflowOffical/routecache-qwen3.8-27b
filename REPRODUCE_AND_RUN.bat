@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo ========================================================================
echo RouteCache v1.2 - full reproduce from Qwen source, install, and run
echo ========================================================================
echo WARNING: advanced path; ~55+ GB download and 120-140 GiB temporary disk.
echo.
call SETUP_PYTHON.bat --no-pause
if errorlevel 1 goto :fail
.venv\Scripts\python.exe -m routecache_runtime reproduce
if errorlevel 1 goto :fail
.venv\Scripts\python.exe -m routecache_runtime install-local
if errorlevel 1 goto :fail
.venv\Scripts\python.exe -m routecache_runtime run --ui
exit /b %errorlevel%
:fail
echo.
echo Reproduce/install/run failed. See runtime\reproduction_report.json and DIAGNOSTICS.bat.
pause
exit /b 1
