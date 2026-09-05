@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo ========================================================================
echo RouteCache v1.2 - reproduce from official Qwen3.8-27B weights
echo ========================================================================
echo Advanced path: ~55+ GB upstream download and ~120-140 GiB temporary disk.
echo Normal users should use SETUP_AND_RUN.bat instead.
echo.
call SETUP_PYTHON.bat --no-pause
if errorlevel 1 goto :fail
.venv\Scripts\python.exe -m routecache_runtime reproduce
if errorlevel 1 goto :fail
echo.
echo Reproduction finished. Installing runtime around the reproduced local GGUF...
.venv\Scripts\python.exe -m routecache_runtime install-local
if errorlevel 1 goto :fail
echo.
echo Reproduced model and runtime are ready. Run RUN_CHAT.bat.
echo See runtime\reproduction_report.json for byte/topology verification.
pause
exit /b 0
:fail
echo.
echo Reproduction failed. The normal prebuilt install path is unaffected.
pause
exit /b 1
