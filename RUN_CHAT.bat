@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe -m routecache_runtime run --port 8080 --ui
set rc=%errorlevel%
if not "%rc%"=="0" if not "%rc%"=="130" pause
exit /b %rc%
