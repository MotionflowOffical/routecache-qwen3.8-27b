@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe -m routecache_runtime run --port 8080
exit /b %errorlevel%
