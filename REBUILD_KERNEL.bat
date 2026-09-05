@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe -m routecache_runtime rebuild-kernel
pause
