@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe -m routecache_runtime ollama-create --name qwen3.8-27b-routecache
pause
