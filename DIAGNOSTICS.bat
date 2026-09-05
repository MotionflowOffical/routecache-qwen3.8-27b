@echo off
setlocal
cd /d "%~dp0"
echo === RouteCache public runtime diagnostics ===
where python 2>nul
where py 2>nul
where winget 2>nul
where ollama 2>nul
where cmake 2>nul
where nvcc 2>nul
where nvidia-smi 2>nul
if exist runtime\manifest.json type runtime\manifest.json
if exist runtime\kernel_build.json type runtime\kernel_build.json
if exist runtime\reproduction_report.json type runtime\reproduction_report.json
pause
