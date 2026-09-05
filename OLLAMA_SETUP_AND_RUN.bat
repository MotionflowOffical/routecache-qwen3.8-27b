@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo ========================================================================
echo RouteCache Qwen3.8-27B - Ollama one-click setup
echo ========================================================================
set OLLAMA=
where ollama >nul 2>nul && set OLLAMA=ollama
if not defined OLLAMA if exist "%LocalAppData%\Programs\Ollama\ollama.exe" set "OLLAMA=%LocalAppData%\Programs\Ollama\ollama.exe"
if not defined OLLAMA (
  echo Ollama was not found. Trying WinGet installation...
  where winget >nul 2>nul || goto :noinstall
  winget install --id Ollama.Ollama -e --accept-package-agreements --accept-source-agreements
  if errorlevel 1 goto :noinstall
  if exist "%LocalAppData%\Programs\Ollama\ollama.exe" set "OLLAMA=%LocalAppData%\Programs\Ollama\ollama.exe"
  if not defined OLLAMA where ollama >nul 2>nul && set OLLAMA=ollama
)
if not defined OLLAMA goto :noinstall
for /f "usebackq delims=" %%R in (`powershell -NoProfile -Command "$p=Get-Content -Raw 'routecache_profile.json'^|ConvertFrom-Json; $p.hf_repo"`) do set "HFREPO=%%R"
if not defined HFREPO goto :badrepo
echo Running the GGUF directly from Hugging Face through Ollama...
echo HF root params sets num_ctx=4096.
"%OLLAMA%" run hf.co/%HFREPO%
exit /b %errorlevel%
:noinstall
echo ERROR: Ollama could not be installed automatically.
echo Official installer: https://ollama.com/download/windows
pause
exit /b 1
:badrepo
echo ERROR: routecache_profile.json has no published Hugging Face repo.
pause
exit /b 1
