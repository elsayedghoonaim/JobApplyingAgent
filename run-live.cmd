@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv is missing. Run: py -3.12 -m venv .venv
  exit /b 1
)
".venv\Scripts\python.exe" -m jobapply.live %*
exit /b %ERRORLEVEL%
