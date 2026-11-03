@echo off
setlocal EnableExtensions
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "REPO_ROOT=%%~fI"
where uv >nul 2>&1
if errorlevel 1 (
  echo uv not found on PATH 1>&2
  exit /b 3
)
set "MODE=FAST"
set "SINCE="
if /I "%~1"=="fast" set "MODE=FAST"
if /I "%~1"=="full" set "MODE=FULL"
if /I "%~1"=="freeze" set "MODE=FREEZE"
if /I "%~1"=="changed" (
  set "MODE=CHANGED-SINCE"
  set "SINCE=%~2"
)
for /f "delims=" %%S in ('git -C "%REPO_ROOT%" rev-parse HEAD') do set "SHA=%%S"
set "OUT=%REPO_ROOT%\.tmp\proof\ent-audit-x\local-%SHA%-%RANDOM%"
if /I "%MODE%"=="CHANGED-SINCE" (
  if "%SINCE%"=="" (
    echo CHANGED-SINCE requires since SHA 1>&2
    exit /b 1
  )
  uv run python "%REPO_ROOT%\scripts\enterprise_audit\run_enterprise_audit.py" --mode CHANGED-SINCE --since-sha %SINCE% --output-dir "%OUT%"
) else (
  uv run python "%REPO_ROOT%\scripts\enterprise_audit\run_enterprise_audit.py" --mode %MODE% --output-dir "%OUT%"
)
exit /b %ERRORLEVEL%
