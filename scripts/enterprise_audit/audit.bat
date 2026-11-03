@echo off
setlocal EnableExtensions
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "REPO_ROOT=%%~fI"
where uv >nul 2>&1
if errorlevel 1 (
  echo uv not found on PATH 1>&2
  exit /b 3
)
set "MODE="
set "SINCE="
if "%~1"=="" goto usage_error
if /I "%~1"=="fast" set "MODE=FAST" & goto mode_ok
if /I "%~1"=="full" set "MODE=FULL" & goto mode_ok
if /I "%~1"=="freeze" set "MODE=FREEZE" & goto mode_ok
if /I "%~1"=="changed" (
  if "%~2"=="" goto changed_missing_since
  set "MODE=CHANGED-SINCE"
  set "SINCE=%~2"
  goto mode_ok
)
goto usage_error
:changed_missing_since
echo CHANGED-SINCE requires since SHA 1>&2
exit /b 1
:usage_error
echo Usage: audit.bat fast^|full^|freeze^|changed ^<since-sha^> 1>&2
exit /b 1
:mode_ok
for /f "delims=" %%S in ('git -C "%REPO_ROOT%" rev-parse HEAD') do set "SHA=%%S"
for /f "delims=" %%U in ('powershell -NoProfile -Command "[guid]::NewGuid().ToString('n')"') do set "RUN_ID=%%U"
set "OUT=%REPO_ROOT%\.tmp\proof\ent-audit-x\local-%SHA%-%RUN_ID%"
if /I "%MODE%"=="CHANGED-SINCE" (
  if "%SINCE%"=="" (
    echo CHANGED-SINCE requires since SHA 1>&2
    exit /b 1
  )
  uv run python "%REPO_ROOT%\scripts\enterprise_audit\run_enterprise_audit.py" --mode CHANGED-SINCE --since-sha %SINCE% --output-dir "%OUT%"
) else if /I "%MODE%"=="FAST" (
  uv run python "%REPO_ROOT%\scripts\enterprise_audit\run_enterprise_audit.py" --mode FAST --non-certifying --output-dir "%OUT%"
) else if /I "%MODE%"=="FULL" (
  uv run python "%REPO_ROOT%\scripts\enterprise_audit\run_enterprise_audit.py" --mode FULL --output-dir "%OUT%" --evidence-bundle
) else (
  uv run python "%REPO_ROOT%\scripts\enterprise_audit\run_enterprise_audit.py" --mode %MODE% --output-dir "%OUT%"
)
exit /b %ERRORLEVEL%
