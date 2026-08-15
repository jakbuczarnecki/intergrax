@echo off
REM © Artur Czarnecki. All rights reserved.
REM Intergrax framework – proprietary and confidential.
REM Use, modification, or distribution without written permission is prohibited.

setlocal EnableExtensions

set "CONFIG=docs/project/maintainers/public-adoption/curated_public_issues.yml"
set "SCRIPT=scripts/public_adoption/create_curated_issues.py"

pushd "%~dp0\..\.." >nul 2>&1
if errorlevel 1 (
  echo ERROR: Cannot move to repository root.
  exit /b 1
)

set "MODE=%~1"
set "WAVE=%~2"

if "%MODE%"=="" goto usage

if /I "%MODE%"=="dry" goto run
if /I "%MODE%"=="dry-run" goto run
if /I "%MODE%"=="apply" goto run
if /I "%MODE%"=="check" goto run
if /I "%MODE%"=="sync" goto run
if /I "%MODE%"=="help" goto usage
if /I "%MODE%"=="/help" goto usage
if /I "%MODE%"=="-h" goto usage
if /I "%MODE%"=="--help" goto usage

echo ERROR: Unknown mode: %MODE%
echo.
goto usage

:run
if not "%WAVE%"=="" (
  call :run_one "%MODE%" "%WAVE%"
  goto done
)

if /I "%MODE%"=="check" (
  python "%SCRIPT%" --config "%CONFIG%" --check-sync
  goto done
)
if /I "%MODE%"=="sync" (
  python "%SCRIPT%" --config "%CONFIG%" --check-sync
  goto done
)

for /f "usebackq delims=" %%W in (`python -c "import yaml; from pathlib import Path; data=yaml.safe_load(Path(r'%CONFIG%').read_text(encoding='utf-8')); print('\n'.join(str(k) for k in data.get('waves', {}).keys()))"`) do (
  call :run_one "%MODE%" "%%W"
  if errorlevel 1 goto done
)

goto done

:run_one
set "RUN_MODE=%~1"
set "RUN_WAVE=%~2"

echo.
echo === %RUN_MODE% %RUN_WAVE% ===

if /I "%RUN_MODE%"=="dry" (
  python "%SCRIPT%" --config "%CONFIG%" --wave "%RUN_WAVE%"
  exit /b %ERRORLEVEL%
)

if /I "%RUN_MODE%"=="dry-run" (
  python "%SCRIPT%" --config "%CONFIG%" --wave "%RUN_WAVE%"
  exit /b %ERRORLEVEL%
)

if /I "%RUN_MODE%"=="apply" (
  python "%SCRIPT%" --config "%CONFIG%" --wave "%RUN_WAVE%" --apply
  exit /b %ERRORLEVEL%
)

if /I "%RUN_MODE%"=="check" (
  python "%SCRIPT%" --config "%CONFIG%" --wave "%RUN_WAVE%" --check-sync
  exit /b %ERRORLEVEL%
)

if /I "%RUN_MODE%"=="sync" (
  python "%SCRIPT%" --config "%CONFIG%" --wave "%RUN_WAVE%" --check-sync
  exit /b %ERRORLEVEL%
)

echo ERROR: Internal unknown run mode: %RUN_MODE%
exit /b 1

:usage
echo Usage:
echo   scripts\public_adoption\manage_curated_issues.bat dry
echo   scripts\public_adoption\manage_curated_issues.bat apply
echo   scripts\public_adoption\manage_curated_issues.bat check
echo.
echo Optional single-wave mode:
echo   scripts\public_adoption\manage_curated_issues.bat dry wave_3
echo   scripts\public_adoption\manage_curated_issues.bat apply wave_3
echo   scripts\public_adoption\manage_curated_issues.bat check wave_3
echo.
echo Behavior:
echo   - Source of truth: docs\project\maintainers\public-adoption\curated_public_issues.yml
echo   - With no wave argument, processes every wave defined in the YAML
echo   - Existing GitHub issues are skipped by exact title
echo   - Missing GitHub issues are created only in apply mode
exit /b 1

:done
set "EXIT_CODE=%ERRORLEVEL%"
popd >nul 2>&1
exit /b %EXIT_CODE%
