@echo off
REM © Artur Czarnecki. All rights reserved.
REM Intergrax framework – proprietary and confidential.
REM Use, modification, or distribution without written permission is prohibited.

setlocal EnableExtensions

set "CONFIG=docs/public-adoption/curated_public_issues.yml"
set "SCRIPT=scripts/public_adoption/create_curated_issues.py"

pushd "%~dp0\..\.." >nul 2>&1
if errorlevel 1 (
  echo ERROR: Cannot move to repository root.
  exit /b 1
)

set "MODE=%~1"
set "WAVE=%~2"

if "%MODE%"=="" goto usage
if "%WAVE%"=="" set "WAVE=wave_3"

if /I "%MODE%"=="dry" goto dry
if /I "%MODE%"=="dry-run" goto dry
if /I "%MODE%"=="apply" goto apply
if /I "%MODE%"=="check" goto check
if /I "%MODE%"=="sync" goto check
if /I "%MODE%"=="help" goto usage
if /I "%MODE%"=="/help" goto usage
if /I "%MODE%"=="-h" goto usage
if /I "%MODE%"=="--help" goto usage

echo ERROR: Unknown mode: %MODE%
echo.
goto usage

:dry
call :run_for_wave "dry" "%WAVE%"
goto done

:apply
call :run_for_wave "apply" "%WAVE%"
goto done

:check
call :run_for_wave "check" "%WAVE%"
goto done

:run_for_wave
set "RUN_MODE=%~1"
set "RUN_WAVE=%~2"

if /I "%RUN_WAVE%"=="all" (
  call :run_one "%RUN_MODE%" "wave_3" || exit /b 1
  call :run_one "%RUN_MODE%" "wave_4" || exit /b 1
  call :run_one "%RUN_MODE%" "wave_5" || exit /b 1
  exit /b 0
)

call :validate_wave "%RUN_WAVE%" || exit /b 1
call :run_one "%RUN_MODE%" "%RUN_WAVE%"
exit /b %ERRORLEVEL%

:validate_wave
if /I "%~1"=="wave_3" exit /b 0
if /I "%~1"=="wave_4" exit /b 0
if /I "%~1"=="wave_5" exit /b 0
echo ERROR: Unknown wave: %~1
echo Allowed waves: wave_3, wave_4, wave_5, all
exit /b 1

:run_one
set "RUN_MODE=%~1"
set "RUN_WAVE=%~2"

echo.
echo === %RUN_MODE% %RUN_WAVE% ===

if /I "%RUN_MODE%"=="dry" (
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

echo ERROR: Internal unknown run mode: %RUN_MODE%
exit /b 1

:usage
echo Usage:
echo   scripts\public_adoption\manage_discussion_issues.bat dry   wave_3
echo   scripts\public_adoption\manage_discussion_issues.bat apply wave_3
echo   scripts\public_adoption\manage_discussion_issues.bat check wave_3
echo.
echo Waves:
echo   wave_3   Architecture discussion issues
echo   wave_4   Product / application validation issues
echo   wave_5   Deep technical discussion issues
echo   all      Run wave_3, wave_4 and wave_5 in sequence
echo.
echo Examples:
echo   scripts\public_adoption\manage_discussion_issues.bat dry wave_3
echo   scripts\public_adoption\manage_discussion_issues.bat apply wave_3
echo   scripts\public_adoption\manage_discussion_issues.bat apply all
echo   scripts\public_adoption\manage_discussion_issues.bat check all
exit /b 1

:done
set "EXIT_CODE=%ERRORLEVEL%"
popd >nul 2>&1
exit /b %EXIT_CODE%
