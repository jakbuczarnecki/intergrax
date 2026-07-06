@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
set "APP_DIR=%SCRIPT_DIR%.."
set "DOCKER_DIR=%APP_DIR%\docker"
set "SENTRY_PROOF_DIR=%DOCKER_DIR%\sentry-proof"
set "RUN_ALL=%SCRIPT_DIR%run-local-docker-all.bat"

if not exist "%RUN_ALL%" (
    echo Missing canonical docker runner: %RUN_ALL%
    exit /b 1
)

pushd "%APP_DIR%\..\.." >nul

if errorlevel 1 (
    echo Failed to enter repository root.
    exit /b 1
)

echo LKW local Docker hard reset
echo Repository root: %CD%
echo.
echo This will remove Docker containers, volumes, orphans, and local Sentry proof runtime state.
echo It will not remove source files, .env, committed relay credentials, or sample documents.
echo.

echo [1/3] Stopping and removing local Docker stack with volumes...
call "%RUN_ALL%" down -v --remove-orphans
if errorlevel 1 (
    set "EXIT_CODE=%ERRORLEVEL%"
    echo Docker compose down failed with exit code %EXIT_CODE%.
    popd >nul
    exit /b %EXIT_CODE%
)

echo.
echo [2/3] Removing local Sentry proof runtime state...
if exist "%SENTRY_PROOF_DIR%\generated.env" (
    del /f /q "%SENTRY_PROOF_DIR%\generated.env"
    echo Removed %SENTRY_PROOF_DIR%\generated.env
) else (
    echo No generated.env to remove.
)

if exist "%SENTRY_PROOF_DIR%\generated.env.tmp" (
    del /f /q "%SENTRY_PROOF_DIR%\generated.env.tmp"
    echo Removed %SENTRY_PROOF_DIR%\generated.env.tmp
) else (
    echo No generated.env.tmp to remove.
)

if exist "%SENTRY_PROOF_DIR%\.bootstrapped" (
    del /f /q "%SENTRY_PROOF_DIR%\.bootstrapped"
    echo Removed %SENTRY_PROOF_DIR%\.bootstrapped
) else (
    echo No .bootstrapped marker to remove.
)

echo.
echo [3/3] Starting clean local Docker stack in detached mode...
call "%RUN_ALL%" up -d --build
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo Docker compose startup failed with exit code %EXIT_CODE%.
    popd >nul
    exit /b %EXIT_CODE%
)

echo.
echo LKW local Docker hard reset complete.
echo.
echo Next step:
echo   applications\local_workspace_application\scripts\check-lkw-platform-proof-status.bat
echo.
echo If the status checker prints proof_status=WAIT, wait 30-60 seconds and run it again.
echo If the status checker prints proof_status=FAIL, use the diagnostic command printed by the checker.
echo.

popd >nul
exit /b 0
