@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\..\.."
set "DOCKER_DIR=%SCRIPT_DIR%..\docker"
set "BASE_COMPOSE=%DOCKER_DIR%\docker-compose.yml"

pushd "%REPO_ROOT%" >nul
if errorlevel 1 (
    echo proof_status=FAIL
    echo reason=failed_to_enter_repository_root
    exit /b 1
)

set "COMPOSE_ARGS=-f %BASE_COMPOSE%"
for %%F in ("%DOCKER_DIR%\docker-compose.*.yml") do (
    set "COMPOSE_ARGS=!COMPOSE_ARGS! -f %%~fF"
)

set "REQUIRED_UP=local_workspace elasticsearch kibana sentry-web sentry-relay sentry-nginx sentry-events-consumer"
set "REQUIRED_EXITED_ZERO=sentry-bootstrap sentry-upgrade sentry-snuba-bootstrap sentry-kafka-topics"
set "FAILURES=0"
set "WAITING=0"

echo LKW platform proof status check
echo Repository root: %CD%
echo.

for %%S in (%REQUIRED_UP%) do call :check_up %%S
for %%S in (%REQUIRED_EXITED_ZERO%) do call :check_exited_zero %%S

echo.
if "%FAILURES%"=="0" if "%WAITING%"=="0" (
    echo proof_status=PASS
    echo next_step=run-sentry-observability-proof
    popd >nul
    exit /b 0
)

if not "%FAILURES%"=="0" (
    echo proof_status=FAIL
    echo failed_checks=%FAILURES%
    echo waiting_checks=%WAITING%
    echo.
    echo Inspect details with:
    echo   applications\local_workspace_application\scripts\run-local-docker-all.bat ps -a
    popd >nul
    exit /b 1
)

echo proof_status=WAIT
echo waiting_checks=%WAITING%
echo.
echo Wait 30-60 seconds and run this status checker again.
popd >nul
exit /b 2

:check_up
set "SERVICE=%~1"
set "STATUS="
for /f "delims=" %%L in ('docker compose %COMPOSE_ARGS% ps --format "{{.Service}}|{{.Status}}" 2^>nul') do (
    for /f "tokens=1,* delims=|" %%A in ("%%L") do (
        if "%%A"=="%SERVICE%" set "STATUS=%%B"
    )
)

if "%STATUS%"=="" (
    echo [FAIL] %SERVICE% missing
    set /a FAILURES+=1
    exit /b 0
)

echo %STATUS% | findstr /i /c:"Up" >nul
if errorlevel 1 (
    echo %STATUS% | findstr /i /c:"Restarting" >nul
    if not errorlevel 1 (
        echo [WAIT] %SERVICE% %STATUS%
        set /a WAITING+=1
        exit /b 0
    )
    echo [FAIL] %SERVICE% %STATUS%
    set /a FAILURES+=1
    exit /b 0
)

echo [ OK ] %SERVICE% %STATUS%
exit /b 0

:check_exited_zero
set "SERVICE=%~1"
set "STATUS="
for /f "delims=" %%L in ('docker compose %COMPOSE_ARGS% ps -a --format "{{.Service}}|{{.Status}}" 2^>nul') do (
    for /f "tokens=1,* delims=|" %%A in ("%%L") do (
        if "%%A"=="%SERVICE%" set "STATUS=%%B"
    )
)

if "%STATUS%"=="" (
    echo [FAIL] %SERVICE% missing
    set /a FAILURES+=1
    exit /b 0
)

echo %STATUS% | findstr /i /c:"Exited (0)" >nul
if errorlevel 1 (
    echo %STATUS% | findstr /i /c:"Up" >nul
    if not errorlevel 1 (
        echo [WAIT] %SERVICE% %STATUS%
        set /a WAITING+=1
        exit /b 0
    )
    echo [FAIL] %SERVICE% %STATUS%
    set /a FAILURES+=1
    exit /b 0
)

echo [ OK ] %SERVICE% %STATUS%
exit /b 0
