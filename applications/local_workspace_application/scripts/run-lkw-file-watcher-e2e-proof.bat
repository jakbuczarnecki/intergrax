@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "APP_DIR=%SCRIPT_DIR%.."
set "REPO_ROOT=%SCRIPT_DIR%..\..\.."
set "DOCKER_DIR=%APP_DIR%\docker"
set "BASE_COMPOSE=%DOCKER_DIR%\docker-compose.yml"
set "KAFKA_COMPOSE=%DOCKER_DIR%\docker-compose.kafka.yml"
set "WATCHER_COMPOSE=%DOCKER_DIR%\file-watcher-e2e.compose.yml"
set "PROOF=%SCRIPT_DIR%run-lkw-file-watcher-e2e-proof.py"
set "PROOF_DOCS_DIR=%APP_DIR%\.proof_docs"
set "WATCHER_STATE_DIR=%APP_DIR%\.file_watcher_e2e_state"
set "COMPOSE_CONFIG=%TEMP%\intergrax_lkw_file_watcher_e2e_compose_%RANDOM%%RANDOM%.yml"

set "LKW_BASE_URL=%LOCAL_WORKSPACE_BACKEND_BASE_URL%"
if "%LKW_BASE_URL%"=="" set "LKW_BASE_URL=http://127.0.0.1:8020"

set "KAFKA_BOOTSTRAP=%LKW_FILE_WATCHER_E2E_KAFKA_BOOTSTRAP%"
if "%KAFKA_BOOTSTRAP%"=="" set "KAFKA_BOOTSTRAP=127.0.0.1:9094"

set "TASK_TOPIC=intergrax.tasks"

where docker >nul 2>nul
if errorlevel 1 (
    echo proof_result=FAIL
    echo failure_reason=docker_not_available
    exit /b 1
)

if not exist "%PROOF%" (
    echo proof_result=FAIL
    echo failure_reason=proof_script_missing
    exit /b 1
)

if not exist "%BASE_COMPOSE%" (
    echo proof_result=FAIL
    echo failure_reason=base_compose_missing
    exit /b 1
)

if not exist "%KAFKA_COMPOSE%" (
    echo proof_result=FAIL
    echo failure_reason=kafka_compose_missing
    exit /b 1
)

if not exist "%WATCHER_COMPOSE%" (
    echo proof_result=FAIL
    echo failure_reason=watcher_compose_missing
    exit /b 1
)

pushd "%REPO_ROOT%" >nul
if errorlevel 1 (
    echo proof_result=FAIL
    echo failure_reason=repo_root_unavailable
    exit /b 1
)

echo LKW.7C1 watcher-triggered persistent search E2E proof
echo Repository root: %CD%
echo LKW base URL: %LKW_BASE_URL%
echo.

if not exist "%PROOF_DOCS_DIR%" (
    mkdir "%PROOF_DOCS_DIR%"
    if errorlevel 1 (
        echo proof_result=FAIL
        echo failure_reason=proof_docs_prepare_failed
        goto proof_fail
    )
)

echo Resetting dedicated watcher proof checkpoint state...
if exist "%WATCHER_STATE_DIR%" (
    rmdir /s /q "%WATCHER_STATE_DIR%"
    if errorlevel 1 (
        echo proof_result=FAIL
        echo failure_reason=watcher_state_reset_failed
        goto proof_fail
    )
)
mkdir "%WATCHER_STATE_DIR%"
if errorlevel 1 (
    echo proof_result=FAIL
    echo failure_reason=watcher_state_reset_failed
    goto proof_fail
)
echo watcher_state_reset=true

echo.
echo Validating Docker Compose merge...
docker compose -f "%BASE_COMPOSE%" -f "%KAFKA_COMPOSE%" -f "%WATCHER_COMPOSE%" config > "%COMPOSE_CONFIG%"
if errorlevel 1 goto proof_fail
echo compose_overlay_valid=true
del /f /q "%COMPOSE_CONFIG%" >nul 2>nul

echo.
echo Starting watcher E2E proof stack...
docker compose -f "%BASE_COMPOSE%" -f "%KAFKA_COMPOSE%" -f "%WATCHER_COMPOSE%" up -d --build local_workspace lkw-background-worker lkw-file-watcher lkw-kafka lkw-kafka-topics lkw-redis qdrant ollama
if errorlevel 1 goto proof_fail

echo Waiting for LKW health...
set "LKW_HEALTH_URL=%LKW_BASE_URL%/health"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $url = $env:LKW_HEALTH_URL; $deadline = (Get-Date).AddSeconds(240); do { try { $response = Invoke-RestMethod -Method Get -Uri $url -TimeoutSec 5; if ($response.status -eq 'ok') { Write-Host 'lkw_health=ok'; exit 0 } } catch { Start-Sleep -Seconds 2 } Start-Sleep -Seconds 2 } while ((Get-Date) -lt $deadline); throw 'LKW health check did not pass before timeout'"
if errorlevel 1 goto proof_fail

echo Waiting for watcher baseline checkpoint...
set "BASE_COMPOSE=%BASE_COMPOSE%"
set "KAFKA_COMPOSE=%KAFKA_COMPOSE%"
set "WATCHER_COMPOSE=%WATCHER_COMPOSE%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $deadline = (Get-Date).AddSeconds(240); do { $status = docker compose -f $env:BASE_COMPOSE -f $env:KAFKA_COMPOSE -f $env:WATCHER_COMPOSE ps --format json lkw-file-watcher 2>$null | ConvertFrom-Json; $running = $false; if ($null -ne $status) { if ($status -is [System.Array]) { $status = $status[0] }; if ($status.State -eq 'running' -or ($status.Status -and $status.Status.ToLower().StartsWith('up'))) { $running = $true } }; if ($running -and $status.Health -eq 'healthy') { Write-Host 'watcher_container_running=true'; Write-Host 'watcher_checkpoint_ready=true'; exit 0 }; Start-Sleep -Seconds 2 } while ((Get-Date) -lt $deadline); throw 'Watcher baseline health check did not pass before timeout'"
if errorlevel 1 goto proof_fail

echo.
echo Invoking Python watcher E2E proof workload...
uv run python "%PROOF%" --base-url "%LKW_BASE_URL%" --kafka-bootstrap "%KAFKA_BOOTSTRAP%" --topic "%TASK_TOPIC%" --repo-root "%CD%" --proof-docs-dir "%PROOF_DOCS_DIR%" --base-compose "%BASE_COMPOSE%" --kafka-compose "%KAFKA_COMPOSE%" --watcher-compose "%WATCHER_COMPOSE%" %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" goto proof_fail

echo.
echo LKW.7C1 workload complete. Stack left running for inspection.
del /f /q "%COMPOSE_CONFIG%" >nul 2>nul
popd >nul
exit /b 0

:proof_fail
echo.
echo proof_result=FAIL
del /f /q "%COMPOSE_CONFIG%" >nul 2>nul
popd >nul
exit /b 1
