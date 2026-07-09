@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\..\.."
set "DOCKER_DIR=%REPO_ROOT%\applications\local_workspace_application\docker"
set "BASE_COMPOSE=%DOCKER_DIR%\docker-compose.yml"
set "KAFKA_COMPOSE=%DOCKER_DIR%\docker-compose.kafka.yml"
set "PROOF=%SCRIPT_DIR%run-lkw-background-task-proof.py"

set "LKW_BASE_URL=%LOCAL_WORKSPACE_BACKEND_BASE_URL%"
if "%LKW_BASE_URL%"=="" set "LKW_BASE_URL=http://127.0.0.1:8020"

set "KAFKA_UI_URL=%LKW_BACKGROUND_TASK_PROOF_KAFKA_UI_URL%"
if "%KAFKA_UI_URL%"=="" set "KAFKA_UI_URL=http://127.0.0.1:8085"

if not exist "%PROOF%" (
    echo Missing proof helper: %PROOF%
    exit /b 1
)

pushd "%REPO_ROOT%" >nul
if errorlevel 1 (
    echo Failed to enter repository root.
    exit /b 1
)

echo LKW Kafka background-task platform proof helper
echo Repository root: %CD%
echo LKW base URL: %LKW_BASE_URL%
echo Kafka UI URL: %KAFKA_UI_URL%
echo.

echo Step 1/3: starting Docker stack with Kafka overlay...
docker compose -f "%BASE_COMPOSE%" -f "%KAFKA_COMPOSE%" up -d --build local_workspace lkw-background-worker lkw-kafka lkw-kafka-topics lkw-kafka-ui lkw-redis
if errorlevel 1 goto proof_fail

echo Waiting for LKW health...
set "LKW_HEALTH_URL=%LKW_BASE_URL%/health"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $url = $env:LKW_HEALTH_URL; $deadline = (Get-Date).AddSeconds(180); do { try { $response = Invoke-RestMethod -Method Get -Uri $url -TimeoutSec 5; if ($response.status -eq 'ok') { Write-Host 'lkw_health=ok'; exit 0 } } catch { Start-Sleep -Seconds 2 } Start-Sleep -Seconds 2 } while ((Get-Date) -lt $deadline); throw 'LKW health check did not pass before timeout'"
if errorlevel 1 goto proof_fail

echo Waiting for Kafka UI...
set "LKW_KAFKA_UI_URL=%KAFKA_UI_URL%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $url = $env:LKW_KAFKA_UI_URL; $deadline = (Get-Date).AddSeconds(120); do { try { $response = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 5; if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) { Write-Host 'kafka_ui=ok'; exit 0 } } catch { Start-Sleep -Seconds 2 } Start-Sleep -Seconds 2 } while ((Get-Date) -lt $deadline); throw 'Kafka UI did not become reachable before timeout'"
if errorlevel 1 goto proof_fail

echo.
echo Step 2/3: executing background-task platform proof...
uv run python "%PROOF%" --base-url "%LKW_BASE_URL%" --kafka-ui "%KAFKA_UI_URL%" %*
set "EXIT_CODE=%ERRORLEVEL%"
if errorlevel 1 goto proof_fail

echo.
echo Step 3/3: open Kafka UI and inspect task topics.
echo Kafka UI URL:
echo   %KAFKA_UI_URL%
echo Topics:
echo   intergrax.tasks
echo   intergrax.task-events
echo   intergrax.task-status
echo   intergrax.task-results
echo.
popd >nul
exit /b 0

:proof_fail
echo.
echo proof_result=FAIL
popd >nul
exit /b 1
