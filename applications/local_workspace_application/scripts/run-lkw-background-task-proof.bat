@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\..\.."
set "DOCKER_DIR=%REPO_ROOT%\applications\local_workspace_application\docker"
set "BASE_COMPOSE=%DOCKER_DIR%\docker-compose.yml"
set "KAFKA_COMPOSE=%DOCKER_DIR%\docker-compose.kafka.yml"
set "MONGODB_COMPOSE=%DOCKER_DIR%\docker-compose.mongodb.yml"
set "PROOF=%SCRIPT_DIR%run-lkw-background-task-proof.py"
set "COMPOSE_CONFIG=%TEMP%\intergrax_lkw_background_task_compose_%RANDOM%%RANDOM%.yml"

set "LKW_BASE_URL=%LOCAL_WORKSPACE_BACKEND_BASE_URL%"
if "%LKW_BASE_URL%"=="" set "LKW_BASE_URL=http://127.0.0.1:8020"

set "KAFKA_UI_URL=%LKW_BACKGROUND_TASK_PROOF_KAFKA_UI_URL%"
if "%KAFKA_UI_URL%"=="" set "KAFKA_UI_URL=http://127.0.0.1:8085"

set "MONGO_EXPRESS_URL=%LKW_MONGO_EXPRESS_URL%"
if "%MONGO_EXPRESS_URL%"=="" set "MONGO_EXPRESS_URL=http://127.0.0.1:8086"

set "LKW_MONGODB_HOST_PORT=%LKW_MONGODB_HOST_PORT%"
if "%LKW_MONGODB_HOST_PORT%"=="" set "LKW_MONGODB_HOST_PORT=27018"

if not exist "%PROOF%" (
    echo Missing proof helper: %PROOF%
    exit /b 1
)

where docker >nul 2>nul
if errorlevel 1 (
    echo proof_result=FAIL
    echo reason=docker_not_available
    exit /b 1
)

pushd "%REPO_ROOT%" >nul
if errorlevel 1 (
    echo Failed to enter repository root.
    exit /b 1
)

echo LKW Kafka background-task platform proof helper (PROOF-RECEIPTS-1E)
echo Repository root: %CD%
echo LKW base URL: %LKW_BASE_URL%
echo Kafka UI URL: %KAFKA_UI_URL%
echo Mongo Express URL: %MONGO_EXPRESS_URL%
echo MongoDB host port: %LKW_MONGODB_HOST_PORT%
echo.

set "PROOF_DOCS_DIR=%REPO_ROOT%\applications\local_workspace_application\.proof_docs"
if not exist "%PROOF_DOCS_DIR%" (
    mkdir "%PROOF_DOCS_DIR%"
    if errorlevel 1 (
        echo Failed to create proof docs directory: %PROOF_DOCS_DIR%
        exit /b 1
    )
)

echo Step 1/4: validating Docker Compose overlays...
docker compose -f "%BASE_COMPOSE%" -f "%KAFKA_COMPOSE%" -f "%MONGODB_COMPOSE%" config > "%COMPOSE_CONFIG%"
if errorlevel 1 goto proof_fail
echo compose_overlay_valid=true

echo.
echo Step 2/4: starting combined Kafka + MongoDB proof stack...
docker compose -f "%BASE_COMPOSE%" -f "%KAFKA_COMPOSE%" -f "%MONGODB_COMPOSE%" up -d --build local_workspace lkw-background-worker lkw-kafka lkw-kafka-topics lkw-kafka-ui lkw-redis lkw-mongodb lkw-mongo-express
if errorlevel 1 goto proof_fail

echo Waiting for LKW health...
set "LKW_HEALTH_URL=%LKW_BASE_URL%/health"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $url = $env:LKW_HEALTH_URL; $deadline = (Get-Date).AddSeconds(180); do { try { $response = Invoke-RestMethod -Method Get -Uri $url -TimeoutSec 5; if ($response.status -eq 'ok') { Write-Host 'lkw_health=ok'; exit 0 } } catch { Start-Sleep -Seconds 2 } Start-Sleep -Seconds 2 } while ((Get-Date) -lt $deadline); throw 'LKW health check did not pass before timeout'"
if errorlevel 1 goto proof_fail

echo Waiting for MongoDB health...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $deadline = (Get-Date).AddSeconds(180); do { $status = docker compose -f $env:BASE_COMPOSE -f $env:KAFKA_COMPOSE -f $env:MONGODB_COMPOSE ps --format json lkw-mongodb 2>$null | ConvertFrom-Json; if ($status.Health -eq 'healthy') { Write-Host 'mongodb_container_healthy=true'; exit 0 }; Start-Sleep -Seconds 2 } while ((Get-Date) -lt $deadline); throw 'MongoDB health check did not pass before timeout'"
if errorlevel 1 goto proof_fail

echo Waiting for Kafka UI...
set "LKW_KAFKA_UI_URL=%KAFKA_UI_URL%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $url = $env:LKW_KAFKA_UI_URL; $deadline = (Get-Date).AddSeconds(120); do { try { $response = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 5; if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) { Write-Host 'kafka_ui=ok'; exit 0 } } catch { Start-Sleep -Seconds 2 } Start-Sleep -Seconds 2 } while ((Get-Date) -lt $deadline); throw 'Kafka UI did not become reachable before timeout'"
if errorlevel 1 goto proof_fail

echo Waiting for Mongo Express HTTP endpoint...
set "LKW_MONGO_EXPRESS_CHECK_URL=%MONGO_EXPRESS_URL%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $url = $env:LKW_MONGO_EXPRESS_CHECK_URL; $deadline = (Get-Date).AddSeconds(120); do { try { $response = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 5; if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) { Write-Host 'mongo_express_available=true'; exit 0 } } catch { Start-Sleep -Seconds 2 } Start-Sleep -Seconds 2 } while ((Get-Date) -lt $deadline); throw 'Mongo Express did not become reachable before timeout'"
if errorlevel 1 goto proof_fail

echo.
echo Step 3/4: executing background-task platform proof with receipt recording...
if "%LKW_MONGODB_ROOT_USERNAME%"=="" set "LKW_MONGODB_ROOT_USERNAME=intergrax"
if "%LKW_MONGODB_ROOT_PASSWORD%"=="" set "LKW_MONGODB_ROOT_PASSWORD=intergrax-local-dev-only"
if "%LKW_MONGODB_DATABASE%"=="" set "LKW_MONGODB_DATABASE=intergrax_proofs"
if "%LKW_MONGODB_COLLECTION%"=="" set "LKW_MONGODB_COLLECTION=proof_receipts"
set "INTERGRAX_MONGODB_URI=mongodb://%LKW_MONGODB_ROOT_USERNAME%:%LKW_MONGODB_ROOT_PASSWORD%@127.0.0.1:%LKW_MONGODB_HOST_PORT%/%LKW_MONGODB_DATABASE%?authSource=admin"
set "INTERGRAX_MONGODB_DATABASE=%LKW_MONGODB_DATABASE%"
set "INTERGRAX_MONGODB_COLLECTION=%LKW_MONGODB_COLLECTION%"

uv run --extra integrations-mongodb python "%PROOF%" --base-url "%LKW_BASE_URL%" --kafka-ui "%KAFKA_UI_URL%" --mongo-express "%MONGO_EXPRESS_URL%" %*
set "EXIT_CODE=%ERRORLEVEL%"
if errorlevel 1 goto proof_fail

echo.
echo Step 4/4: reviewer inspection endpoints
echo Kafka UI URL:
echo   %KAFKA_UI_URL%
echo Mongo Express URL:
echo   %MONGO_EXPRESS_URL%
echo Topics:
echo   intergrax.tasks
echo   intergrax.task-events
echo   intergrax.task-status
echo   intergrax.task-results
echo MongoDB database/collection:
echo   intergrax_proofs / proof_receipts
echo.
del /f /q "%COMPOSE_CONFIG%" >nul 2>nul
popd >nul
exit /b 0

:proof_fail
echo.
echo proof_result=FAIL
del /f /q "%COMPOSE_CONFIG%" >nul 2>nul
popd >nul
exit /b 1
