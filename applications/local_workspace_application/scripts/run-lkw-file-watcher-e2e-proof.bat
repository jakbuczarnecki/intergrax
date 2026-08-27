@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "APP_DIR=%SCRIPT_DIR%.."
set "REPO_ROOT=%SCRIPT_DIR%..\..\.."
set "DOCKER_DIR=%APP_DIR%\docker"
set "BASE_COMPOSE=%DOCKER_DIR%\docker-compose.yml"
set "KAFKA_COMPOSE=%DOCKER_DIR%\docker-compose.kafka.yml"
set "WATCHER_COMPOSE=%DOCKER_DIR%\file-watcher-e2e.compose.yml"
set "MONGODB_COMPOSE=%DOCKER_DIR%\docker-compose.mongodb.yml"
set "PROOF=%SCRIPT_DIR%run-lkw-file-watcher-e2e-proof.py"
set "LIFECYCLE=%SCRIPT_DIR%lkw_proof_compose_lifecycle.py"
set "PROOF_DOCS_DIR=%APP_DIR%\.proof_docs"
set "WATCHER_STATE_DIR=%APP_DIR%\.file_watcher_e2e_state"
set "COMPOSE_CONFIG=%TEMP%\intergrax_lkw_file_watcher_e2e_compose_%RANDOM%%RANDOM%.yml"
set "LKW_COMPOSE_PROJECT=lkw-file-watcher-e2e-proof"
set "LKW_COMPOSE_OWNERSHIP_ENTERED=false"
set "EXIT_CODE=1"

set "LKW_BASE_URL=%LOCAL_WORKSPACE_BACKEND_BASE_URL%"
if "%LKW_BASE_URL%"=="" set "LKW_BASE_URL=http://127.0.0.1:8020"

set "KAFKA_BOOTSTRAP=%LKW_FILE_WATCHER_E2E_KAFKA_BOOTSTRAP%"
if "%KAFKA_BOOTSTRAP%"=="" set "KAFKA_BOOTSTRAP=127.0.0.1:9094"

set "KAFKA_UI_URL=%LKW_FILE_WATCHER_E2E_KAFKA_UI_URL%"
if "%KAFKA_UI_URL%"=="" set "KAFKA_UI_URL=http://127.0.0.1:8085"

set "MONGO_EXPRESS_URL=%LKW_MONGO_EXPRESS_URL%"
if "%MONGO_EXPRESS_URL%"=="" set "MONGO_EXPRESS_URL=http://127.0.0.1:8086"

set "LKW_MONGODB_HOST_PORT=%LKW_MONGODB_HOST_PORT%"
if "%LKW_MONGODB_HOST_PORT%"=="" set "LKW_MONGODB_HOST_PORT=27018"

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

if not exist "%MONGODB_COMPOSE%" (
    echo proof_result=FAIL
    echo failure_reason=mongodb_compose_missing
    exit /b 1
)

pushd "%REPO_ROOT%" >nul
if errorlevel 1 (
    echo proof_result=FAIL
    echo failure_reason=repo_root_unavailable
    exit /b 1
)

echo Establishing canonical Tier-3 source import roots...
for /f "usebackq delims=" %%I in (`uv run python "%SCRIPT_DIR%lkw_tier3_source_roots.py" --repo-root "%CD%" --format windows-path-list`) do set "PYTHONPATH=%%I"
if errorlevel 1 (
    echo proof_result=FAIL
    echo failure_reason=source_import_context_failed
    goto proof_fail
)
echo source_import_context=ready

echo LKW.7C2 watcher-triggered persistent search E2E proof with ProofReceipt
echo Repository root: %CD%
echo LKW base URL: %LKW_BASE_URL%
echo Kafka UI URL: %KAFKA_UI_URL%
echo Mongo Express URL: %MONGO_EXPRESS_URL%
echo MongoDB host port: %LKW_MONGODB_HOST_PORT%
echo Compose project: %LKW_COMPOSE_PROJECT%
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
echo Materializing minimal runtime context for local_workspace_application...
uv run python scripts/build/build_application_image.py --application local_workspace_application --context-dir applications/local_workspace_application/docker/runtime-context --materialize-only
if errorlevel 1 (
    echo proof_result=FAIL
    echo failure_reason=runtime_context_materialization_failed
    goto proof_fail
)
echo runtime_context_materialization=PASS

echo.
echo Validating Docker Compose merge...
docker compose -p "%LKW_COMPOSE_PROJECT%" -f "%BASE_COMPOSE%" -f "%KAFKA_COMPOSE%" -f "%WATCHER_COMPOSE%" -f "%MONGODB_COMPOSE%" config > "%COMPOSE_CONFIG%"
if errorlevel 1 goto proof_fail
echo compose_overlay_valid=true
del /f /q "%COMPOSE_CONFIG%" >nul 2>nul

echo.
echo Starting watcher E2E proof stack...
set "LKW_COMPOSE_OWNERSHIP_ENTERED=true"
docker compose -p "%LKW_COMPOSE_PROJECT%" -f "%BASE_COMPOSE%" -f "%KAFKA_COMPOSE%" -f "%WATCHER_COMPOSE%" -f "%MONGODB_COMPOSE%" up -d --build local_workspace lkw-background-worker lkw-file-watcher lkw-kafka lkw-kafka-topics lkw-kafka-ui lkw-redis qdrant ollama lkw-mongodb lkw-mongo-express
if errorlevel 1 goto proof_fail

echo Waiting for LKW health...
set "LKW_HEALTH_URL=%LKW_BASE_URL%/health"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $url = $env:LKW_HEALTH_URL; $deadline = (Get-Date).AddSeconds(240); do { try { $response = Invoke-RestMethod -Method Get -Uri $url -TimeoutSec 5; if ($response.status -eq 'ok') { Write-Host 'lkw_health=ok'; exit 0 } } catch { Start-Sleep -Seconds 2 } Start-Sleep -Seconds 2 } while ((Get-Date) -lt $deadline); throw 'LKW health check did not pass before timeout'"
if errorlevel 1 goto proof_fail

echo Waiting for watcher baseline checkpoint...
set "BASE_COMPOSE=%BASE_COMPOSE%"
set "KAFKA_COMPOSE=%KAFKA_COMPOSE%"
set "WATCHER_COMPOSE=%WATCHER_COMPOSE%"
set "MONGODB_COMPOSE=%MONGODB_COMPOSE%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $deadline = (Get-Date).AddSeconds(240); do { $status = docker compose -p $env:LKW_COMPOSE_PROJECT -f $env:BASE_COMPOSE -f $env:KAFKA_COMPOSE -f $env:WATCHER_COMPOSE -f $env:MONGODB_COMPOSE ps --format json lkw-file-watcher 2>$null | ConvertFrom-Json; $running = $false; if ($null -ne $status) { if ($status -is [System.Array]) { $status = $status[0] }; if ($status.State -eq 'running' -or ($status.Status -and $status.Status.ToLower().StartsWith('up'))) { $running = $true } }; if ($running -and $status.Health -eq 'healthy') { Write-Host 'watcher_container_running=true'; Write-Host 'watcher_checkpoint_ready=true'; exit 0 }; Start-Sleep -Seconds 2 } while ((Get-Date) -lt $deadline); throw 'Watcher baseline health check did not pass before timeout'"
if errorlevel 1 goto proof_fail

echo Waiting for MongoDB health...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $deadline = (Get-Date).AddSeconds(180); do { $status = docker compose -p $env:LKW_COMPOSE_PROJECT -f $env:BASE_COMPOSE -f $env:KAFKA_COMPOSE -f $env:WATCHER_COMPOSE -f $env:MONGODB_COMPOSE ps --format json lkw-mongodb 2>$null | ConvertFrom-Json; if ($null -ne $status) { if ($status -is [System.Array]) { $status = $status[0] }; if ($status.Health -eq 'healthy') { Write-Host 'mongodb_container_healthy=true'; exit 0 } }; Start-Sleep -Seconds 2 } while ((Get-Date) -lt $deadline); throw 'MongoDB health check did not pass before timeout'"
if errorlevel 1 goto proof_fail

echo Waiting for Mongo Express HTTP endpoint...
set "LKW_MONGO_EXPRESS_CHECK_URL=%MONGO_EXPRESS_URL%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $url = $env:LKW_MONGO_EXPRESS_CHECK_URL; $deadline = (Get-Date).AddSeconds(120); do { try { $response = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 5; if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) { Write-Host 'mongo_express_available=true'; exit 0 } } catch { Start-Sleep -Seconds 2 } Start-Sleep -Seconds 2 } while ((Get-Date) -lt $deadline); throw 'Mongo Express did not become reachable before timeout'"
if errorlevel 1 goto proof_fail

echo Waiting for Kafka UI...
set "LKW_KAFKA_UI_URL=%KAFKA_UI_URL%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $url = $env:LKW_KAFKA_UI_URL; $deadline = (Get-Date).AddSeconds(120); do { try { $response = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 5; if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) { Write-Host 'kafka_ui=ok'; exit 0 } } catch { Start-Sleep -Seconds 2 } Start-Sleep -Seconds 2 } while ((Get-Date) -lt $deadline); throw 'Kafka UI did not become reachable before timeout'"
if errorlevel 1 goto proof_fail

echo.
echo Invoking Python watcher E2E proof workload with receipt recording...
if "%LKW_MONGODB_ROOT_USERNAME%"=="" set "LKW_MONGODB_ROOT_USERNAME=intergrax"
if "%LKW_MONGODB_ROOT_PASSWORD%"=="" set "LKW_MONGODB_ROOT_PASSWORD=intergrax-local-dev-only"
if "%LKW_MONGODB_DATABASE%"=="" set "LKW_MONGODB_DATABASE=intergrax_proofs"
if "%LKW_MONGODB_COLLECTION%"=="" set "LKW_MONGODB_COLLECTION=proof_receipts"
set "INTERGRAX_MONGODB_URI=mongodb://%LKW_MONGODB_ROOT_USERNAME%:%LKW_MONGODB_ROOT_PASSWORD%@127.0.0.1:%LKW_MONGODB_HOST_PORT%/%LKW_MONGODB_DATABASE%?authSource=admin"
set "INTERGRAX_MONGODB_DATABASE=%LKW_MONGODB_DATABASE%"
set "INTERGRAX_MONGODB_COLLECTION=%LKW_MONGODB_COLLECTION%"

uv run --no-sync --project applications/local_workspace_application python "%PROOF%" --base-url "%LKW_BASE_URL%" --kafka-bootstrap "%KAFKA_BOOTSTRAP%" --topic "%TASK_TOPIC%" --repo-root "%CD%" --proof-docs-dir "%PROOF_DOCS_DIR%" --base-compose "%BASE_COMPOSE%" --kafka-compose "%KAFKA_COMPOSE%" --watcher-compose "%WATCHER_COMPOSE%" --mongodb-compose "%MONGODB_COMPOSE%" --mongo-express "%MONGO_EXPRESS_URL%" %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" goto proof_fail

echo.
echo LKW.7C2 proof complete.
echo Kafka UI URL:
echo   %KAFKA_UI_URL%
echo Mongo Express URL:
echo   %MONGO_EXPRESS_URL%
del /f /q "%COMPOSE_CONFIG%" >nul 2>nul
call :finalize_proof
popd >nul
exit /b %EXIT_CODE%

:proof_fail
echo.
echo proof_result=FAIL
del /f /q "%COMPOSE_CONFIG%" >nul 2>nul
call :finalize_proof
popd >nul
exit /b %EXIT_CODE%

:finalize_proof
if /I not "%LKW_COMPOSE_OWNERSHIP_ENTERED%"=="true" exit /b 0
uv run --no-sync --project applications/local_workspace_application python "%LIFECYCLE%" teardown --stack-id lkw-file-watcher-e2e-proof
set "TEARDOWN_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
    if not "%TEARDOWN_CODE%"=="0" set "EXIT_CODE=1"
)
exit /b 0
