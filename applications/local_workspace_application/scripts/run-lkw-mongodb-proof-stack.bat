@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\..\.."
set "DOCKER_DIR=%REPO_ROOT%\applications\local_workspace_application\docker"
set "BASE_COMPOSE=%DOCKER_DIR%\docker-compose.yml"
set "MONGODB_COMPOSE=%DOCKER_DIR%\docker-compose.mongodb.yml"
set "VERIFY=%SCRIPT_DIR%verify_lkw_mongodb_stack.py"
set "COMPOSE_CONFIG=%TEMP%\intergrax_lkw_mongodb_compose_%RANDOM%%RANDOM%.yml"

set "LKW_BASE_URL=%LOCAL_WORKSPACE_BACKEND_BASE_URL%"
if "%LKW_BASE_URL%"=="" set "LKW_BASE_URL=http://127.0.0.1:8020"

set "MONGO_EXPRESS_URL=%LKW_MONGO_EXPRESS_URL%"
if "%MONGO_EXPRESS_URL%"=="" set "MONGO_EXPRESS_URL=http://127.0.0.1:8086"

set "LKW_MONGODB_HOST_PORT=%LKW_MONGODB_HOST_PORT%"
if "%LKW_MONGODB_HOST_PORT%"=="" set "LKW_MONGODB_HOST_PORT=27018"

if not exist "%VERIFY%" (
    echo Missing MongoDB proof validator: %VERIFY%
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
    echo proof_result=FAIL
    echo reason=repository_root_unavailable
    exit /b 1
)

echo LKW MongoDB document-store platform proof helper (PROOF-RECEIPTS-1D)
echo Repository root: %CD%
echo LKW base URL: %LKW_BASE_URL%
echo Mongo Express URL: %MONGO_EXPRESS_URL%
echo MongoDB host port: %LKW_MONGODB_HOST_PORT%
echo.

echo Step 1/6: validating Docker Compose overlay...
docker compose -f "%BASE_COMPOSE%" -f "%MONGODB_COMPOSE%" config > "%COMPOSE_CONFIG%"
if errorlevel 1 goto proof_fail

uv run --extra integrations-mongodb python "%VERIFY%" --verify-volume-configured --volume-only --compose-config "%COMPOSE_CONFIG%"
if errorlevel 1 goto proof_fail
echo compose_overlay_valid=true

echo.
echo Step 2/6: starting MongoDB overlay services...
docker compose -f "%BASE_COMPOSE%" -f "%MONGODB_COMPOSE%" up -d --build lkw-mongodb lkw-mongo-express local_workspace
if errorlevel 1 goto proof_fail

echo Waiting for MongoDB health...
set "MONGODB_COMPOSE_FILES=-f \"%BASE_COMPOSE%\" -f \"%MONGODB_COMPOSE%\""
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $deadline = (Get-Date).AddSeconds(180); do { $status = docker compose -f $env:BASE_COMPOSE -f $env:MONGODB_COMPOSE ps --format json lkw-mongodb 2>$null | ConvertFrom-Json; if ($status.Health -eq 'healthy') { Write-Host 'mongodb_container_healthy=true'; exit 0 }; Start-Sleep -Seconds 2 } while ((Get-Date) -lt $deadline); throw 'MongoDB health check did not pass before timeout'"
if errorlevel 1 goto proof_fail

echo Waiting for Mongo Express HTTP endpoint...
set "LKW_MONGO_EXPRESS_CHECK_URL=%MONGO_EXPRESS_URL%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $url = $env:LKW_MONGO_EXPRESS_CHECK_URL; $deadline = (Get-Date).AddSeconds(120); do { try { $response = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 5; if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) { Write-Host 'mongo_express_available=true'; exit 0 } } catch { Start-Sleep -Seconds 2 } Start-Sleep -Seconds 2 } while ((Get-Date) -lt $deadline); throw 'Mongo Express did not become reachable before timeout'"
if errorlevel 1 goto proof_fail
echo mongo_express_configured=true

echo Waiting for LKW health...
set "LKW_HEALTH_URL=%LKW_BASE_URL%/health"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $url = $env:LKW_HEALTH_URL; $deadline = (Get-Date).AddSeconds(180); do { try { $response = Invoke-RestMethod -Method Get -Uri $url -TimeoutSec 5; if ($response.status -eq 'ok') { Write-Host 'lkw_health=ok'; exit 0 } } catch { Start-Sleep -Seconds 2 } Start-Sleep -Seconds 2 } while ((Get-Date) -lt $deadline); throw 'LKW health check did not pass before timeout'"
if errorlevel 1 goto proof_fail

echo.
echo Step 3/6: platform-backed MongoDB smoke write/read...
if "%LKW_MONGODB_ROOT_USERNAME%"=="" set "LKW_MONGODB_ROOT_USERNAME=intergrax"
if "%LKW_MONGODB_ROOT_PASSWORD%"=="" set "LKW_MONGODB_ROOT_PASSWORD=intergrax-local-dev-only"
if "%LKW_MONGODB_DATABASE%"=="" set "LKW_MONGODB_DATABASE=intergrax_proofs"
if "%LKW_MONGODB_COLLECTION%"=="" set "INTERGRAX_MONGODB_COLLECTION=proof_receipts"
set "INTERGRAX_MONGODB_URI=mongodb://%LKW_MONGODB_ROOT_USERNAME%:%LKW_MONGODB_ROOT_PASSWORD%@127.0.0.1:%LKW_MONGODB_HOST_PORT%/%LKW_MONGODB_DATABASE%?authSource=admin"

uv run --extra integrations-mongodb python "%VERIFY%" --mode smoke
if errorlevel 1 goto proof_fail

echo.
echo Step 4/6: restarting MongoDB container (volume retained)...
docker compose -f "%BASE_COMPOSE%" -f "%MONGODB_COMPOSE%" restart lkw-mongodb
if errorlevel 1 goto proof_fail

echo Waiting for MongoDB health after restart...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $deadline = (Get-Date).AddSeconds(180); do { $status = docker compose -f $env:BASE_COMPOSE -f $env:MONGODB_COMPOSE ps --format json lkw-mongodb 2>$null | ConvertFrom-Json; if ($status.Health -eq 'healthy') { Write-Host 'mongodb_container_healthy=true'; exit 0 }; Start-Sleep -Seconds 2 } while ((Get-Date) -lt $deadline); throw 'MongoDB health check did not pass after restart'"
if errorlevel 1 goto proof_fail

echo.
echo Step 5/6: persistence read-back through platform provider...
uv run --extra integrations-mongodb python "%VERIFY%" --mode read-only
if errorlevel 1 goto proof_fail
echo persistent_volume=true

echo.
echo Step 6/6: proof summary
echo proof_result=PASS
echo proof_kind=platform_document_store_infrastructure
echo document_store_provider=mongodb
echo integration_kind=document_store
echo adapter_resolved=true
echo mongodb_container_healthy=true
echo mongo_express_configured=true
echo platform_put=true
echo platform_get=true
echo smoke_record_verified=true
echo persistent_volume_configured=true
echo direct_mongosh_write=false
echo direct_pymongo_from_lkw=false
echo proof_receipt_recording=false
echo lkw_url=%LKW_BASE_URL%
echo mongo_express_url=%MONGO_EXPRESS_URL%
echo.
echo Mongo Express reviewer UI:
echo   %MONGO_EXPRESS_URL%
echo LKW URL:
echo   %LKW_BASE_URL%
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
