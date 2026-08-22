@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\..\.."
set "DOCKER_DIR=%REPO_ROOT%\applications\local_workspace_application\docker"
set "BASE_COMPOSE=%DOCKER_DIR%\docker-compose.yml"
set "MONGODB_COMPOSE=%DOCKER_DIR%\docker-compose.mongodb.yml"
set "PROOF=%SCRIPT_DIR%run-lkw-hosting-proof.py"
set "LIFECYCLE=%SCRIPT_DIR%lkw_proof_compose_lifecycle.py"
set "COMPOSE_CONFIG=%TEMP%\intergrax_lkw_hosting_proof_compose_%RANDOM%%RANDOM%.yml"
set "LKW_COMPOSE_PROJECT=lkw-hosting-proof"
set "LKW_COMPOSE_OWNERSHIP_ENTERED=false"
set "EXIT_CODE=1"

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

echo LKW Application Hosting platform proof helper (APP-HOST-8E)
echo Repository root: %CD%
echo Mongo Express URL: %MONGO_EXPRESS_URL%
echo MongoDB host port: %LKW_MONGODB_HOST_PORT%
echo Compose project: %LKW_COMPOSE_PROJECT%
echo.
echo This proof starts only MongoDB and Mongo Express.
echo Accepted APP-HOST-8C/8D live tests own the hosted LKW processes.
echo.

echo Step 1/4: validating MongoDB proof overlay
docker compose -p "%LKW_COMPOSE_PROJECT%" -f "%BASE_COMPOSE%" -f "%MONGODB_COMPOSE%" config > "%COMPOSE_CONFIG%"
if errorlevel 1 goto proof_fail
echo compose_overlay_valid=true

echo.
echo Step 2/4: starting MongoDB and Mongo Express
set "LKW_COMPOSE_OWNERSHIP_ENTERED=true"
docker compose -p "%LKW_COMPOSE_PROJECT%" -f "%BASE_COMPOSE%" -f "%MONGODB_COMPOSE%" up -d lkw-mongodb lkw-mongo-express
if errorlevel 1 goto proof_fail

echo Waiting for MongoDB health...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $deadline = (Get-Date).AddSeconds(180); do { $status = docker compose -p $env:LKW_COMPOSE_PROJECT -f $env:BASE_COMPOSE -f $env:MONGODB_COMPOSE ps --format json lkw-mongodb 2>$null | ConvertFrom-Json; if ($status.Health -eq 'healthy') { Write-Host 'mongodb_container_healthy=true'; exit 0 }; Start-Sleep -Seconds 2 } while ((Get-Date) -lt $deadline); throw 'MongoDB health check did not pass before timeout'"
if errorlevel 1 goto proof_fail

echo Waiting for Mongo Express HTTP endpoint...
set "LKW_MONGO_EXPRESS_CHECK_URL=%MONGO_EXPRESS_URL%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $url = $env:LKW_MONGO_EXPRESS_CHECK_URL; $deadline = (Get-Date).AddSeconds(120); do { try { $response = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 5; if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) { Write-Host 'mongo_express_available=true'; exit 0 } } catch { Start-Sleep -Seconds 2 } Start-Sleep -Seconds 2 } while ((Get-Date) -lt $deadline); throw 'Mongo Express did not become reachable before timeout'"
if errorlevel 1 goto proof_fail

echo.
echo Step 3/4: executing accepted hosting tests and recording ProofReceipt
if "%LKW_MONGODB_ROOT_USERNAME%"=="" set "LKW_MONGODB_ROOT_USERNAME=intergrax"
if "%LKW_MONGODB_ROOT_PASSWORD%"=="" set "LKW_MONGODB_ROOT_PASSWORD=intergrax-local-dev-only"
if "%LKW_MONGODB_DATABASE%"=="" set "LKW_MONGODB_DATABASE=intergrax_proofs"
if "%LKW_MONGODB_COLLECTION%"=="" set "LKW_MONGODB_COLLECTION=proof_receipts"
set "INTERGRAX_MONGODB_URI=mongodb://%LKW_MONGODB_ROOT_USERNAME%:%LKW_MONGODB_ROOT_PASSWORD%@127.0.0.1:%LKW_MONGODB_HOST_PORT%/%LKW_MONGODB_DATABASE%?authSource=admin"
set "INTERGRAX_MONGODB_DATABASE=%LKW_MONGODB_DATABASE%"
set "INTERGRAX_MONGODB_COLLECTION=%LKW_MONGODB_COLLECTION%"

uv run --project applications/local_workspace_application python "%PROOF%" %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" goto proof_fail

echo.
echo Step 4/4: reviewer inspection
echo Mongo Express URL:
echo   %MONGO_EXPRESS_URL%
echo MongoDB database/collection:
echo   intergrax_proofs / proof_receipts
echo Filter:
echo   proof_kind = platform_application_hosting
echo   run_id = ^<printed proof_receipt_run_id^>
echo.
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
uv run --project applications/local_workspace_application python "%LIFECYCLE%" teardown --stack-id lkw-hosting-proof
set "TEARDOWN_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
    if not "%TEARDOWN_CODE%"=="0" set "EXIT_CODE=1"
)
exit /b 0
