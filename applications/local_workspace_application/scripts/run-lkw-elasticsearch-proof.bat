@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\..\.."
set "DOCKER_DIR=%REPO_ROOT%\applications\local_workspace_application\docker"
set "BASE_COMPOSE=%DOCKER_DIR%\docker-compose.yml"
set "ES_COMPOSE=%DOCKER_DIR%\docker-compose.elasticsearch.yml"
set "VALIDATOR=%SCRIPT_DIR%run-elasticsearch-observability-proof.bat"
set "RUN_ID_FILE=%TEMP%\intergrax_lkw_es_run_id_%RANDOM%%RANDOM%.txt"

set "LKW_BASE_URL=%LOCAL_WORKSPACE_BACKEND_BASE_URL%"
if "%LKW_BASE_URL%"=="" set "LKW_BASE_URL=http://127.0.0.1:8020"

set "KIBANA_URL=%LOCAL_WORKSPACE_OBSERVABILITY_PROOF_KIBANA_URL%"
if "%KIBANA_URL%"=="" set "KIBANA_URL=http://127.0.0.1:5601"

if not exist "%VALIDATOR%" (
    echo Missing Elasticsearch validator: %VALIDATOR%
    exit /b 1
)

pushd "%REPO_ROOT%" >nul
if errorlevel 1 (
    echo Failed to enter repository root.
    exit /b 1
)

echo LKW Elasticsearch/Kibana one-command proof helper
echo Repository root: %CD%
echo LKW base URL: %LKW_BASE_URL%
echo Kibana URL: %KIBANA_URL%
echo.
echo Step 1/4: switching local_workspace to Elasticsearch observability backend...
docker compose -f "%BASE_COMPOSE%" -f "%ES_COMPOSE%" up -d --build local_workspace
if errorlevel 1 goto proof_fail

echo.
echo Step 2/4: executing a real LKW run...

set "LKW_RUN_ID_FILE=%RUN_ID_FILE%"
set "LKW_PROOF_BASE_URL=%LKW_BASE_URL%"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $base = $env:LKW_PROOF_BASE_URL.TrimEnd('/'); $body = @{ message = 'Find documents about local workspace observability proof'; capability = 'local.workspace.search'; metadata = @{ proof = 'LKW_PLATFORM_PROOF'; proof_helper = 'run-lkw-elasticsearch-proof.bat' } } | ConvertTo-Json -Depth 5; $response = Invoke-RestMethod -Method Post -Uri ($base + '/v1/local_workspace/run') -ContentType 'application/json' -Body $body; if (-not $response.run_id) { throw 'LKW response did not include run_id' }; Set-Content -LiteralPath $env:LKW_RUN_ID_FILE -Value $response.run_id -NoNewline -Encoding ascii; Write-Host ('run_id=' + $response.run_id); if ($response.state) { Write-Host ('state=' + $response.state) }; if ($response.agent_id) { Write-Host ('agent_id=' + $response.agent_id) }"
if errorlevel 1 goto proof_fail

if not exist "%RUN_ID_FILE%" goto missing_run_id
set /p RUN_ID=<"%RUN_ID_FILE%"
del /f /q "%RUN_ID_FILE%" >nul 2>nul

if "%RUN_ID%"=="" goto missing_run_id

echo.
echo Step 3/4: validating Elasticsearch observability for run_id=%RUN_ID%...
call "%VALIDATOR%" "%RUN_ID%"
if errorlevel 1 goto proof_fail

echo.
echo Step 4/4: open Kibana and inspect this run.
echo Kibana URL:
echo   %KIBANA_URL%
echo.
echo Kibana Discover filter:
echo   intergrax.run_id: "%RUN_ID%"
echo.
echo Proof result: PASS
echo run_id=%RUN_ID%
echo kibana_url=%KIBANA_URL%
echo elasticsearch_validation=passed
echo.
popd >nul
exit /b 0

:missing_run_id
echo.
echo Proof result: FAIL
echo LKW run_id was not captured.
del /f /q "%RUN_ID_FILE%" >nul 2>nul
popd >nul
exit /b 1

:proof_fail
echo.
echo Proof result: FAIL
del /f /q "%RUN_ID_FILE%" >nul 2>nul
popd >nul
exit /b 1
