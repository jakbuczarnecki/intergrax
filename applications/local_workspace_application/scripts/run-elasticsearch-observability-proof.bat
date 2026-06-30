@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\..\.."
set "INSPECTOR=%SCRIPT_DIR%inspect-elasticsearch-observability.bat"

set "LKW_HEALTH_URL=%LOCAL_WORKSPACE_OBSERVABILITY_PROOF_LKW_HEALTH_URL%"
if "%LKW_HEALTH_URL%"=="" set "LKW_HEALTH_URL=http://127.0.0.1:8020/health"

set "ES_URL=%LOCAL_WORKSPACE_OBSERVABILITY_PROOF_ES_URL%"
if "%ES_URL%"=="" set "ES_URL=http://127.0.0.1:9200"

set "ES_INDEX=%LOCAL_WORKSPACE_OBSERVABILITY_PROOF_ES_INDEX%"
if "%ES_INDEX%"=="" set "ES_INDEX=intergrax-lkw-observability"

set "RUN_ID="
if /i "%~1"=="--run-id" (
    set "RUN_ID=%~2"
) else (
    set "RUN_ID=%~1"
)

if not exist "%INSPECTOR%" (
    echo Missing inspector wrapper: %INSPECTOR%
    exit /b 1
)

pushd "%REPO_ROOT%" >nul

echo LKW Elasticsearch observability proof helper
echo Repository root: %CD%
echo LKW health URL: %LKW_HEALTH_URL%
echo Elasticsearch URL: %ES_URL%
echo Elasticsearch index: %ES_INDEX%
echo.

echo Checking LKW health...
curl -fsS "%LKW_HEALTH_URL%"
if errorlevel 1 goto health_fail
echo.
echo.

echo Checking Elasticsearch health...
curl -fsS "%ES_URL%/_cluster/health"
if errorlevel 1 goto health_fail
echo.
echo.

if "%RUN_ID%"=="" goto list_runs

echo Inspecting run_id: %RUN_ID%
echo.
call "%INSPECTOR%" --url "%ES_URL%" --index "%ES_INDEX%" --run-id "%RUN_ID%"
if errorlevel 1 goto proof_fail
echo.

echo Running duplicate check...
call "%INSPECTOR%" --url "%ES_URL%" --index "%ES_INDEX%" --run-id "%RUN_ID%" --check-duplicates
if errorlevel 1 goto proof_fail
echo.

echo Running safety-key check...
call "%INSPECTOR%" --url "%ES_URL%" --index "%ES_INDEX%" --run-id "%RUN_ID%" --check-safety
if errorlevel 1 goto proof_fail
echo.

echo Running combined proof check...
call "%INSPECTOR%" --url "%ES_URL%" --index "%ES_INDEX%" --run-id "%RUN_ID%" --check-duplicates --check-safety
if errorlevel 1 goto proof_fail
echo.

echo Proof result: PASS
echo.
echo Documentation summary:
echo   run_id=%RUN_ID%
echo   elasticsearch_url=%ES_URL%
echo   elasticsearch_index=%ES_INDEX%
echo   duplicate_check=0
echo   safety_check=passed
echo   command=%~nx0 %RUN_ID%
popd >nul
exit /b 0

:list_runs
echo Listing recent Elasticsearch observability runs...
call "%INSPECTOR%" --url "%ES_URL%" --index "%ES_INDEX%" --list-runs
if errorlevel 1 goto proof_fail
echo.
echo Next steps:
echo   1. Execute a real LKW run via Swagger or curl.
echo   2. Copy the resulting run_id, or use the latest run_id listed above.
echo   3. Run:
echo      applications\local_workspace_application\scripts\run-elasticsearch-observability-proof.bat ^<run_id^>
echo.
popd >nul
exit /b 0

:health_fail
echo.
echo Proof precheck failed. Start the full local stack first, for example:
echo   applications\local_workspace_application\scripts\run-local-docker-all.bat
popd >nul
exit /b 1

:proof_fail
echo.
echo Proof result: FAIL
echo Do not mark OBS-VENDOR-7 as Done until duplicate check and safety check pass for a real run_id.
popd >nul
exit /b 1
