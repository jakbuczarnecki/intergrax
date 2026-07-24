@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "APP_DIR=%SCRIPT_DIR%.."
set "COMPOSE_FILE=%APP_DIR%\docker\docker-compose.yml"
set "COMPOSE_PROJECT_NAME=intergrax_lkw"
set "LEGACY_COMPOSE_PROJECT_NAME=intergrax_local_workspace"

pushd "%APP_DIR%\..\.." >nul
if errorlevel 1 (
    echo Failed to locate repository root.
    exit /b 1
)

echo Materializing minimal runtime context for local_workspace_application...
uv run python scripts/build/build_application_image.py --application local_workspace_application --context-dir applications/local_workspace_application/docker/runtime-context --materialize-only
if errorlevel 1 (
    popd >nul
    exit /b 1
)

echo Stopping previous local workspace stack...
docker compose -p "%COMPOSE_PROJECT_NAME%" -f "%COMPOSE_FILE%" down --remove-orphans
if errorlevel 1 (
    popd >nul
    exit /b 1
)

if /I not "%LEGACY_COMPOSE_PROJECT_NAME%"=="%COMPOSE_PROJECT_NAME%" (
    echo Cleaning legacy local workspace stack...
    docker compose -p "%LEGACY_COMPOSE_PROJECT_NAME%" -f "%COMPOSE_FILE%" down --remove-orphans
    if errorlevel 1 (
        popd >nul
        exit /b 1
    )
)

echo Building and starting local workspace via Docker Compose...
docker compose -p "%COMPOSE_PROJECT_NAME%" -f "%COMPOSE_FILE%" up --build -d
if errorlevel 1 (
    popd >nul
    exit /b 1
)

echo Stack is starting under Compose project %COMPOSE_PROJECT_NAME%. Verify with:
echo   curl http://127.0.0.1:8020/health
echo   docker compose -p %COMPOSE_PROJECT_NAME% -f "%COMPOSE_FILE%" logs -f local_workspace

popd >nul
endlocal
