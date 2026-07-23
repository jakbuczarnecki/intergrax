@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "APP_DIR=%SCRIPT_DIR%.."
set "COMPOSE_FILE=%APP_DIR%\docker\docker-compose.yml"

pushd "%APP_DIR%\..\.." >nul
if errorlevel 1 (
    echo Failed to locate repository root.
    exit /b 1
)

echo Materializing minimal runtime context for governed_contractor_application...
uv run python scripts/build/build_application_image.py --application governed_contractor_application --context-dir applications/governed_contractor_application/docker/runtime-context --materialize-only
if errorlevel 1 (
    popd >nul
    exit /b 1
)

echo Building and starting governed contractor via Docker Compose...
docker compose -f "%COMPOSE_FILE%" up --build -d
if errorlevel 1 (
    popd >nul
    exit /b 1
)

echo Stack is starting. Verify with:
echo   curl http://127.0.0.1:8000/health

popd >nul
endlocal
