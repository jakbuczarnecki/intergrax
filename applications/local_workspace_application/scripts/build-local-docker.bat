@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "APP_DIR=%SCRIPT_DIR%.."
set "COMPOSE_FILE=%APP_DIR%\docker\docker-compose.yml"
set "COMPOSE_PROJECT_NAME=intergrax_lkw"

pushd "%APP_DIR%\..\.." >nul
if errorlevel 1 (
    echo Failed to locate repository root.
    exit /b 1
)

echo Materializing minimal runtime context for local_workspace_application...
uv run python scripts/build/build_application_image.py --application local_workspace_application --context-dir applications/local_workspace_application/docker/runtime-context --materialize-only
if errorlevel 1 goto fail

echo Building and starting complete LKW stack...
docker compose -p "%COMPOSE_PROJECT_NAME%" -f "%COMPOSE_FILE%" up --build -d
if errorlevel 1 goto fail

if not defined INTERGRAX_LLM_MODEL for /f "tokens=1,* delims==" %%A in ('docker compose -p "%COMPOSE_PROJECT_NAME%" -f "%COMPOSE_FILE%" config --environment 2^>nul') do if /I "%%A"=="INTERGRAX_LLM_MODEL" set "INTERGRAX_LLM_MODEL=%%B"
if not defined INTERGRAX_LLM_MODEL set "INTERGRAX_LLM_MODEL=llama3.1:latest"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$model = $env:INTERGRAX_LLM_MODEL; if ($model -notmatch '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$') { exit 1 }"
if errorlevel 1 goto fail

echo Ensuring the configured Ollama generation model is available...
docker compose -p "%COMPOSE_PROJECT_NAME%" -f "%COMPOSE_FILE%" exec -T --env "INTERGRAX_LLM_MODEL=%INTERGRAX_LLM_MODEL%" ollama sh -c "ollama pull \"$INTERGRAX_LLM_MODEL\""
if errorlevel 1 goto fail

echo Waiting for LKW health...
set "LKW_HEALTH_URL=http://127.0.0.1:8020/health"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $deadline = (Get-Date).AddSeconds(240); do { try { $response = Invoke-RestMethod -Method Get -Uri $env:LKW_HEALTH_URL -TimeoutSec 5; if ($response.status -eq 'ok') { Write-Host 'lkw_health=ok'; exit 0 } } catch {} Start-Sleep -Seconds 2 } while ((Get-Date) -lt $deadline); throw 'LKW health check did not pass before timeout'"
if errorlevel 1 goto fail

echo.
echo LKW local stack is ready.
echo Services: local_workspace, lkw-mongodb, qdrant, ollama, otel-collector
echo Slack companion starts automatically when enabled in .env.
echo Test in Slack with: workspaces
echo Logs:
echo   docker compose -p %COMPOSE_PROJECT_NAME% -f "%COMPOSE_FILE%" logs -f local_workspace

popd >nul
exit /b 0

:fail
echo.
echo LKW local stack bootstrap failed.
echo Inspect with:
echo   docker compose -p %COMPOSE_PROJECT_NAME% -f "%COMPOSE_FILE%" ps
echo   docker compose -p %COMPOSE_PROJECT_NAME% -f "%COMPOSE_FILE%" logs --tail 200
popd >nul
exit /b 1
