@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "APP_DIR=%SCRIPT_DIR%.."
set "COMPOSE_FILE=%APP_DIR%\docker\docker-compose.yml"
set "ENV_FILE=%APP_DIR%\.env"
set "ENV_EXAMPLE=%APP_DIR%\.env.example"

pushd "%APP_DIR%\..\.." >nul

if not exist "%ENV_FILE%" (
    if not exist "%ENV_EXAMPLE%" (
        echo Missing .env and .env.example in %SCRIPT_DIR%
        popd >nul
        exit /b 1
    )
    copy "%ENV_EXAMPLE%" "%ENV_FILE%" >nul
    echo Created %ENV_FILE% from .env.example
)

set "MODEL="
for /f "usebackq tokens=1,* delims==" %%A in ("%ENV_FILE%") do (
    set "KEY=%%A"
    set "VALUE=%%B"
    set "KEY=!KEY: =!"
    if /i "!KEY!"=="INTERGRAX_DEFAULT_OLLAMA_MODEL" set "MODEL=!VALUE!"
)
if "!MODEL!"=="" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%ENV_FILE%") do (
        set "KEY=%%A"
        set "VALUE=%%B"
        set "KEY=!KEY: =!"
        if /i "!KEY!"=="INTERGRAX_LLM_MODEL" set "MODEL=!VALUE!"
    )
)
if "!MODEL!"=="" set "MODEL=llama3.1:latest"
set "MODEL=!MODEL:"=!"
set "MODEL=!MODEL:'=!"

echo Building Docker image...
docker compose -f "%COMPOSE_FILE%" build
if errorlevel 1 goto fail

echo Starting Ollama service...
docker compose -f "%COMPOSE_FILE%" up -d ollama
if errorlevel 1 goto fail

echo Pulling Ollama model: !MODEL!
call :pull_model
if errorlevel 1 goto fail

echo Starting local stack...
docker compose -f "%COMPOSE_FILE%" up -d
if errorlevel 1 goto fail

echo Stack is starting. Verify with:
echo   curl http://127.0.0.1:8020/health
echo   curl http://127.0.0.1:8020/v1/local_workspace/agents

popd >nul
exit /b 0

:pull_model
for /l %%I in (1,1,3) do (
    echo Ollama pull attempt %%I/3...
    docker compose -f "%COMPOSE_FILE%" exec -T ollama ollama pull "!MODEL!"
    if not errorlevel 1 exit /b 0
    timeout /t 5 /nobreak >nul
)
exit /b 1

:fail
echo Local Docker setup failed.
popd >nul
exit /b 1
