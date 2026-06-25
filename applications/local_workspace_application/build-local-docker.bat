@REM © Artur Czarnecki. All rights reserved.
@REM Intergrax framework – proprietary and confidential.
@REM Use, modification, or distribution without written permission is prohibited.
@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "COMPOSE_FILE=%SCRIPT_DIR%docker\docker-compose.yml"
set "ENV_FILE=%SCRIPT_DIR%.env"
set "ENV_EXAMPLE=%SCRIPT_DIR%.env.example"

pushd "%SCRIPT_DIR%\..\.." >nul

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

if "%MODEL%"=="" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%ENV_FILE%") do (
        set "KEY=%%A"
        set "VALUE=%%B"
        set "KEY=!KEY: =!"
        if /i "!KEY!"=="INTERGRAX_LLM_MODEL" set "MODEL=!VALUE!"
    )
)

if "%MODEL%"=="" set "MODEL=llama3.1:latest"
set "MODEL=%MODEL:"=%"
set "MODEL=%MODEL:'=%"

echo Building LKW Docker image and local services...
docker compose -f "%COMPOSE_FILE%" build
if errorlevel 1 goto fail

echo Starting Ollama service...
docker compose -f "%COMPOSE_FILE%" up -d ollama
if errorlevel 1 goto fail

echo Pulling Ollama model: %MODEL%
docker compose -f "%COMPOSE_FILE%" exec ollama ollama pull "%MODEL%"
if errorlevel 1 goto fail

echo Starting LKW local stack...
docker compose -f "%COMPOSE_FILE%" up -d
if errorlevel 1 goto fail

echo LKW stack is starting. Verify with:
echo   curl http://127.0.0.1:8020/health
echo   curl http://127.0.0.1:8020/v1/local_workspace/agents

popd >nul
exit /b 0

:fail
echo LKW local Docker setup failed.
popd >nul
exit /b 1
