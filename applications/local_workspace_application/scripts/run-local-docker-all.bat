@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "APP_DIR=%SCRIPT_DIR%.."
set "DOCKER_DIR=%APP_DIR%\docker"
set "BASE_COMPOSE=%DOCKER_DIR%\docker-compose.yml"
set "ENV_FILE=%APP_DIR%\.env"
set "ENV_EXAMPLE=%APP_DIR%\.env.example"

pushd "%APP_DIR%\..\.." >nul

if not exist "%BASE_COMPOSE%" (
    echo Missing base compose file: %BASE_COMPOSE%
    popd >nul
    exit /b 1
)

if not exist "%ENV_FILE%" (
    if not exist "%ENV_EXAMPLE%" (
        echo Missing .env and .env.example in %APP_DIR%
        popd >nul
        exit /b 1
    )
    copy "%ENV_EXAMPLE%" "%ENV_FILE%" >nul
    echo Created %ENV_FILE% from .env.example
)

set "COMPOSE_ARGS=-f ^"%BASE_COMPOSE%^""

echo Compose files:
echo   %BASE_COMPOSE%
for %%F in ("%DOCKER_DIR%\docker-compose.*.yml") do (
    if exist "%%~fF" (
        set "COMPOSE_ARGS=!COMPOSE_ARGS! -f ^"%%~fF^""
        echo   %%~fF
    )
)

if "%~1"=="" (
    echo Running: docker compose !COMPOSE_ARGS! up --build
    docker compose !COMPOSE_ARGS! up --build
) else (
    echo Running: docker compose !COMPOSE_ARGS! %*
    docker compose !COMPOSE_ARGS! %*
)

set "EXIT_CODE=%ERRORLEVEL%"
popd >nul
exit /b %EXIT_CODE%
