@echo off
setlocal

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

set "LKW_COMPOSE_PROJECT=lkw-core-platform-proof"
set "LKW_COMPOSE_BASE=%BASE_COMPOSE%"
set "LKW_COMPOSE_DOCKER_DIR=%DOCKER_DIR%"
if "%~1"=="" (
    set "LKW_COMPOSE_COMMAND=up --build"
) else (
    set "LKW_COMPOSE_COMMAND=%*"
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $base = (Resolve-Path -LiteralPath $env:LKW_COMPOSE_BASE).Path; $dockerDir = (Resolve-Path -LiteralPath $env:LKW_COMPOSE_DOCKER_DIR).Path; $files = @($base) + @(Get-ChildItem -LiteralPath $dockerDir -Filter 'docker-compose.*.yml' | Sort-Object Name | ForEach-Object { $_.FullName }); Write-Host 'Compose files:'; $files | ForEach-Object { Write-Host ('  ' + $_) }; $composeArgs = @(); foreach ($file in $files) { $composeArgs += @('-f', $file) }; $commandArgs = if ([string]::IsNullOrWhiteSpace($env:LKW_COMPOSE_COMMAND)) { @('up', '--build') } else { $env:LKW_COMPOSE_COMMAND -split ' ' }; $dockerArgs = @('compose', '-p', $env:LKW_COMPOSE_PROJECT) + $composeArgs + $commandArgs; Write-Host ('Running: docker ' + ($dockerArgs -join ' ')); & docker @dockerArgs; exit $LASTEXITCODE"
set "EXIT_CODE=%ERRORLEVEL%"

popd >nul
exit /b %EXIT_CODE%
