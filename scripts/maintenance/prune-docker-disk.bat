@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
set "LAUNCHER=%SCRIPT_DIR%prune-docker-disk.ps1"

if not exist "%LAUNCHER%" (
    echo Missing PowerShell launcher: %LAUNCHER%
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%LAUNCHER%" %*
exit /b %ERRORLEVEL%
