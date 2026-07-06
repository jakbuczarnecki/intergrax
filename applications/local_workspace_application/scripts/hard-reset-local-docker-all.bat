@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
set "LAUNCHER=%SCRIPT_DIR%hard-reset-local-docker-all.ps1"

if not exist "%LAUNCHER%" (
    echo Missing PowerShell hard reset launcher: %LAUNCHER%
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%LAUNCHER%"
exit /b %ERRORLEVEL%
