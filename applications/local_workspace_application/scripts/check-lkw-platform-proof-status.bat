@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
set "CHECKER=%SCRIPT_DIR%check-lkw-platform-proof-status.ps1"

if not exist "%CHECKER%" (
    echo proof_status=FAIL
    echo reason=missing_powershell_checker
    echo checker=%CHECKER%
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%CHECKER%"
exit /b %ERRORLEVEL%
