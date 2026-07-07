@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
set "PROOF=%SCRIPT_DIR%run-lkw-persistence-proof.ps1"

if not exist "%PROOF%" (
    echo proof_result=FAIL
    echo reason=missing_powershell_proof_helper
    echo helper=%PROOF%
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%PROOF%"
exit /b %ERRORLEVEL%
