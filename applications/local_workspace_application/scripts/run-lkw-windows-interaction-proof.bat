@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\..\.."
set "PROOF=%SCRIPT_DIR%run-lkw-os-interaction-proof.py"

where uv >nul 2>nul
if errorlevel 1 (
    echo uv was not found on PATH.
    exit /b 1
)

if not exist "%PROOF%" (
    echo Missing shared OS interaction proof runner: %PROOF%
    exit /b 1
)

pushd "%REPO_ROOT%" >nul
if errorlevel 1 (
    echo Failed to enter repository root.
    exit /b 1
)

set "PYTHONUNBUFFERED=1"
uv run --extra integrations-mongodb python "%PROOF%" --os-family windows %*
set "EXIT_CODE=%ERRORLEVEL%"

popd >nul
exit /b %EXIT_CODE%
