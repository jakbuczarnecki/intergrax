@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\..\.."
set "RUNNER=%SCRIPT_DIR%run-lkw-product-quickstart.py"

where uv >nul 2>nul
if errorlevel 1 (
    echo uv was not found on PATH.
    exit /b 1
)

if not exist "%RUNNER%" (
    echo Missing shared quickstart runner: %RUNNER%
    exit /b 1
)

pushd "%REPO_ROOT%" >nul
if errorlevel 1 (
    echo Failed to enter repository root.
    exit /b 1
)

set "PYTHONUNBUFFERED=1"
uv run --project applications/local_workspace_application python "%RUNNER%" --os-family windows --wrapper-id windows_bat %*
set "EXIT_CODE=%ERRORLEVEL%"

popd >nul
exit /b %EXIT_CODE%
