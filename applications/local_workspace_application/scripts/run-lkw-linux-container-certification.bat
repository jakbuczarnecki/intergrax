@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\..\.."
set "ORCH=%SCRIPT_DIR%run-lkw-linux-container-certification.py"

if not exist "%ORCH%" (
    echo Missing Linux container certification orchestrator: %ORCH%
    exit /b 1
)

pushd "%REPO_ROOT%" >nul
if errorlevel 1 (
    echo Failed to enter repository root.
    exit /b 1
)

set "PYTHONUNBUFFERED=1"

where python >nul 2>nul
if not errorlevel 1 (
    python "%ORCH%" --pre-commit-certification %*
    set "EXIT_CODE=%ERRORLEVEL%"
    popd >nul
    exit /b %EXIT_CODE%
)

where py >nul 2>nul
if not errorlevel 1 (
    py -3 "%ORCH%" --pre-commit-certification %*
    set "EXIT_CODE=%ERRORLEVEL%"
    popd >nul
    exit /b %EXIT_CODE%
)

echo Python was not found on PATH.
popd >nul
exit /b 1
