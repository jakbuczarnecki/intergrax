@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\..\.."
set "PROOF=%SCRIPT_DIR%run-lkw-background-task-proof.py"

if not exist "%PROOF%" (
    echo Missing proof helper: %PROOF%
    exit /b 1
)

pushd "%REPO_ROOT%" >nul
if errorlevel 1 (
    echo Failed to enter repository root.
    exit /b 1
)

echo LKW Kafka background-task platform proof helper
echo Repository root: %CD%
echo.

uv run python "%PROOF%" %*
set "EXIT_CODE=%ERRORLEVEL%"
popd >nul
exit /b %EXIT_CODE%
