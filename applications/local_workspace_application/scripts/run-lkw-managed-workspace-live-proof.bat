@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\..\.."
set "PROOF=%SCRIPT_DIR%run-lkw-managed-workspace-live-proof.py"

where uv >nul 2>nul
if errorlevel 1 (
    echo uv was not found on PATH.
    exit /b 1
)

if not exist "%PROOF%" (
    echo Missing managed workspace live proof helper: %PROOF%
    exit /b 1
)

pushd "%REPO_ROOT%" >nul
if errorlevel 1 (
    echo Failed to enter repository root.
    exit /b 1
)

set "PYTHONUNBUFFERED=1"
if "%LKW_MONGODB_HOST_PORT%"=="" set "LKW_MONGODB_HOST_PORT=27018"
if "%LKW_MONGODB_ROOT_USERNAME%"=="" set "LKW_MONGODB_ROOT_USERNAME=intergrax"
if "%LKW_MONGODB_ROOT_PASSWORD%"=="" set "LKW_MONGODB_ROOT_PASSWORD=intergrax-local-dev-only"
if "%LKW_MONGODB_DATABASE%"=="" set "LKW_MONGODB_DATABASE=intergrax_proofs"
if "%LKW_MONGODB_COLLECTION%"=="" set "LKW_MONGODB_COLLECTION=proof_receipts"
if "%LKW_MANAGED_WORKSPACE_COLLECTION%"=="" set "LKW_MANAGED_WORKSPACE_COLLECTION=lkw_managed_workspaces"

uv run --extra integrations-mongodb python "%PROOF%" %*
set "EXIT_CODE=%ERRORLEVEL%"

popd >nul
exit /b %EXIT_CODE%
