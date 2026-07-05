@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\..\.."
set "PROOF_SCRIPT=%SCRIPT_DIR%run-sentry-observability-proof.py"

set "RUN_ID="
set "CORRELATION_ID="

:parse_args
if "%~1"=="" goto run_proof
if /i "%~1"=="--run-id" (
    set "RUN_ID=%~2"
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--correlation-id" (
    set "CORRELATION_ID=%~2"
    shift
    shift
    goto parse_args
)
if "%RUN_ID%"=="" set "RUN_ID=%~1"
shift
goto parse_args

:run_proof
if not exist "%PROOF_SCRIPT%" (
    echo Missing proof script: %PROOF_SCRIPT%
    exit /b 1
)

pushd "%REPO_ROOT%" >nul

echo LKW Sentry observability proof helper
echo Repository root: %CD%
echo.

set "ARGS="
if not "%RUN_ID%"=="" set "ARGS=%ARGS% --run-id %RUN_ID%"
if not "%CORRELATION_ID%"=="" set "ARGS=%ARGS% --correlation-id %CORRELATION_ID%"

uv run python "%PROOF_SCRIPT%" %ARGS%
set "STATUS=%ERRORLEVEL%"

echo.
if "%STATUS%"=="0" (
    echo Sentry search filters:
    echo   tag:intergrax.problem_kind=lkw.proof_controlled_failure
    echo   tag:intergrax.problem_error_code=LKW_PROOF_CONTROLLED_FAILURE
    if not "%RUN_ID%"=="" echo   tag:intergrax.run_id=%RUN_ID%
    if not "%CORRELATION_ID%"=="" echo   tag:intergrax.correlation_id=%CORRELATION_ID%
)

popd >nul
exit /b %STATUS%
