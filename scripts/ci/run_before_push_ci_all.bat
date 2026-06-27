@echo off
setlocal

REM Run from repository root so paths match GitHub Actions.
cd /d "%~dp0\..\.."

echo [INFO] Running full pre-push CI gate...
echo [INFO] Command: uv run python scripts/ci/run_regression_gate_ci.py --profile all
uv run python scripts/ci/run_regression_gate_ci.py --profile all
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" (
    echo [ERROR] Full pre-push CI gate failed. Do not push.
    exit /b %EXIT_CODE%
)

echo [OK] Full pre-push CI gate passed.
exit /b 0
