@echo off
setlocal

REM Run from repository root so paths match GitHub Actions.
cd /d "%~dp0\..\.."

echo [INFO] Running pre-push CI smoke gate...
echo [INFO] Command: uv run python scripts/ci/run_regression_gate_ci.py --profile smoke
uv run python scripts/ci/run_regression_gate_ci.py --profile smoke
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" (
    echo [ERROR] Pre-push CI smoke gate failed. Do not push.
    exit /b %EXIT_CODE%
)

echo [OK] Pre-push CI smoke gate passed.
exit /b 0
