@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "APP_DIR=%SCRIPT_DIR%..\"
set "REPO_ROOT=%SCRIPT_DIR%..\..\.."
set "DEFAULT_LOG=%APP_DIR%.observability\otel\lkw-otlp-logs.jsonl"

pushd "%REPO_ROOT%" >nul
uv run python "%SCRIPT_DIR%inspect_otlp_logs.py" --file "%DEFAULT_LOG%" %*
set "EXIT_CODE=%ERRORLEVEL%"
popd >nul

exit /b %EXIT_CODE%
