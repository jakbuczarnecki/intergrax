@echo off
setlocal

set "APP_DIR=%~dp0"
set "REPO_ROOT=%APP_DIR%..\.."
set "DEFAULT_LOG=%APP_DIR%.observability\otel\lkw-otlp-logs.jsonl"

pushd "%REPO_ROOT%" >nul
uv run python "%APP_DIR%scripts\inspect_otlp_logs.py" --file "%DEFAULT_LOG%" %*
set "EXIT_CODE=%ERRORLEVEL%"
popd >nul

exit /b %EXIT_CODE%
