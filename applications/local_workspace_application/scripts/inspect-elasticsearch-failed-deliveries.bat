@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "APP_DIR=%SCRIPT_DIR%..\"
set "REPO_ROOT=%SCRIPT_DIR%..\..\.."
set "DEFAULT_FILE=%APP_DIR%.observability\elasticsearch\failed-deliveries.jsonl"

pushd "%REPO_ROOT%" >nul
uv run python "%SCRIPT_DIR%inspect_elasticsearch_failed_deliveries.py" --file "%DEFAULT_FILE%" %*
set "EXIT_CODE=%ERRORLEVEL%"
popd >nul

exit /b %EXIT_CODE%
