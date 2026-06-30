@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\..\.."

pushd "%REPO_ROOT%" >nul
uv run python "%SCRIPT_DIR%inspect_elasticsearch_observability.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
popd >nul

exit /b %EXIT_CODE%
