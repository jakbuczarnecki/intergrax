@echo off
setlocal

REM © Artur Czarnecki. All rights reserved.
REM Sync GitHub repository description, homepage, and topics from .github/repository-metadata.json

cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv is not installed or not in PATH.
    echo Install it from: https://docs.astral.sh/uv/
    exit /b 1
)

set MODE=%1

if /I "%MODE%"=="apply" goto :apply
if /I "%MODE%"=="-h" goto :help
if /I "%MODE%"=="--help" goto :help
if /I "%MODE%"=="help" goto :help
if not "%MODE%"=="" goto :unknown

echo [INFO] Dry run — validating manifest only. Use "apply" to sync to GitHub.
uv run python scripts/sync_github_repository_metadata.py
exit /b %ERRORLEVEL%

:apply
where gh >nul 2>&1
if errorlevel 1 (
    echo [ERROR] gh CLI is required for apply. Install: https://cli.github.com/
    echo Then authenticate: gh auth login
    exit /b 1
)
echo [INFO] Applying manifest to GitHub repository settings...
uv run python scripts/sync_github_repository_metadata.py --apply
exit /b %ERRORLEVEL%

:help
echo.
echo Usage: sync-github-metadata.bat [apply^|help]
echo.
echo   sync-github-metadata.bat        Validate .github/repository-metadata.json
echo   sync-github-metadata.bat apply  Push description, homepage, and topics to GitHub
echo.
echo Requires: uv. For apply also: gh auth login
exit /b 0

:unknown
echo [ERROR] Unknown option: %MODE%
goto :help
