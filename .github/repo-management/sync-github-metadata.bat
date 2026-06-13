@echo off
setlocal

REM © Artur Czarnecki. All rights reserved.
REM Sync GitHub repository metadata — see README.md in this folder.

cd /d "%~dp0\..\.."

where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv is not installed or not in PATH.
    echo Install it from: https://docs.astral.sh/uv/
    exit /b 1
)

set MODE=%1
set SCRIPT=.github\repo-management\sync_github_repository_metadata.py

if /I "%MODE%"=="check" goto :check
if /I "%MODE%"=="dry-run" goto :check
if /I "%MODE%"=="validate" goto :check
if /I "%MODE%"=="-h" goto :help
if /I "%MODE%"=="--help" goto :help
if /I "%MODE%"=="help" goto :help
if not "%MODE%"=="" goto :unknown

goto :apply

:check
echo [INFO] Dry run - validating manifest only (no GitHub changes).
uv run python %SCRIPT%
exit /b %ERRORLEVEL%

:apply
echo [INFO] Applying manifest to GitHub repository settings...
uv run python %SCRIPT% --apply
exit /b %ERRORLEVEL%

:help
echo.
echo Usage: .github\repo-management\sync-github-metadata.bat [check^|help]
echo.
echo   sync-github-metadata.bat         Push description, homepage, and topics to GitHub
echo   sync-github-metadata.bat check   Validate manifest only (dry run)
echo.
echo Setup: .github\repo-management\README.md
exit /b 0

:unknown
echo [ERROR] Unknown option: %MODE%
goto :help
