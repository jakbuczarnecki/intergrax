@echo off
setlocal

REM © Artur Czarnecki. All rights reserved.
REM Intergrax framework – proprietary and confidential.
REM Use, modification, or distribution without written permission is prohibited.

REM --- Ensure we run from project root ---
cd /d %~dp0\..

REM --- Guard: uv must be available ---
where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] uv is not installed or not in PATH.
    exit /b 1
)

REM --- Create venv if missing ---
if not exist .venv\Scripts\python.exe (
    echo [INFO] Virtual environment not found. Creating...
    uv venv
    if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
)

REM --- Sync dev + distributed extras ---
echo [INFO] Syncing dev + distributed dependencies...
uv sync --extra dev --extra distributed

exit /b %ERRORLEVEL%