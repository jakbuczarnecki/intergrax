@echo off
setlocal

REM © Artur Czarnecki. All rights reserved.
REM Intergrax framework – proprietary and confidential.
REM Use, modification, or distribution without written permission is prohibited.

REM --- Guard: venv must exist ---
if not exist .venv\Scripts\python.exe (
    echo [ERROR] Virtual environment not found.
    echo Run: scripts\setup.bat
    exit /b 1
)

set MODE=%1

echo [INFO] Test mode: %MODE%

REM --- Default = unit only ---
if "%MODE%"=="" (
    echo [INFO] Running UNIT tests only
    .\.venv\Scripts\python -m pytest -m "not integration and not e2e"
    exit /b %ERRORLEVEL%
)

REM --- Integration only ---
if "%MODE%"=="integration" (
    echo [INFO] Running INTEGRATION tests
    .\.venv\Scripts\python -m pytest -m "integration"
    exit /b %ERRORLEVEL%
)

REM --- E2E only ---
if "%MODE%"=="e2e" (
    echo [INFO] Running E2E tests
    .\.venv\Scripts\python -m pytest -m "e2e"
    exit /b %ERRORLEVEL%
)

REM --- All tests ---
if "%MODE%"=="all" (
    echo [INFO] Running ALL tests
    .\.venv\Scripts\python -m pytest
    exit /b %ERRORLEVEL%
)

REM --- Unknown mode ---
echo [ERROR] Unknown test mode: %MODE%
echo Allowed: integration, e2e, all
exit /b 1