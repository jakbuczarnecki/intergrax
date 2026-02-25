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

REM --- Default: run all tests ---
if "%MODE%"=="" (
    echo [INFO] Running ALL tests
    .\.venv\Scripts\python -m pytest
    exit /b %ERRORLEVEL%
)

REM --- Run by marker ---
echo [INFO] Running tests with marker: %MODE%
.\.venv\Scripts\python -m pytest -m "%MODE%"

exit /b %ERRORLEVEL%