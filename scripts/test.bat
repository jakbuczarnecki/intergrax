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

REM --- Run tests using venv interpreter ---
echo [INFO] Running test suite
.\.venv\Scripts\python -m pytest %*

exit /b %ERRORLEVEL%
