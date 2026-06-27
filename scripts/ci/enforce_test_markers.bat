@echo off
setlocal

REM © Artur Czarnecki. All rights reserved.
REM Intergrax framework – proprietary and confidential.
REM Use, modification, or distribution without written permission is prohibited.

REM --- Guard: venv must exist ---
if not exist .venv\Scripts\python.exe (
    echo [ERROR] Virtual environment not found.
    echo Run: scripts\setup\setup.bat
    exit /b 1
)

echo [INFO] Enforcing pytest markers in tests directory...

.\.venv\Scripts\python tools\enforce_test_markers.py

exit /b %ERRORLEVEL%