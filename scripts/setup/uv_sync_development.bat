@echo off
setlocal

REM © Artur Czarnecki. All rights reserved.
REM Intergrax framework – proprietary and confidential.
REM Use, modification, or distribution without written permission is prohibited.

REM --- Ensure we run from project root ---
cd /d %~dp0\..\..

REM Compatibility wrapper: setup.bat is the canonical ENV-1 bootstrap.
call "%~dp0setup.bat"
exit /b %ERRORLEVEL%
