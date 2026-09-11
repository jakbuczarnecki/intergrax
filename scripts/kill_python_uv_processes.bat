@echo off
setlocal EnableExtensions

REM © Artur Czarnecki. All rights reserved.
REM Intergrax framework – proprietary and confidential.
REM Use, modification, or distribution without written permission is prohibited.
REM
REM Force-terminates ALL python.exe and uv.exe processes on this machine.
REM May break Cursor, other IDEs, Jupyter, system services using Python, etc.

if /I "%~1"=="/Y" goto :kill
if /I "%~1"=="--yes" goto :kill

echo.
echo [WARN] This will force-kill EVERY python.exe and uv.exe on this computer.
echo        Close important work first (IDEs, notebooks, long-running jobs).
echo.
set /p CONFIRM=Type YES to continue: 
if /I not "%CONFIRM%"=="YES" (
    echo Aborted.
    exit /b 1
)

:kill
echo [INFO] Terminating uv.exe ...
taskkill /F /IM uv.exe /T >nul 2>&1
if errorlevel 1 echo [INFO] No uv.exe processes found or access denied.

echo [INFO] Terminating python.exe ...
taskkill /F /IM python.exe /T >nul 2>&1
if errorlevel 1 echo [INFO] No python.exe processes found or access denied.

echo [INFO] Done.
exit /b 0
