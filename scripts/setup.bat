@echo off
setlocal enabledelayedexpansion

REM © Artur Czarnecki. All rights reserved.
REM Intergrax framework – proprietary and confidential.
REM Use, modification, or distribution without written permission is prohibited.

echo ========================================
echo Intergrax - environment setup (Windows)
echo ========================================

REM --- 0. Guard: conda must not be active ---
if defined CONDA_PREFIX (
    echo [ERROR] Conda environment detected.
    echo Please deactivate conda before running this script.
    echo   conda deactivate
    exit /b 1
)

REM --- 1. Ensure uv is available ---
where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv is not installed or not in PATH.
    echo Install it from: https://docs.astral.sh/uv/
    exit /b 1
)

REM --- 2. Remove existing venv ---
if exist .venv (
    echo [INFO] Removing existing .venv
    rmdir /s /q .venv
)

REM --- 3. Create venv ---
echo [INFO] Creating virtual environment (.venv)
uv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    exit /b 1
)

REM --- 4. Install ALL dependencies (runtime + dev) ---
echo [INFO] Installing dependencies (including dev extras)
uv sync --extra dev
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    exit /b 1
)

@REM REM --- 5. Run tests using venv interpreter ---
@REM echo [INFO] Running test suite
@REM .\.venv\Scripts\python -m pytest
@REM if errorlevel 1 (
@REM     echo [ERROR] Tests failed.
@REM     exit /b 1
@REM )

echo.
echo ========================================
echo Setup completed successfully.
echo To start working:
echo   .\.venv\Scripts\activate
echo ========================================

exit /b 0
