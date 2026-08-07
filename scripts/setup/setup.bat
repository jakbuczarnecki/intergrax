@echo off
setlocal enabledelayedexpansion

REM © Artur Czarnecki. All rights reserved.
REM Intergrax framework – proprietary and confidential.
REM Use, modification, or distribution without written permission is prohibited.

echo ========================================
echo Intergrax - environment setup (Windows)
echo ========================================

REM Canonical runtime: uv-managed CPython 3.12. Do not recreate .venv with
REM an arbitrary Python installation. Recreate .venv through this script.
REM Global Python and Anaconda are not part of the project runtime.

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

REM --- 2. Ensure the repository-required uv-managed Python exists ---
set "UV_MANAGED_PYTHON=1"
echo [INFO] Ensuring uv-managed CPython 3.12 is installed
uv python install 3.12 --managed-python
if errorlevel 1 (
    echo [ERROR] Failed to install or locate uv-managed CPython 3.12.
    exit /b 1
)

REM Resolve the managed interpreter through uv's machine-readable JSON API;
REM do not hardcode its user path or let an existing .venv win discovery.
set "UV_MANAGED_PYTHON_PATH="
for /f "delims=" %%P in ('powershell -NoProfile -Command "$p = uv python list 3.12 --managed-python --only-installed --output-format json | ConvertFrom-Json; $p | Where-Object implementation -eq cpython | Select-Object -First 1 -ExpandProperty path"') do if not defined UV_MANAGED_PYTHON_PATH set "UV_MANAGED_PYTHON_PATH=%%P"
if not defined UV_MANAGED_PYTHON_PATH (
    echo [ERROR] uv did not return a managed CPython 3.12 interpreter.
    exit /b 1
)

set "UV_MANAGED_PYTHON_ROOT="
for /f "delims=" %%D in ('uv python dir') do if not defined UV_MANAGED_PYTHON_ROOT set "UV_MANAGED_PYTHON_ROOT=%%D"
if not defined UV_MANAGED_PYTHON_ROOT (
    echo [ERROR] uv did not return its managed Python directory.
    exit /b 1
)

REM --- 3. Create venv from uv-managed Python only ---
echo [INFO] Creating virtual environment (.venv) from uv-managed CPython 3.12
uv venv --clear --python "%UV_MANAGED_PYTHON_PATH%" --managed-python .venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    exit /b 1
)

REM --- 4. Validate interpreter provenance before installing dependencies ---
echo [INFO] Validating .venv interpreter provenance
.\.venv\Scripts\python.exe -c "import os,pathlib,sys,sysconfig; v=pathlib.Path('.venv').resolve(); e=pathlib.Path(sys.executable).resolve(); b=pathlib.Path(getattr(sys,'_base_executable','')).resolve(); expected=pathlib.Path(os.environ['UV_MANAGED_PYTHON_PATH']).resolve(); managed=pathlib.Path(os.environ['UV_MANAGED_PYTHON_ROOT']).resolve(); home=pathlib.Path(next(x.split('=',1)[1].strip() for x in (v/'pyvenv.cfg').read_text().splitlines() if x.lower().startswith('home ='))).resolve(); assert sys.version_info[:2] == (3,12), 'Python must be 3.12'; assert e == (v/'Scripts'/'python.exe').resolve(), 'sys.executable is outside .venv'; assert pathlib.Path(sys.prefix).resolve() == v, 'sys.prefix is outside .venv'; assert pathlib.Path(sys.base_prefix).resolve() == home, 'sys.base_prefix does not match pyvenv.cfg'; assert b == expected, 'base executable differs from uv-managed interpreter'; assert expected.is_relative_to(managed), 'base executable is outside uv managed Python directory'; assert 'PYTHONPATH' not in os.environ or not os.environ['PYTHONPATH'], 'PYTHONPATH must be empty'; assert not any(x in str(b).casefold() for x in ('conda','anaconda')), 'Conda/Anaconda interpreter detected'; print('  Python='+sys.version.split()[0]); print('  sys.executable='+str(e)); print('  sys.prefix='+str(pathlib.Path(sys.prefix).resolve())); print('  sys.base_prefix='+str(pathlib.Path(sys.base_prefix).resolve())); print('  uv-managed base='+str(b)); print('  site-packages='+sysconfig.get_paths()['purelib'])"
if errorlevel 1 (
    echo [ERROR] .venv interpreter provenance validation failed.
    exit /b 1
)

REM --- 5. Install ALL dependencies (runtime + dev) from uv.lock only ---
echo [INFO] Installing dependencies (including dev extras) from uv.lock
uv sync --extra dev --frozen --managed-python
if errorlevel 1 (
    echo [ERROR] Frozen dependency synchronization failed.
    exit /b 1
)

@REM REM --- 6. Run tests using venv interpreter ---
@REM echo [INFO] Running test suite
@REM .\.venv\Scripts\python -m pytest
@REM if errorlevel 1 (
@REM     echo [ERROR] Tests failed.
@REM     exit /b 1
@REM )

echo.
echo ========================================
echo Setup completed successfully.
echo Activation is optional. Use:
echo   uv run ...
echo   .\.venv\Scripts\python -m ...
echo ========================================

exit /b 0
