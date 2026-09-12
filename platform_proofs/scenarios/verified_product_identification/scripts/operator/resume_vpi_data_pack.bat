@echo off
setlocal EnableExtensions
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\..\..\..\.."
for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"
set "PYTHONPATH=%REPO_ROOT%"
set "CUDA_PYTHON=%REPO_ROOT%\.tmp\session\vpi-5c4a2\cuda-venv\Scripts\python.exe"
cd /d "%REPO_ROOT%"
"%CUDA_PYTHON%" -m platform_proofs.scenarios.verified_product_identification.scripts.operator.run_vpi_data_pack_resume
exit /b %ERRORLEVEL%
