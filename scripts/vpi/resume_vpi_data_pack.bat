@echo off
REM Deprecated compatibility launcher.
REM Use scenario-local operator script:
REM   platform_proofs/scenarios/verified_product_identification/scripts/operator/resume_vpi_data_pack.bat
setlocal EnableExtensions
set "SCRIPT_DIR=%~dp0"
call "%SCRIPT_DIR%..\..\platform_proofs\scenarios\verified_product_identification\scripts\operator\resume_vpi_data_pack.bat"
exit /b %ERRORLEVEL%
