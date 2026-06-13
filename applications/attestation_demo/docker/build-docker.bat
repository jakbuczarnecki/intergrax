@echo off
REM Build Tier-3 attestation_demo image from monorepo root (Windows).
setlocal
set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%..\..\.."
set PKG=attestation_demo
set IMAGE_TAG=attestation-demo
set PORT=8097

docker build -f applications\%PKG%\docker\Dockerfile -t %IMAGE_TAG% .
if errorlevel 1 exit /b 1

echo.
echo Built: %IMAGE_TAG%
echo Run:   docker run --rm --env-file applications\%PKG%\.env -p %PORT%:%PORT% %IMAGE_TAG%
popd
