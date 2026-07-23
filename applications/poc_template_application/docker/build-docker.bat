@echo off
REM Build Tier-3 application image via materialized runtime graph.
setlocal EnableExtensions

set "PKG=poc_template_application"
set "IMAGE_TAG=poc_template-application"
if not "%~1"=="" set "IMAGE_TAG=%~1"
set "PORT=8095"

cd /d "%~dp0\..\..\.."
if errorlevel 1 (
  echo Failed to locate repository root.
  exit /b 1
)

uv run python scripts/build/build_application_image.py --application %PKG% --tag %IMAGE_TAG% --context-dir applications/%PKG%/docker/runtime-context --keep-context
if errorlevel 1 exit /b 1

echo.
echo Built: %IMAGE_TAG%
echo Run:   docker run --rm --env-file applications/%PKG%/.env -p %PORT%:%PORT% %IMAGE_TAG%
endlocal
