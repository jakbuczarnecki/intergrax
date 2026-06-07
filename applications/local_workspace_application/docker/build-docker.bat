@echo off
REM Build Tier-3 application image from monorepo root (Phase N).
setlocal EnableExtensions

set "PKG=local_workspace_application"
set "IMAGE_TAG=local_workspace-application"
if not "%~1"=="" set "IMAGE_TAG=%~1"
set "PORT=8020"

cd /d "%~dp0\..\..\.."
if errorlevel 1 (
  echo Failed to locate repository root.
  exit /b 1
)

docker buildx version >nul 2>&1
if %ERRORLEVEL% equ 0 (
  echo Building %IMAGE_TAG% ^(BuildKit^)...
  docker buildx build -f applications/%PKG%/docker/Dockerfile --ignorefile applications/%PKG%/docker/.dockerignore -t %IMAGE_TAG% .
) else (
  echo BuildKit not found — using docker build
  docker build -f applications/%PKG%/docker/Dockerfile -t %IMAGE_TAG% .
)
if errorlevel 1 exit /b 1

echo.
echo Built: %IMAGE_TAG%
echo Run:   docker run --rm --env-file applications/%PKG%/.env -p %PORT%:%PORT% %IMAGE_TAG%
endlocal
