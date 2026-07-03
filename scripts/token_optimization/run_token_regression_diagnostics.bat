@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\.."
set "ARTIFACT_DIR=.artifacts\token_optimization\regression_synthetic_v1"
set "DATASET_DIR=benchmarks/token_optimization/fixtures/regression_synthetic_v1"

pushd "%REPO_ROOT%" || exit /b 1

echo Running token regression diagnostic benchmark...
if exist "%ARTIFACT_DIR%" rmdir /s /q "%ARTIFACT_DIR%"

uv run python scripts/check_token_regression_benchmarks.py --report --fixture-dataset "%DATASET_DIR%" --diagnostic-artifact-dir "%ARTIFACT_DIR%"
set "BENCHMARK_EXIT=%ERRORLEVEL%"

echo.
echo Reviewing diagnostic artifacts...
set "REVIEW_EXIT=1"
if exist "%ARTIFACT_DIR%" (
  uv run python scripts/review_token_regression_artifacts.py "%ARTIFACT_DIR%"
  set "REVIEW_EXIT=%ERRORLEVEL%"
) else (
  echo Diagnostic artifact directory was not created: %ARTIFACT_DIR%
)

echo.
if "%BENCHMARK_EXIT%"=="0" if "%REVIEW_EXIT%"=="0" (
  echo Done.
  popd
  exit /b 0
)

echo Done with failures.
echo benchmark_exit=%BENCHMARK_EXIT%
echo review_exit=%REVIEW_EXIT%
popd
exit /b 1
