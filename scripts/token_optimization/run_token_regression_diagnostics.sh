#!/usr/bin/env sh
set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)

ARTIFACT_DIR=".artifacts/token_optimization/regression_synthetic_v1"
DATASET_DIR="benchmarks/token_optimization/fixtures/regression_synthetic_v1"

cd "$REPO_ROOT" || exit 1

echo "Running token regression diagnostic benchmark..."
rm -rf "$ARTIFACT_DIR"

uv run python scripts/check_token_regression_benchmarks.py --report --fixture-dataset "$DATASET_DIR" --diagnostic-artifact-dir "$ARTIFACT_DIR"
BENCHMARK_EXIT=$?

echo
echo "Reviewing diagnostic artifacts..."
REVIEW_EXIT=1
if [ -d "$ARTIFACT_DIR" ]; then
  uv run python scripts/review_token_regression_artifacts.py "$ARTIFACT_DIR"
  REVIEW_EXIT=$?
else
  echo "Diagnostic artifact directory was not created: $ARTIFACT_DIR"
fi

echo
if [ "$BENCHMARK_EXIT" -eq 0 ] && [ "$REVIEW_EXIT" -eq 0 ]; then
  echo "Done."
  exit 0
fi

echo "Done with failures."
echo "benchmark_exit=$BENCHMARK_EXIT"
echo "review_exit=$REVIEW_EXIT"
exit 1
