#!/usr/bin/env sh
# © Artur Czarnecki. All rights reserved.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
PROOF_SCRIPT="$SCRIPT_DIR/run-sentry-observability-proof.py"

RUN_ID=""
CORRELATION_ID=""
while [ $# -gt 0 ]; do
  case "$1" in
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --correlation-id)
      CORRELATION_ID="${2:-}"
      shift 2
      ;;
    *)
      if [ -z "$RUN_ID" ]; then
        RUN_ID="$1"
      fi
      shift
      ;;
  esac
done

if [ ! -f "$PROOF_SCRIPT" ]; then
  echo "Missing proof script: $PROOF_SCRIPT" >&2
  exit 1
fi

cd "$REPO_ROOT"

echo "LKW Sentry observability proof helper"
echo "Repository root: $(pwd)"
echo "Target: LOCAL_WORKSPACE_BACKEND_BASE_URL=${LOCAL_WORKSPACE_BACKEND_BASE_URL:-http://127.0.0.1:8020}"
echo

ARGS=""
if [ -n "$RUN_ID" ]; then
  ARGS="$ARGS --run-id $RUN_ID"
fi
if [ -n "$CORRELATION_ID" ]; then
  ARGS="$ARGS --correlation-id $CORRELATION_ID"
fi

# shellcheck disable=SC2086
uv run python "$PROOF_SCRIPT" $ARGS
STATUS=$?

echo
if [ "$STATUS" -eq 0 ]; then
  echo "Local Sentry UI: ${LKW_SENTRY_PROOF_UI_URL:-http://127.0.0.1:9000}"
  echo "Sentry search filters:"
  echo "  tag:intergrax.problem_kind=lkw.proof_controlled_failure"
  echo "  tag:intergrax.problem_error_code=LKW_PROOF_CONTROLLED_FAILURE"
  if [ -n "$RUN_ID" ]; then
    echo "  tag:intergrax.run_id=$RUN_ID"
  fi
  if [ -n "$CORRELATION_ID" ]; then
    echo "  tag:intergrax.correlation_id=$CORRELATION_ID"
  fi
fi

exit "$STATUS"
