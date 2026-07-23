#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
PROOF="$SCRIPT_DIR/run-lkw-managed-workspace-live-proof.py"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv was not found on PATH."
  exit 1
fi

if [ ! -f "$PROOF" ]; then
  echo "Missing managed workspace live proof helper: $PROOF"
  exit 1
fi

cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1
export LKW_MONGODB_HOST_PORT="${LKW_MONGODB_HOST_PORT:-27018}"
export LKW_MONGODB_ROOT_USERNAME="${LKW_MONGODB_ROOT_USERNAME:-intergrax}"
export LKW_MONGODB_ROOT_PASSWORD="${LKW_MONGODB_ROOT_PASSWORD:-intergrax-local-dev-only}"
export LKW_MONGODB_DATABASE="${LKW_MONGODB_DATABASE:-intergrax_proofs}"
export LKW_MONGODB_COLLECTION="${LKW_MONGODB_COLLECTION:-proof_receipts}"
export LKW_MANAGED_WORKSPACE_COLLECTION="${LKW_MANAGED_WORKSPACE_COLLECTION:-lkw_managed_workspaces}"

uv run --project applications/local_workspace_application python "$PROOF" "$@"
