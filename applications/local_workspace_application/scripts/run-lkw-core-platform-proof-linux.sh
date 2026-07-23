#!/usr/bin/env sh
# © Artur Czarnecki. All rights reserved.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
PROOF="$SCRIPT_DIR/run-lkw-core-platform-proof.py"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv was not found on PATH." >&2
  exit 1
fi

if [ ! -f "$PROOF" ]; then
  echo "Missing shared core proof runner: $PROOF" >&2
  exit 1
fi

cd "$REPO_ROOT"
PYTHONUNBUFFERED=1
export PYTHONUNBUFFERED
exec uv run --project applications/local_workspace_application python "$PROOF" \
  --os-family linux \
  --wrapper-id linux_sh \
  "$@"
