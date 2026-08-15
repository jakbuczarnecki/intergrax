#!/usr/bin/env sh
# © Artur Czarnecki. All rights reserved.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
RUNNER="$SCRIPT_DIR/run-lkw-product-quickstart.py"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv was not found on PATH."
    exit 1
fi

if [ ! -f "$RUNNER" ]; then
    echo "Missing shared quickstart runner: $RUNNER"
    exit 1
fi

cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1
exec uv run --project applications/local_workspace_application python "$RUNNER" --os-family linux --wrapper-id linux_sh "$@"
