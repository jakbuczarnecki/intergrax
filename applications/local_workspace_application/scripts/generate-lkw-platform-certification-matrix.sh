#!/usr/bin/env sh
# © Artur Czarnecki. All rights reserved.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
GEN="$SCRIPT_DIR/generate-lkw-platform-certification-matrix.py"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "Python was not found on PATH." >&2
  exit 1
fi

if [ ! -f "$GEN" ]; then
  echo "Missing matrix generator: $GEN" >&2
  exit 1
fi

cd "$REPO_ROOT"
PYTHONUNBUFFERED=1
export PYTHONUNBUFFERED
exec "$PYTHON" "$GEN" "$@"
