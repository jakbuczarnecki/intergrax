#!/usr/bin/env sh
# © Artur Czarnecki. All rights reserved.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
ORCH="$SCRIPT_DIR/run-lkw-linux-container-certification.py"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "Python was not found on PATH." >&2
  exit 1
fi

if [ ! -f "$ORCH" ]; then
  echo "Missing Linux container certification orchestrator: $ORCH" >&2
  exit 1
fi

cd "$REPO_ROOT"
PYTHONUNBUFFERED=1
export PYTHONUNBUFFERED
exec "$PYTHON" "$ORCH" --pre-commit-certification "$@"
