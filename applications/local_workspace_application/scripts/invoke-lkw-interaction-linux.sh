#!/usr/bin/env sh
# © Artur Czarnecki. All rights reserved.
# Thin Linux launcher for the shared LKW interaction client.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CLIENT="$SCRIPT_DIR/invoke-lkw-interaction.py"

if [ ! -f "$CLIENT" ]; then
  echo "Missing shared interaction client: $CLIENT" >&2
  exit 1
fi

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "Python interpreter was not found on PATH." >&2
  exit 1
fi

exec "$PYTHON" "$CLIENT" \
  --os-family linux \
  --adapter-id lkw.linux_shell \
  --source linux_shell \
  --wrapper-runtime posix_sh \
  "$@"
