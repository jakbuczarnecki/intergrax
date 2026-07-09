#!/usr/bin/env bash
# © Artur Czarnecki. All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROOF="${SCRIPT_DIR}/run-lkw-background-task-proof.py"

if [[ ! -f "${PROOF}" ]]; then
  echo "Missing proof helper: ${PROOF}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
echo "LKW Kafka background-task platform proof helper"
echo "Repository root: ${REPO_ROOT}"
echo

uv run python "${PROOF}" "$@"
