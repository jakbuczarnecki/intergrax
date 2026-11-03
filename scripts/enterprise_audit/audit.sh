#!/usr/bin/env bash
# © Artur Czarnecki. All rights reserved.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODE="FAST"
SINCE=""
case "${1:-}" in
  fast|"") MODE="FAST" ;;
  full) MODE="FULL" ;;
  freeze) MODE="FREEZE" ;;
  changed)
    MODE="CHANGED-SINCE"
    SINCE="${2:-}"
    ;;
esac
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found on PATH" >&2
  exit 3
fi
SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
OUT="${REPO_ROOT}/.tmp/proof/ent-audit-x/local-${SHA}-${RANDOM}"
if [[ "${MODE}" == "CHANGED-SINCE" ]]; then
  if [[ -z "${SINCE}" ]]; then
    echo "CHANGED-SINCE requires since SHA" >&2
    exit 1
  fi
  uv run python "${REPO_ROOT}/scripts/enterprise_audit/run_enterprise_audit.py" \
    --mode CHANGED-SINCE --since-sha "${SINCE}" --output-dir "${OUT}"
else
  uv run python "${REPO_ROOT}/scripts/enterprise_audit/run_enterprise_audit.py" \
    --mode "${MODE}" --output-dir "${OUT}"
fi
