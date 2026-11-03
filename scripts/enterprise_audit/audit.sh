#!/usr/bin/env bash
# © Artur Czarnecki. All rights reserved.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODE=""
SINCE=""
if [[ $# -lt 1 ]]; then
  echo "Usage: audit.sh fast|full|freeze|changed <since-sha>" >&2
  exit 1
fi
case "${1}" in
  fast) MODE="FAST" ;;
  full) MODE="FULL" ;;
  freeze) MODE="FREEZE" ;;
  changed)
    MODE="CHANGED-SINCE"
    SINCE="${2:-}"
    ;;
  *)
    echo "unknown mode: ${1}" >&2
    exit 1
    ;;
esac
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found on PATH" >&2
  exit 3
fi
SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
RUN_ID="$(python -c "import uuid; print(uuid.uuid4().hex)" 2>/dev/null || echo "${RANDOM}-${RANDOM}")"
OUT="${REPO_ROOT}/.tmp/proof/ent-audit-x/local-${SHA}-${RUN_ID}"
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
