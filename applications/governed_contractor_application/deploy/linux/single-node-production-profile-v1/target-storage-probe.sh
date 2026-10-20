#!/usr/bin/env bash
# Read-only TARGET-VERIFIED helper for operator Linux VM (profile v1).
# Does not mutate storage. Run as least-privilege user after paths are configured.

set -euo pipefail

DATA_DIR="${INTERGRAX_SINGLE_NODE_PRODUCTION_V1_DATA_DIR:-}"
if [[ -z "${DATA_DIR}" ]]; then
  echo "INTERGRAX_SINGLE_NODE_PRODUCTION_V1_DATA_DIR is required" >&2
  exit 1
fi

echo "== mount =="
findmnt -T "${DATA_DIR}" || true
echo "== fstype =="
stat -f -c '%T' "${DATA_DIR}" 2>/dev/null || df -T "${DATA_DIR}"
echo "== permissions =="
namei -l "${DATA_DIR}" 2>/dev/null || ls -ld "${DATA_DIR}"
echo "== sqlite pragmas (if files exist) =="
for db in \
  "${DATA_DIR}/gca_host.sqlite" \
  "${DATA_DIR}/provider_invocation.sqlite" \
  "${DATA_DIR}/agent_distribution.sqlite"; do
  if [[ -f "${db}" ]]; then
    echo "-- ${db}"
    sqlite3 "${db}" "PRAGMA journal_mode; PRAGMA synchronous;"
  fi
done
