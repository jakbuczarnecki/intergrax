#!/usr/bin/env bash
# INFORMATIONAL ONLY — not a fail-closed admission gate (profile v1 TARGET-VERIFIED helper).
# For CONFIGURED admission use: python -m governed_contractor_application.host.single_node_profile_v1_admission_gate
# Read-only on operator Linux VM. Does not mutate storage.

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
