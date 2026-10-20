#!/usr/bin/env bash
# LOCAL QUALIFICATION ONLY — not TARGET-VERIFIED production or operator acceptance.
# CONFIGURED admission (fail-closed), then operator INTERGRAX_GCA_STRICT_SERVE_EXEC (Package B + D4).
# Does not substitute governance, provider pins, or authorized mutation bundles.
set -euo pipefail

qualification_mode="${INTERGRAX_GCA_DOCKER_LOCAL_QUALIFICATION_MODE:-}"

_run_admission_gate() {
  python -m governed_contractor_application.host.single_node_profile_v1_admission_gate
}

_ensure_durable_layout() {
  local data_dir="${INTERGRAX_SINGLE_NODE_PRODUCTION_V1_DATA_DIR:-}"
  if [[ -z "${data_dir}" ]]; then
    echo "INTERGRAX_SINGLE_NODE_PRODUCTION_V1_DATA_DIR is required" >&2
    exit 2
  fi
  mkdir -p "${data_dir}" "${data_dir}/trusted_artifacts"
}

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: entrypoint.sh [--preflight-only | --configured-verify | --storage-probe | serve]

  --preflight-only / --configured-verify  CONFIGURED admission gate only (no writer lock).
  --storage-probe                         informational mount/SQLite probe (not admission).
  serve (default)                         admission gate, then INTERGRAX_GCA_STRICT_SERVE_EXEC.

BLOCKED — PACKAGE B OPERATOR SERVE CONFIGURATION REQUIRED when serve mode lacks
INTERGRAX_GCA_STRICT_SERVE_EXEC naming the sanctioned D4 strict session opener.
EOF
  exit 0
fi

_ensure_durable_layout

mode="${1:-serve}"
if [[ "${mode}" == "--preflight-only" || "${mode}" == "--configured-verify" || "${qualification_mode}" == "configured_verify" ]]; then
  _run_admission_gate
  exit 0
fi

if [[ "${mode}" == "--storage-probe" ]]; then
  exec /opt/intergrax/bin/target-storage-probe.sh
fi

_run_admission_gate

serve_exec="${INTERGRAX_GCA_STRICT_SERVE_EXEC:-}"
if [[ -z "${serve_exec}" ]]; then
  echo "BLOCKED — PACKAGE B OPERATOR SERVE CONFIGURATION REQUIRED" >&2
  echo "INTERGRAX_GCA_STRICT_SERVE_EXEC must name the operator serve command that opens" >&2
  echo "open_gca_strict_production_serving_session_from_single_node_production_profile_v1" >&2
  exit 2
fi

exec bash -lc "${serve_exec}"
