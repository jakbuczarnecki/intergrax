#!/usr/bin/env bash
# Sanctioned operator entrypoint template (install to /opt/intergrax/bin/).
# CONFIGURED admission gate (fail-closed) then operator-supplied serve exec that must call
# open_gca_strict_production_serving_session_from_single_node_production_profile_v1 (D4).
set -euo pipefail

if [[ "${1:-}" == "--preflight-only" ]]; then
  exec uv run python -m governed_contractor_application.host.single_node_profile_v1_admission_gate
fi

if [[ -z "${INTERGRAX_SINGLE_NODE_PRODUCTION_V1_DATA_DIR:-}" ]]; then
  echo "INTERGRAX_SINGLE_NODE_PRODUCTION_V1_DATA_DIR is required" >&2
  exit 2
fi

uv run python -m governed_contractor_application.host.single_node_profile_v1_admission_gate

serve_exec="${INTERGRAX_GCA_STRICT_SERVE_EXEC:-}"
if [[ -z "${serve_exec}" ]]; then
  echo "INTERGRAX_GCA_STRICT_SERVE_EXEC must name the operator serve command that opens" >&2
  echo "open_gca_strict_production_serving_session_from_single_node_production_profile_v1" >&2
  exit 2
fi

exec bash -lc "${serve_exec}"
