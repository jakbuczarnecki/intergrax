#!/usr/bin/env bash
# © Artur Czarnecki. All rights reserved.
# Execute one real LKW run and validate its Elasticsearch/Kibana observability output.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VALIDATOR="${SCRIPT_DIR}/run-elasticsearch-observability-proof.sh"
RUN_ID_FILE="$(mktemp -t intergrax-lkw-es-run-id.XXXXXX)"

LKW_BASE_URL="${LOCAL_WORKSPACE_BACKEND_BASE_URL:-http://127.0.0.1:8020}"
KIBANA_URL="${LOCAL_WORKSPACE_OBSERVABILITY_PROOF_KIBANA_URL:-http://127.0.0.1:5601}"

cleanup() {
  rm -f "${RUN_ID_FILE}"
}
trap cleanup EXIT

if [[ ! -f "${VALIDATOR}" ]]; then
  echo "Missing Elasticsearch validator: ${VALIDATOR}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

echo "LKW Elasticsearch/Kibana one-command proof helper"
echo "Repository root: ${PWD}"
echo "LKW base URL: ${LKW_BASE_URL}"
echo "Kibana URL: ${KIBANA_URL}"
echo
echo "Step 1/3: executing a real LKW run..."

response="$(curl -fsS -X POST "${LKW_BASE_URL%/}/v1/local_workspace/run" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Find documents about local workspace observability proof",
    "capability": "local.workspace.search",
    "metadata": {
      "proof": "LKW_PLATFORM_PROOF",
      "proof_helper": "run-lkw-elasticsearch-proof.sh"
    }
  }')"

run_id="$(python -c 'import json,sys; payload=json.load(sys.stdin); run_id=payload.get("run_id");
if not run_id: raise SystemExit("LKW response did not include run_id");
print(run_id)' <<<"${response}")"

state="$(python -c 'import json,sys; payload=json.load(sys.stdin); print(payload.get("state", ""))' <<<"${response}")"
agent_id="$(python -c 'import json,sys; payload=json.load(sys.stdin); print(payload.get("agent_id", ""))' <<<"${response}")"

echo "run_id=${run_id}"
if [[ -n "${state}" ]]; then echo "state=${state}"; fi
if [[ -n "${agent_id}" ]]; then echo "agent_id=${agent_id}"; fi

echo
echo "Step 2/3: validating Elasticsearch observability for run_id=${run_id}..."
bash "${VALIDATOR}" "${run_id}"

echo
echo "Step 3/3: open Kibana and inspect this run."
echo "Kibana URL:"
echo "  ${KIBANA_URL}"
echo
echo "Kibana Discover filter:"
echo "  intergrax.run_id: \"${run_id}\""
echo
echo "Proof result: PASS"
echo "run_id=${run_id}"
echo "kibana_url=${KIBANA_URL}"
echo "elasticsearch_validation=passed"
