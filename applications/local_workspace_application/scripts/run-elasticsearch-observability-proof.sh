#!/usr/bin/env sh
# © Artur Czarnecki. All rights reserved.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
INSPECTOR="$SCRIPT_DIR/inspect_elasticsearch_observability.py"

LKW_HEALTH_URL=${LOCAL_WORKSPACE_OBSERVABILITY_PROOF_LKW_HEALTH_URL:-http://127.0.0.1:8020/health}
ES_URL=${LOCAL_WORKSPACE_OBSERVABILITY_PROOF_ES_URL:-http://127.0.0.1:9200}
ES_INDEX=${LOCAL_WORKSPACE_OBSERVABILITY_PROOF_ES_INDEX:-intergrax-lkw-observability}

RUN_ID=""
if [ "${1:-}" = "--run-id" ]; then
  RUN_ID="${2:-}"
elif [ "${1:-}" != "" ]; then
  RUN_ID="$1"
fi

if [ ! -f "$INSPECTOR" ]; then
  echo "Missing inspector script: $INSPECTOR" >&2
  exit 1
fi

cd "$REPO_ROOT"

echo "LKW Elasticsearch observability proof helper"
echo "Repository root: $(pwd)"
echo "LKW health URL: $LKW_HEALTH_URL"
echo "Elasticsearch URL: $ES_URL"
echo "Elasticsearch index: $ES_INDEX"
echo

echo "Checking LKW health..."
curl -fsS "$LKW_HEALTH_URL"
echo
echo

echo "Checking Elasticsearch health..."
curl -fsS "$ES_URL/_cluster/health"
echo
echo

if [ -z "$RUN_ID" ]; then
  echo "Listing recent Elasticsearch observability runs..."
  uv run python "$INSPECTOR" --url "$ES_URL" --index "$ES_INDEX" --list-runs
  echo
  echo "Next steps:"
  echo "  1. Execute a real LKW run via Swagger or curl."
  echo "  2. Copy the resulting run_id, or use the latest run_id listed above."
  echo "  3. Run:"
  echo "     applications/local_workspace_application/scripts/run-elasticsearch-observability-proof.sh <run_id>"
  exit 0
fi

echo "Inspecting run_id: $RUN_ID"
uv run python "$INSPECTOR" --url "$ES_URL" --index "$ES_INDEX" --run-id "$RUN_ID"
echo

echo "Running duplicate check..."
uv run python "$INSPECTOR" --url "$ES_URL" --index "$ES_INDEX" --run-id "$RUN_ID" --check-duplicates
echo

echo "Running safety-key check..."
uv run python "$INSPECTOR" --url "$ES_URL" --index "$ES_INDEX" --run-id "$RUN_ID" --check-safety
echo

echo "Running combined proof check..."
uv run python "$INSPECTOR" --url "$ES_URL" --index "$ES_INDEX" --run-id "$RUN_ID" --check-duplicates --check-safety
echo

echo "Proof result: PASS"
echo
echo "Documentation summary:"
echo "  run_id=$RUN_ID"
echo "  elasticsearch_url=$ES_URL"
echo "  elasticsearch_index=$ES_INDEX"
echo "  duplicate_check=0"
echo "  safety_check=passed"
echo "  command=run-elasticsearch-observability-proof.sh $RUN_ID"
