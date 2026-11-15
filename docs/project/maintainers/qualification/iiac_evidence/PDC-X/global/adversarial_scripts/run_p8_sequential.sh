#!/usr/bin/env bash
# PDC-P8-03 sequential runner: one uv/pytest process at a time. Logs under .tmp/proof/pdc-x-p8/
cd "$(dirname "$0")/../../.." || exit 2
L=.tmp/proof/pdc-x-p8
mkdir -p $L
echo "audited_sha=$(git rev-parse HEAD)" > $L/00_sha.txt
step() { # name timeout cmd...
  name=$1; to=$2; shift 2
  date -u +%FT%TZ > $L/$name.start
  timeout $to "$@" > $L/$name.log 2>&1
  echo "exit=$?" > $L/$name.end; date -u +%FT%TZ >> $L/$name.end
}
step 00_uv_sync 900 uv sync --frozen --extra dev-ci --extra mcp --extra llm-openai --extra llm-anthropic --extra llm-mistral --extra llm-ollama --extra llm-langchain-ollama --extra llm-gemini --extra llm-bedrock --extra llm-cohere-native --extra integrations-duckdb
step 01_ecap_full 5400 uv run --no-sync python scripts/enterprise_audit/run_enterprise_audit.py run --mode FULL --output-dir $L/ecap
step 02_pdcx 3600 uv run --no-sync pytest tests/qualification/pdc_x -q -rf --tb=short
step 03_gates 1800 uv run --no-sync pytest tests/qualification/dup_x tests/unit/architecture/test_ebh_4_r1_nexus_encapsulation_gate.py tests/unit/scripts/test_check_application_production_gates.py tests/unit/runtime/qualification/test_ebh_4_b4_tenant_isolation.py tests/unit/runtime/architecture/test_platform_execution_unification_u5_final_zero_bypass.py tests/unit/runtime/execution/test_runtime_request_tenant_envelope.py tests/unit/runtime/task/test_unified_task_runner_execution_boundary.py -q -rf --tb=short
while IFS=$'\t' read -r l cmd; do
  step "04_contract_$l" 5400 bash -c "${cmd/pytest /pytest -rf --tb=short }"
done < .tmp/session/ia-global/contracts.txt
step 05_tenant_x 1800 uv run --no-sync pytest tests/qualification/tenant_x -q -rf --tb=short
step 06_known_reds 3600 uv run --no-sync pytest tests/qualification/ebh_7 tests/qualification/roadmap_replay_x tests/qualification/dist_scale_x tests/unit/governance/test_iiac_activation_gate.py tests/unit/governance/test_iiac_negative_guards.py tests/qualification/compat_x -q -rf --tb=short
echo done > $L/ALL_DONE
