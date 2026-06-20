# MAINT implementation batch — ACP, FLOW, UAEP, ECC, MOD, LLM, TOOLS

**Date:** 2026-06-18  
**Scope:** Audit §6.1av maintenance queue (partial P2 batch)  
**Domains:** AGENT_CONTRACTS_AND_ASSEMBLY, NEXUS_EXECUTION_FLOW, UNIFIED_EXECUTION_RUNTIME, CODE_CRAFT, MODALITY, LLM_ADAPTERS, TOOLS

## Completed items

| ID | Summary |
|----|---------|
| ACP-MAINT-01 | `boundary_demo` migrated to `data.records_admin` skill; no author-time `allowed_tools` |
| ACP-MAINT-02 | `check_agent_skill_resolution.py` wired into ACP close CI umbrella |
| FLOW-MAINT-01 | `graph_runner` respects `ResiliencePolicy.allow_partial_result` |
| UAEP-AUDIT-01 | `tenant_id` propagated on UAEP `_emit` and `_emit_context_assembled` |
| ECC-MAINT-01 | `Task.metadata.codecraft_mode` overrides host profile in orchestrator |
| MOD-MAINT-01/02 | OpenCV runtime availability probe + robust test skip when `cv2` stubbed |
| LLM-MAINT-01 | LLM typed-returns check in `intergrax doctor check` |
| TOOL-MAINT-04 | Tool injection + legacy plan boolean checks in doctor |

## Verification

```bash
python scripts/check_agent_skill_resolution.py
python scripts/check_agent_acp_close_ci.py
uv run pytest tests/unit/tools/providers/codecraft/test_codecraft_run.py::test_resolve_codecraft_profile_task_metadata_overrides_mode -q
uv run pytest tests/unit/model_inference/ -q
uv run pytest tests/unit/agents/test_uaep_executor.py -q
```

## ADR

No ADR needed — maintenance wiring and fleet hygiene; no new platform contracts.
