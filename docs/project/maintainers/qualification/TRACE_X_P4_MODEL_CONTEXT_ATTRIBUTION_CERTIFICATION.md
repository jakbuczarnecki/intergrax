# TRACE-X-P4 — Model Call & Context Decision Attribution

Status: **P4-R5 on `development`; READY FOR AUDIT** (not independently closed).

Primary freeze gate: **FRZ-TRC-05** → **OPEN** until independent audit on GitHub HEAD.

## P4-R5 production composition & reachability (R4 blocker remediation)

| Blocker | Status |
|---------|--------|
| P4-R4-Q-BLK-PRODUCTION-COMPOSITION-01 | REMEDIATED in R5 (mechanical production composition discovery + registry parity + reachability proofs; behavioral P4 wrap gate supersedes source-text-only TXP4R4-Q09 for sanctioned bridge) |

**R5 evidence baseline**

| Field | Value |
|-------|-------|
| START_HEAD | `bef997db3fd40eab7d47ca605f12dcd0145bcb0e` |
| FINAL_COMMIT | `36d1733d8a4eca34861135029bb02fbb9f62e876` |

### Production composition closed world (R5)

- **Discovery:** `discover_production_composition_sites_ast()` → `DISCOVERED_PRODUCTION_COMPOSITION_SITES` (StrategyExecutionRouter, inference_executor=, InferenceExecutor, build_governed_inference_executor, RuntimeConfig llm_adapter, P4 wrap assignments).
- **Registry:** static `PRODUCTION_COMPOSITION_SITE_REGISTRY` in `tests/qualification/trace_x/_trace_x_p4_production_composition_registry.py`.
- **Parity:** `compare_production_composition_sites_to_registry()` (TXP4R5-Q02 / Q03 negative sensitivity).
- **NON_PRODUCTION reachability:** static `NON_PRODUCTION_MODEL_REACHABILITY_REGISTRY` (29 rows, parity TXP4R5-Q06).

### Counts (parity at R5 START_HEAD `bef997db3fd40eab7d47ca605f12dcd0145bcb0e`)

| Inventory | discovered | registered | unknown | orphan |
|-----------|------------|------------|---------|--------|
| Production composition sites | 16 | 16 | 0 | 0 |
| Model-call (R4) | 62 | 62 | 0 | 0 |
| Context (R4) | 10 | 10 | 0 | 0 |
| NON_PRODUCTION reachability rows | 29 | 29 | 0 | 0 |

### InferenceExecutor disposition (R5)

| Question | Answer |
|----------|--------|
| Reachable from sanctioned production composition | **NO** |
| Evidence | Zero `STRATEGY_ROUTER_INFERENCE_EXECUTOR_KW` and zero `GOVERNED_INFERENCE_EXECUTOR_CALL` in closed-world production scan; sanctioned routers (`graph_executor`, `host_task`, `orchestration`) register `inference_executor_supplied=False`; `build_governed_inference_executor` = internal factory only |
| Registry classification | `NON_PRODUCTION` (`inference.py` / `generate_structured`) |
| IN-SCOPE BLOCKER | **NO** |

### Sanctioned production LLM path

`materialize_runtime_config` → `resolve_llm_adapter` → `wrap_model_call_runtime_evidence` → `config.llm_adapter` (TXP4R5-Q11 behavioral). Alternate `ModelCallRuntimeEvidenceAdapter(...)` instantiation outside canonical module = **0** (TXP4R5-Q10).

## P4-R4 closed-world disposition (accepted)

| Blocker | Status |
|---------|--------|
| P4-R3-Q-BLK-CLOSED-WORLD-TAUTOLOGY-01 | REMEDIATED in R4 |
| P4-R3-Q-BLK-CONTEXT-CLOSED-WORLD-02 | REMEDIATED in R4 |

Model/context discovery, independent static registries, parity, and R4 negative sensitivity remain required (TXP4R4-Q02..Q09).

## P4-R1 / P4-R2 / P4-R3 blocker disposition (accepted)

| Blocker | Status |
|---------|--------|
| P4-BLK-TENANT-CONTEXT-01 | RESOLVED |
| P4-BLK-CONTEXT-DECISION-01 | RESOLVED |
| P4-R2-BLK-ABANDONED-CONTEXT-01 | RESOLVED |
| P4-R2-Q-BLK-CLOSED-WORLD-01 | SUPERSEDED by R4 |

## Qualification (sequential, `-p no:xdist`)

```text
uv run pytest tests/qualification/trace_x/test_trace_x_p4_model_context_attribution.py tests/qualification/trace_x/test_trace_x_p4_r1_recorder_tenant.py tests/qualification/trace_x/test_trace_x_p4_r2_lifecycle.py tests/qualification/trace_x/test_trace_x_p4_r2_qualification_gates.py tests/qualification/trace_x/test_trace_x_p4_r3_abandoned_context.py tests/qualification/trace_x/test_trace_x_p4_r3_closed_world.py tests/qualification/trace_x/test_trace_x_p4_r4_closed_world.py tests/qualification/trace_x/test_trace_x_p4_r5_closed_world.py -p no:xdist -q
```

**R5 session evidence:** `.tmp/session/trace-x-p4-r5/p4-full-pytest.log` (73 passed).

Do **not** mark TRACE-X-P4 CLOSED or promote FRZ-TRC-05 without independent audit.
