# TRACE-X-P4 — Model Call & Context Decision Attribution

Status: **P4-R7 on `development`; READY FOR AUDIT** (not independently closed).

Primary freeze gate: **FRZ-TRC-05** → **OPEN** until independent audit on GitHub HEAD.

## P4-R7 transitive production composition reachability (R6 blocker remediation)

| Blocker | Status |
|---------|--------|
| P4-R6-Q-BLK-TRANSITIVE-REACHABILITY-01 | REMEDIATED in R7 (`_import_closure` + class/composition-factory edges over full closure; BFS >1 hop) |
| P4-R6-Q-BLK-PATH-PREFIX-CLASSIFICATION-02 | REMEDIATED in R7 (removed `agents/` / `applications/` verdict shortcuts) |
| P4-R6-Q-BLK-STREAM-CLOSURE-03 | REMEDIATED in R7 (stream surfaces use same graph; external `stream_*` call scan on `production_reachable_modules`) |

**R7 evidence baseline**

| Field | Value |
|-------|-------|
| START_HEAD | `7350e6b5827de23b53173a4edd6654b89d2f6371` |
| FINAL_COMMIT | `ca1b4ec636af6c9c04d1c1a5c3453055384fb4eb` |

### Mechanical reachability graph (R7)

- **Single owner:** `build_production_reachability_graph()` in `_trace_x_p4_r6_reachability_analysis.py`.
- **Seeds:** sanctioned `SANCTIONED_PRODUCTION_ROOT` + `SANCTIONED_PRODUCTION_ROUTER` module paths from `PRODUCTION_COMPOSITION_SITE_REGISTRY`.
- **Import closure:** repository-local imports from seeds (fixed point).
- **Composition edges:** AST class-instantiation edges within closure + explicit P4 composition factory callees (`StrategyExecutionRouter`, `InferenceExecutor`, `build_governed_inference_executor`, `wrap_model_call_runtime_evidence`, `ModelCallRuntimeEvidenceAdapter`, `RuntimeConfig`) — not general `apply_*` procedural calls.
- **Traversal:** BFS from seeds; `MechanicalReachabilityResult` includes `reachable_from` / `evidence_edges`; `UNRESOLVED` for `unresolved:*` synthetic edges and ambiguous composition-relevant resolution.
- **Gates:** TXP4R7-Q01–Q09; R6 regression `test_trace_x_p4_r6_closed_world.py` unchanged green.

### Reachability counts (mechanical at R7 START_HEAD)

| Metric | Value |
|--------|------:|
| Sanctioned seed modules | 4 |
| Import closure modules | 3690 |
| Composition edges | 6161 |
| Production-reachable modules (BFS) | 109 |
| NON_PRODUCTION model-call surfaces | 29 |
| Mechanically `PRODUCTION_REACHABLE` among NON_PRODUCTION | 0 |
| Unresolved | 0 |
| InferenceExecutor `generate_structured` | `NOT_REACHABLE_FROM_SANCTIONED_PRODUCTION_ROOT` |

### Negative sensitivity (R7)

| Case | Gate |
|------|------|
| Two-hop InferenceExecutor synthetic | TXP4R7-Q04 |
| `agents/` path not shortcut | TXP4R7-Q05 |
| `applications/` path not shortcut | TXP4R7-Q06 |
| Transitive external stream | TXP4R7-Q07 |
| Ambiguous edge fail-closed | TXP4R7-Q08 |

```text
uv run pytest tests/qualification/trace_x/test_trace_x_p4_r7_closed_world.py tests/qualification/trace_x/test_trace_x_p4_r6_closed_world.py -p no:xdist -q
uv run pytest tests/qualification/trace_x/ -k test_trace_x_p4 -p no:xdist -q
```

## P4-R6 mechanical reachability & real composition wrap (R5 blocker remediation)

| Blocker | Status |
|---------|--------|
| P4-R5-Q-BLK-NONPROD-PROOF-01 | REMEDIATED in R6 (`evaluate_all_non_production_reachability` vs `NON_PRODUCTION_MODEL_REACHABILITY_REGISTRY`; prose-only parity insufficient) |
| P4-R5-Q-BLK-WRAP-PROPAGATION-02 | REMEDIATED in R6 (`materialize_runtime_config(..., llm_adapter=)` returns canonical `ModelCallRuntimeEvidenceAdapter`) |

**R6 evidence baseline**

| Field | Value |
|-------|-------|
| START_HEAD | `c916c8fff0e772c6d09662be28f81318402b44b0` |
| FINAL_COMMIT | `329e1efecde08ebfac8422decd859e4c29b4db82` |

### Mechanical reachability (R6)

- **Types:** `ModelConsumerSurface`, `CompositionEdge`, `ReachabilityVerdict`, `ReachabilityReason` (`_trace_x_p4_r6_reachability_types.py`).
- **Evaluator:** `_trace_x_p4_r6_reachability_analysis.py` — sanctioned production root/router modules → consumer instantiation edges (AST, P4-bounded) → per-surface verdict; InferenceExecutor via production composition registry (`GOVERNED_INFERENCE_EXECUTOR_CALL`, `STRATEGY_ROUTER_INFERENCE_EXECUTOR_KW`); wrapper `stream_*` via sanctioned-module call scan.
- **Gate:** `compare_mechanical_reachability_to_expectations` (TXP4R6-Q03); contradictions/orphans/unknown → FAIL.
- **Negative sensitivity:** TXP4R6-Q07 (synthetic InferenceExecutor edge), TXP4R6-Q08 (synthetic auxiliary consumer edge); TXP4R6-Q09 registry injection flips InferenceExecutor verdict.

### Reachability counts (mechanical at R6 START_HEAD)

| Category | count |
|----------|------:|
| NON_PRODUCTION model-call surfaces | 29 |
| Mechanically production reachable | 0 |
| Mechanically non-production (not `PRODUCTION_REACHABLE`) | 29 |
| Unresolved | 0 |
| InferenceExecutor `generate_structured` | `NOT_REACHABLE_FROM_SANCTIONED_PRODUCTION_ROOT` |

### Runtime-config wrap (R6)

- **Test:** TXP4R6-Q05 / Q06 — public `llm_adapter=` on `materialize_runtime_config`; raw and pre-wrapped inputs; idempotency via `wrap_model_call_runtime_evidence` behavior (no private field access).

```text
uv run pytest tests/qualification/trace_x/test_trace_x_p4_r6_closed_world.py -p no:xdist -q
```

**R6 session evidence:** `.tmp/session/trace-x-p4-r6/p4-full-pytest.log` (82 passed P4 suite).

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
uv run pytest tests/qualification/trace_x/test_trace_x_p4_model_context_attribution.py tests/qualification/trace_x/test_trace_x_p4_r1_recorder_tenant.py tests/qualification/trace_x/test_trace_x_p4_r2_lifecycle.py tests/qualification/trace_x/test_trace_x_p4_r2_qualification_gates.py tests/qualification/trace_x/test_trace_x_p4_r3_abandoned_context.py tests/qualification/trace_x/test_trace_x_p4_r3_closed_world.py tests/qualification/trace_x/test_trace_x_p4_r4_closed_world.py tests/qualification/trace_x/test_trace_x_p4_r5_closed_world.py tests/qualification/trace_x/test_trace_x_p4_r6_closed_world.py -p no:xdist -q
```

Do **not** mark TRACE-X-P4 CLOSED or promote FRZ-TRC-05 without independent audit.
