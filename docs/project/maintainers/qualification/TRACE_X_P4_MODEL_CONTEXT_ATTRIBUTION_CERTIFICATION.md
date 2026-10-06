# TRACE-X-P4 — Model Call & Context Decision Attribution

Status: **P4-R4 on `development`; READY FOR AUDIT** (not independently closed).

Primary freeze gate: **FRZ-TRC-05** → **OPEN** until independent audit on GitHub HEAD.

## P4-R4 closed-world disposition (R3 blocker remediation)

| Blocker | Status |
|---------|--------|
| P4-R3-Q-BLK-CLOSED-WORLD-TAUTOLOGY-01 | REMEDIATED in R4 (discovery independent from explicit registry; no default NON_PRODUCTION fallback) |
| P4-R3-Q-BLK-CONTEXT-CLOSED-WORLD-02 | REMEDIATED in R4 (AST context discovery + explicit `CONTEXT_SURFACE_REGISTRY`) |

### Model-call closed world (R4 evidence)

- **Discovery:** `discover_model_call_surfaces_ast()` → `DISCOVERED_MODEL_CALL_SURFACES` (syntax-only LLMAdapter-shaped call sites under `intergrax/` / `agents/` / `applications/`).
- **Registry:** static `MODEL_CALL_SURFACE_REGISTRY` in `tests/qualification/trace_x/_trace_x_p4_model_surface_registry.py` (not generated at import).
- **Parity:** `compare_model_call_surfaces_to_registry()` — mechanical `unknown` / `orphan` / duplicate-key checks (TXP4R4-Q02).
- **Negative sensitivity:** synthetic unregistered `(path, method)` fails parity (TXP4R4-Q03).

### Context closed world (R4 evidence)

- **Discovery:** `discover_context_surfaces_ast()` → `DISCOVERED_CONTEXT_SURFACES` (recorder calls, CONTEXT_ASSEMBLED emits, payload construction, attribution bind sites).
- **Registry:** static `CONTEXT_SURFACE_REGISTRY` in `tests/qualification/trace_x/_trace_x_p4_context_surface_registry.py`.
- **Parity:** `compare_context_surfaces_to_registry()` (TXP4R4-Q04).
- **Negative sensitivity:** synthetic unregistered context producer fails parity (TXP4R4-Q05).

### Counts (parity at R4 START_HEAD `069315c5fe45afc4ed39c5b02900fd56889e3bb5`)

| Side | discovered | registered | unknown | orphan |
|------|------------|------------|---------|--------|
| Model | 62 | 62 | 0 | 0 |
| Context | 10 | 10 | 0 | 0 |

## P4-R1 / P4-R2 / P4-R3 blocker disposition (accepted)

| Blocker | Status |
|---------|--------|
| P4-BLK-TENANT-CONTEXT-01 | RESOLVED |
| P4-BLK-CONTEXT-DECISION-01 | RESOLVED |
| P4-R2-BLK-ABANDONED-CONTEXT-01 | RESOLVED (execution-bound pending relation) |
| P4-R2-Q-BLK-CLOSED-WORLD-01 | SUPERSEDED by R4 independent registry model |

## Qualification (sequential, `-p no:xdist`)

```text
uv run pytest tests/qualification/trace_x/test_trace_x_p4_model_context_attribution.py tests/qualification/trace_x/test_trace_x_p4_r1_recorder_tenant.py tests/qualification/trace_x/test_trace_x_p4_r2_lifecycle.py tests/qualification/trace_x/test_trace_x_p4_r2_qualification_gates.py tests/qualification/trace_x/test_trace_x_p4_r3_abandoned_context.py tests/qualification/trace_x/test_trace_x_p4_r3_closed_world.py tests/qualification/trace_x/test_trace_x_p4_r4_closed_world.py -p no:xdist -q
```

Pass 1 manifest: `.tmp/session/trace-x-p4-r4/pass1_observed_nodeids.json` (`TRACE_X_P4_PASS1=1`).

Do **not** mark TRACE-X-P4 CLOSED or promote FRZ-TRC-05 without independent audit.
