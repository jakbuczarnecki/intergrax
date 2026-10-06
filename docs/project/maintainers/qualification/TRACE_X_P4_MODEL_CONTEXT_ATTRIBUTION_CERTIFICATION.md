# TRACE-X-P4 — Model Call & Context Decision Attribution

Status: **P4-R2 on `development`; READY FOR AUDIT** (not independently closed).

Primary freeze gate: **FRZ-TRC-05** → **READY FOR INDEPENDENT CLOSURE REVIEW** when qualification passes on GitHub HEAD.

## P4-R1 / P4-R2 blocker disposition

| Blocker | Status |
|---------|--------|
| P4-BLK-TENANT-CONTEXT-01 | RESOLVED |
| P4-BLK-CONTEXT-DECISION-01 | RESOLVED |
| P4-Q-BLK-01 | RESOLVED |
| P4-TYP-01 | RESOLVED |
| P4-STREAM-01 | RESOLVED (streaming classified NOT PRODUCTION PRIMARY on certified Nexus/agents/applications paths) |
| P4-R1-BLK-CONTEXT-REF-LIFECYCLE-01 | RESOLVED (token/stack scoped relation; `model_call_attribution_scope` floor reset) |
| P4-R1-Q-BLK-02 | RESOLVED (`P4_R2_GATE_REGISTRY` + Pass 1 session manifest) |

## Evidence summary (R2)

- Pending `CONTEXT_ASSEMBLED` → model-call relation is token-scoped with nested stack reset on `model_call_attribution_scope` exit (not solely `record_llm_call_runtime_event` cleanup).
- `ContextAssemblyPayloadV4` / `LlmCallPayloadV3` / `ModelContextAttribution` unchanged (exact EventId + hash + fingerprint).
- Mechanical gates TXP4R2-Q01..Q36 + TXP4R2-LIFE-01..06 mapped in `tests/qualification/trace_x/_trace_x_p4_support.py`.

## Qualification (sequential, `-p no:xdist`)

Pass 1 (set `TRACE_X_P4_PASS1=1` for session manifest):

```text
uv run pytest tests/qualification/trace_x/test_trace_x_p4_model_context_attribution.py tests/qualification/trace_x/test_trace_x_p4_r1_recorder_tenant.py tests/qualification/trace_x/test_trace_x_p4_r2_lifecycle.py tests/qualification/trace_x/test_trace_x_p4_r2_qualification_gates.py -p no:xdist -q
```

Do **not** mark TRACE-X-P4 CLOSED or FRZ-TRC-05 PASS without independent audit.
