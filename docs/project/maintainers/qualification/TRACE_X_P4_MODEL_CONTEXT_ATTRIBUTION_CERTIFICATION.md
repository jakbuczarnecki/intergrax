# TRACE-X-P4 — Model Call & Context Decision Attribution

Status: **P4-R1 implementation on `development`; READY FOR AUDIT** (not independently closed).

Primary freeze gate: **FRZ-TRC-05** → **READY FOR INDEPENDENT CLOSURE REVIEW** when qualification passes on GitHub HEAD.

## P4-R1 blocker disposition

| Blocker | Status |
|---------|--------|
| P4-BLK-TENANT-CONTEXT-01 | RESOLVED |
| P4-BLK-CONTEXT-DECISION-01 | RESOLVED |
| P4-Q-BLK-01 | RESOLVED |
| P4-TYP-01 | RESOLVED |
| P4-STREAM-01 | RESOLVED (streaming classified NOT PRODUCTION PRIMARY on certified Nexus/agents/applications paths) |

## Evidence summary (R1)

- `ContextAssemblyPayloadV4` records `model_input_messages_hash` and `context_decision_evidence_fingerprint` on `CONTEXT_ASSEMBLED`.
- `LlmCallPayloadV3` records matching hash, typed `ModelCallExecutionScope`, `provider`, and exact `context_assembly_event_id` on `LLM_CALL`.
- Canonical attribution: exact `CONTEXT_ASSEMBLED.event_id` + execution identity + tenant + input hash + decision fingerprint coherence.

## Qualification

```text
uv run pytest tests/qualification/trace_x/test_trace_x_p4_model_context_attribution.py tests/qualification/trace_x/test_trace_x_p4_r1_recorder_tenant.py -p no:xdist -q
```

Do **not** mark TRACE-X-P4 CLOSED without independent audit.
