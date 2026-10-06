# TRACE-X-P4 — Model Call & Context Decision Attribution

Status: **READY FOR AUDIT** (implementation commit on `development`; not independently closed).

Primary freeze gate: **FRZ-TRC-05** → **READY FOR INDEPENDENT CLOSURE REVIEW** when qualification passes on GitHub HEAD.

## Evidence summary

- `ContextAssemblyPayloadV3` records `model_input_messages_hash` on `CONTEXT_ASSEMBLED`.
- `LlmCallPayloadV2` records matching hash, `provider`, and `execution_scope` on `LLM_CALL`.
- Canonical join: `(task_id, run_id, attempt_id, execution_id, model_input_messages_hash)`.

## Qualification

```text
uv run pytest tests/qualification/trace_x/test_trace_x_p4_model_context_attribution.py -p no:xdist -q
```
