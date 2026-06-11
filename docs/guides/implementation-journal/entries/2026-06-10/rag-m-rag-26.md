---
id: IJ-2026-06-10-035
date: 2026-06-10
tiers:
  - tier-0
  - tier-3
scope: RAG
plan_ref:
  - M-RAG.26
  - AUDIT-IDEAL-14.6
  - GAP-RAG-05
  - GAP-RAG-06
status: completed
commit: pending
adr: none — composes existing `workflow_orchestrator` + `IngestPipeline` policy gate
---

# Async ingest job contract + sync size guard (M-RAG.26)

## Operator request

Continue M-RAG-DEPTH Wave 2: close large-corpus ingest gaps with Tier-0 async job contract and sync path size threshold.

## Summary

Added `RagProfile.sync_ingest_max_bytes` (env `INTERGRAX_RAG_SYNC_INGEST_MAX_BYTES`, default 50MB) and `async_ingest_workflow_id`. `IngestPipeline` rejects oversized sources before loader with `sync_ingest_size_exceeded` and `async_job_recommended=true`. New catalog tool `rag.schedule_ingest_job` triggers `workflow_orchestrator` with structured parameters and idempotent reuse of active runs (same derived `idempotency_key`). Tool auto-wired when `workflow_orchestrator` integration category is configured.

## Project impact

Tier-3 hosts can enforce sync ingest limits and route large files to Prefect/Airflow/n8n workers without bypassing the harness tool surface. Actual shard/stream ingest remains in orchestrator workflows calling `rag.ingest_document` per shard.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/RAG.md` GAP-RAG-05, 06 closed |
| Plan | `docs/plan/RAG.md` M-RAG.26 **Done**, AUDIT-IDEAL-14.6 **Done** |

## Changed artifacts

- `intergrax/rag/ingest/ingest_policy.py` — sync size guard + idempotency key
- `intergrax/rag/profiles/rag_profile.py` — `sync_ingest_max_bytes`, `async_ingest_workflow_id`
- `intergrax/rag/ingest/ingest_pipeline.py` — pre-load size check
- `intergrax/tools/providers/rag/ingest_job_*.py` — schedule tool service/handler
- `intergrax/tools/providers/rag/bundle.py` — register `rag.schedule_ingest_job`
- `intergrax/applications/_shared/integration_tool_profile.py` — orchestrator category wiring
- Tests: `test_ingest_sync_size_policy.py`, `test_rag_ingest_size_guard.py`, `test_rag_schedule_ingest_job.py`

## Verification

```bash
uv run pytest tests/unit/rag/ingest/test_ingest_sync_size_policy.py tests/unit/tools/providers/rag/test_rag_schedule_ingest_job.py tests/unit/tools/providers/rag/test_rag_ingest_size_guard.py -q
```

Result: 7 passed.

## Risks and follow-ups

- Workflow workers must implement shard ingest (call `rag.ingest_document` per shard) — not in Tier-0 scope.
- Idempotency reuses active runs via `list_runs` metadata — durable dedupe depends on orchestrator backend fidelity.
- Next Wave 2 item: **M-RAG.30** (vector-store prod SLO soak gate).
