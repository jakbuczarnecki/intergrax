---
id: IJ-2026-06-10-014
date: 2026-06-10
tiers:
  - tier-0
scope: RAG
plan_ref:
  - M-RAG.37
  - GAP-RAG-22
status: completed
adr: none — ingest policy extension on existing pipeline guard pattern
---

# Semantic chunking ingest size guard (M-RAG.37)

## Operator request

Execute next M-RAG-DEPTH item after M-RAG.34.

## Summary

- `RagProfile.semantic_chunking_max_chars` (default 100_000; env `INTERGRAX_RAG_SEMANTIC_CHUNKING_MAX_CHARS`)
- `ingest_policy.semantic_chunking_allowed()` — applies only when chunking strategy is `semantic`
- `IngestPipeline` checks after load, before `split_documents` — reason `semantic_chunking_size_exceeded:{chars}>{max}`
- `async_job_recommended=true` on rejection (same posture as sync byte guard)

## Verification

```bash
uv run pytest tests/unit/rag/ingest/test_semantic_chunking_size_guard.py -m gate -q
uv run pytest tests/unit/rag/ -m gate -q
```

## Next step

**M-RAG.36** — RAG load/soak gate (last M-RAG-DEPTH item).
