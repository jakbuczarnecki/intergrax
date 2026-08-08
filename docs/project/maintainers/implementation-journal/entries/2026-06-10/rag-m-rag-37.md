---
id: IJ-2026-06-10-046
date: 2026-06-10
tiers:
  - tier-0
scope: RAG
plan_ref:
  - M-RAG.37
  - GAP-RAG-22
status: completed
commit: pending
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

## Project impact

Semantic chunking ingest rejects oversize documents before expensive LLM splitting, closing GAP-RAG-22 and aligning with async ingest recommendations.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/RAG.md` GAP-RAG-22 closed |
| Plan | `docs/project/maintainers/plans/RAG.md` M-RAG.37 **Done** |

## Changed artifacts

- `intergrax/rag/ingest/ingest_policy.py` — `semantic_chunking_allowed()` guard
- `intergrax/rag/ingest/ingest_pipeline.py` — pre-split size check
- `intergrax/rag/profiles/rag_profile.py` — `semantic_chunking_max_chars` profile field

## Verification

```bash
uv run pytest tests/unit/rag/ingest/test_semantic_chunking_size_guard.py -m gate -q
uv run pytest tests/unit/rag/ -m gate -q
```

## Risks and follow-ups

- Default 100k char limit may need per-product tuning for legal/research corpora.
- Completes M-RAG-DEPTH alongside **M-RAG.36** load/soak gate.
