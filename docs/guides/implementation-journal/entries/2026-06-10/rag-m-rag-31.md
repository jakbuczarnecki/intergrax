---
id: IJ-2026-06-10-040
date: 2026-06-10
tiers:
  - tier-0
scope: RAG
plan_ref:
  - M-RAG.31
  - GAP-RAG-14
status: completed
commit: pending
adr: none — policy module composes existing RagProfile metadata fields
---

# Embedding model version mismatch policy (M-RAG.31)

## Operator request

Continue Wave 3 after M-RAG.29 formal citations.

## Summary

- `intergrax/rag/governance/embedding_version_policy.py`
- Ingest: warn on incoming/indexed version mismatch; `IngestResult.version_warnings` + `reindex_recommended`
- Retrieve: optional filter via `embedding_version_filter_on_retrieve` (env `INTERGRAX_RAG_EMBEDDING_VERSION_FILTER_RETRIEVE`)
- `register_reindex_queue_hook()` for Tier-3/workflow reindex scheduling

## Project impact

Embedding model upgrades surface explicit version warnings and optional retrieve-time filtering, preventing silent retrieval quality degradation after re-embedding campaigns.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/RAG.md` GAP-RAG-14 closed |
| Plan | `docs/plan/RAG.md` M-RAG.31 **Done** |

## Changed artifacts

- `intergrax/rag/governance/embedding_version_policy.py` — mismatch policy
- `intergrax/rag/ingest/ingest_pipeline.py` — ingest warnings on version drift
- `intergrax/rag/retrieval/retrieval_service.py` — optional retrieve filter

## Verification

```bash
uv run pytest tests/unit/rag/governance/test_embedding_version_policy.py tests/unit/rag/ingest/test_ingest_embedding_version_policy.py tests/unit/rag/retrieval/test_retrieval_embedding_version_filter.py -m gate -q
```

## Risks and follow-ups

- Reindex queue hook requires Tier-3 operator wiring for automated remediation.
- Next Wave 3 item: **M-RAG.32** optional LLM QueryRouter tier classifier.
