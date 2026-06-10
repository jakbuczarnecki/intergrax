---
id: IJ-2026-06-10-011
date: 2026-06-10
tiers:
  - tier-0
scope: RAG
plan_ref:
  - M-RAG.31
  - GAP-RAG-14
status: completed
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

## Verification

```bash
uv run pytest tests/unit/rag/governance/test_embedding_version_policy.py tests/unit/rag/ingest/test_ingest_embedding_version_policy.py tests/unit/rag/retrieval/test_retrieval_embedding_version_filter.py -m gate -q
```

## Next step

**M-RAG.32** — completed in IJ-2026-06-10-012.
