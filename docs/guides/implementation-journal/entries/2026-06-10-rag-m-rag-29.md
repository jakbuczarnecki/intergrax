---
id: IJ-2026-06-10-010
date: 2026-06-10
tiers:
  - tier-0
scope: RAG
plan_ref:
  - M-RAG.29
  - GAP-RAG-13
status: completed
adr: none — formalizes existing chunk metadata into typed Citation contracts
---

# Formal Citation model on retrieval engine + rag.retrieve (M-RAG.29)

## Operator request

Execute next Wave 3 step after M-RAG.28 retriever resilience.

## Summary

- `intergrax/rag/retrieval/citation.py` — `Citation`, `citation_from_chunk()`, `citations_from_chunks()`
- `RetrievalResult.citations` populated by `RetrievalService` on successful retrieve
- `RagCitationResult` + `RagRetrieveOutput.citations` on catalog `rag.retrieve`
- Poisoning filter keeps chunks and citations aligned

## Verification

```bash
uv run pytest tests/unit/rag/retrieval/test_rag_citation_engine_gate.py -m gate -q
uv run pytest tests/unit/tools/providers/rag/test_rag_retrieve.py -q
```

## Next step

**M-RAG.31** — completed in IJ-2026-06-10-011.
