---
id: IJ-2026-06-10-038
date: 2026-06-10
tiers:
  - tier-0
scope: RAG
plan_ref:
  - M-RAG.29
  - GAP-RAG-13
status: completed
commit: pending
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

## Project impact

Downstream agents and tools receive typed `Citation` objects on every successful retrieve, enabling auditable grounding without ad-hoc chunk metadata parsing.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/RAG.md` GAP-RAG-13 closed |
| Plan | `docs/plan/RAG.md` M-RAG.29 **Done** |

## Changed artifacts

- `intergrax/rag/retrieval/citation.py` — `Citation`, `citation_from_chunk()`
- `intergrax/rag/retrieval/retrieval_service.py` — `RetrievalResult.citations`
- `intergrax/tools/providers/rag/rag_retrieve.py` — `RagCitationResult` on catalog path

## Verification

```bash
uv run pytest tests/unit/rag/retrieval/test_rag_citation_engine_gate.py -m gate -q
uv run pytest tests/unit/tools/providers/rag/test_rag_retrieve.py -q
```

## Risks and follow-ups

- Citation fidelity depends on ingest metadata completeness; legacy indexes may lack fields until reindex.
- Next Wave 3 item: **M-RAG.31** embedding version mismatch policy.
