---
id: IJ-2026-06-10-005
date: 2026-06-10
tiers:
  - tier-0
scope: RAG
plan_ref:
  - M-RAG.30
  - GAP-RAG-07
status: completed
adr: none — soak contract on existing VectorStore protocol; no new backend
---

# Vector-store prod SLO soak gate (M-RAG.30)

## Operator request

Continue M-RAG-DEPTH Wave 2: add RAG vector-store production SLO soak gate for stable catalog slugs and document ops runbook.

## Summary

Added `intergrax/rag/vectorstore/soak/prod_slo.py` with `run_vectorstore_soak()` — ingest, query, metadata filter, delete, p95 latency budget. Gate unit tests verify soak on in-memory harness and manifest stability tiers (`qdrant`/`pgvector`/`chroma`/`weaviate` stable; `pinecone`/`milvus`/`vespa` beta pending ops soak). Extended integration `test_vectorstore_real_backends.py` with stable-slug soak parametrization and `vectorstore_soak` marker. Updated INTEGRATIONS vector-store runbook with soak commands and promotion policy.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/RAG.md` GAP-RAG-07 closed |
| Plan | `docs/plan/RAG.md` M-RAG.30 **Done** |
| Integrations | `docs/architecture/INTEGRATIONS.md` §Vector store runbook |

## Verification

```bash
uv run pytest tests/unit/rag/vectorstore/test_vectorstore_prod_slo_soak.py -m gate -q
```

## Risks and follow-ups

- Beta promotion (`pinecone`, `milvus`, `vespa`) requires manual manifest update after ops environment soak passes.
- Next Wave 2 item: **M-RAG.33** (GraphRAG Tier-3 prod profile).
