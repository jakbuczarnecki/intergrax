---
id: IJ-2026-06-10-007
date: 2026-06-10
tiers:
  - tier-0
scope: RAG
plan_ref:
  - M-RAG.35
  - GAP-RAG-20
status: completed
adr: none — contract tests on existing VectorStore tenant enforcement
---

# Cross-backend tenant isolation contract tests (M-RAG.35)

## Operator request

Close Wave 2 GAP-RAG-20: uniform tenant isolation contract across vector-store backends.

## Summary

Added `tenant_isolation_contract.py` with `run_tenant_isolation_contract()` — tenant A ingest, tenant B cross-query must not leak, ingest metadata mismatch must raise. Gate tests parametrize `inmemory`, `pgvector`, `weaviate`, and `qdrant` (fake in-process client). Integration `test_tenant_isolation_qdrant_live` reuses contract when Qdrant is reachable.

## Verification

```bash
uv run pytest tests/unit/rag/vectorstore/test_vectorstore_cross_tenant_isolation.py -m gate -q
```

Result: 6 passed.

## Next step

Wave 3 begins with **M-RAG.27** (OTel spans on retrieve + ingest).
