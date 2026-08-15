---
id: IJ-2026-06-10-044
date: 2026-06-10
tiers:
  - tier-0
scope: RAG
plan_ref:
  - M-RAG.35
  - GAP-RAG-20
status: completed
commit: pending
adr: none — contract tests on existing VectorStore tenant enforcement
---

# Cross-backend tenant isolation contract tests (M-RAG.35)

## Operator request

Close Wave 2 GAP-RAG-20: uniform tenant isolation contract across vector-store backends.

## Summary

Added `tenant_isolation_contract.py` with `run_tenant_isolation_contract()` — tenant A ingest, tenant B cross-query must not leak, ingest metadata mismatch must raise. Gate tests parametrize `inmemory`, `pgvector`, `weaviate`, and `qdrant` (fake in-process client). Integration `test_tenant_isolation_qdrant_live` reuses contract when Qdrant is reachable.

## Project impact

All supported vector-store backends must pass a uniform tenant-isolation contract in CI, closing GAP-RAG-20 and hardening multi-tenant RAG deployments.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/RAG.md` GAP-RAG-20 closed |
| Plan | `docs/project/maintainers/plans/RAG.md` M-RAG.35 **Done** |

## Changed artifacts

- `intergrax/rag/vectorstore/tenant_isolation_contract.py` — shared contract runner
- `tests/unit/rag/vectorstore/test_vectorstore_cross_tenant_isolation.py` — parametrized gate
- `tests/integration/rag/test_tenant_isolation_qdrant_live.py` — live Qdrant path

## Verification

```bash
uv run pytest tests/unit/rag/vectorstore/test_vectorstore_cross_tenant_isolation.py -m gate -q
```

Result: 6 passed.

## Risks and follow-ups

- Beta backends (`pinecone`, `milvus`, `vespa`) await contract parametrization after ops soak promotion.
- Wave 3 continues with **M-RAG.27** OTel spans on retrieve and ingest.
