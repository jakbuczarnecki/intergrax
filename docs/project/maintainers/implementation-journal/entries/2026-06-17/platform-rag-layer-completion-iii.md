---
id: IJ-2026-06-17-003
date: 2026-06-17
tiers:
  - tier-0
  - tier-3
scope: RAG
plan_ref:
  - M-RAG.62
  - M-RAG.63
  - M-RAG.64
  - M-RAG.65
  - M-RAG.66
  - M-RAG.67
  - M-RAG.68
status: completed
commit: 7e52ce8c
adr: none — extensions within existing RAG contracts; no new cross-cutting semantics
---

# RAG — Layer Completion iteration III (M-RAG-ITERATION-III)

## Operator request

Execute Layer Completion Mode on the RAG domain after accepting strategic proposals A–H and L (reject I/J/K).

## Summary

Closed remaining P1 gaps (tenant isolation on stable vector slugs, profile bootstrap validation), delivered reference async ingest planner, evaluation metric expansion, beta promotion readiness gate, chunking plugin registry, collection ACL, legacy deprecation timeline, and documentation convergence III.

## Project impact

Tier-3 hosts get fail-fast `RagProfile` validation, extended tenant isolation on all stable vector backends, optional collection ACL, and a reference shard planner for large-corpus ingest workflows. Harness evaluation supports precision@k and nDCG@k.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/RAG.md` — M-RAG-ITERATION-III |
| Plan | `docs/project/maintainers/plans/RAG.md` — Phase M-RAG-ITERATION-III |
| Validator | `intergrax/rag/profiles/rag_profile_validator.py` |
| Tenant contract | `intergrax/rag/vectorstore/tenant/tenant_isolation_contract.py` |
| Reference workflow | `intergrax/applications/_shared/reference_workflows/rag_async_ingest.py` |

## Changed artifacts

- `intergrax/rag/profiles/rag_profile_validator.py` — M-RAG.63
- `intergrax/rag/bootstrap/rag_stack_bootstrap.py` — assert wiring
- `intergrax/rag/vectorstore/tenant/tenant_isolation_contract.py` — M-RAG.62
- `intergrax/rag/vectorstore/governance/collection_access_policy.py` — M-RAG.65
- `intergrax/rag/vectorstore/vectorstore_manager.py` — ACL enforcement
- `intergrax/rag/document_splitters/registry/plugin_registry.py` — M-RAG.66
- `intergrax/rag/evaluation/metrics.py` — precision@k, ndcg@k
- `intergrax/rag/vectorstore/soak/prod_slo.py` — M-RAG.64
- `intergrax/applications/_shared/reference_workflows/rag_async_ingest.py` — M-RAG.67
- `intergrax/legacy/rag_answers/__init__.py` — M-RAG.68 removal timeline
- Tests: tenant isolation, validator, metrics, ACL, plugins, async ingest reference, beta promotion
- Docs: `docs/project/architecture/RAG.md`, `docs/project/maintainers/plans/RAG.md`, `docs/audit_results/RAG.md`

## Verification

```bash
uv run pytest tests/unit/rag/ tests/unit/tools/providers/rag/ tests/unit/applications/test_rag_async_ingest_reference.py -m gate -q
```

Result: **108 passed**

## Risks and follow-ups

- Live ops soak still required before promoting `pinecone`/`milvus`/`vespa` manifests to stable (M-RAG.64 harness gate only).
- `lancedb`/`typesense` tenant contract uses adapter delegation with injected in-memory store — live backend soak remains ops responsibility.
- Tier-0 streaming ingest (proposal I) and ColBERT (proposal J) remain out of scope by design.
- `intergrax.legacy.rag_answers` scheduled for removal after 2026-12-31.
