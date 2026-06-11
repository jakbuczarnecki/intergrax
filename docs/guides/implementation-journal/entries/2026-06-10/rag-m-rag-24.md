---
id: IJ-2026-06-10-003
date: 2026-06-10
tiers:
  - tier-0
  - tier-3
scope: RAG
plan_ref:
  - M-RAG.24
  - AUDIT-IDEAL-14.4
  - GAP-RAG-02
  - GAP-RAG-03
status: completed
adr: none — wires existing DualIndexStrategy + HierarchicalRetriever; no new contracts
---

# Dual-index bootstrap + hierarchical ingest routing (M-RAG.24)

## Operator request

Continue M-RAG-DEPTH Wave 2: wire `DualIndexStrategy`, second `toc_vector_store`, and `HierarchicalRetriever` into default RAG bootstrap and ingest path.

## Summary

Added `hierarchical_bootstrap` helpers and `RagProfile.hierarchical_index_enabled` / `uses_hierarchical_index()`. `create_default_rag_stack` provisions a separate TOC vector store when the profile selects hierarchical mode. `IngestPipeline` routes through `IndexingManager` + `DualIndexStrategy` to build chunk and TOC indexes. Retriever bootstrap passes `toc_vector_store` to `HierarchicalRetriever`. Fixed `HierarchicalRetriever` parent expansion to use typed `MetadataFilter` (was passing raw dict). `ParentChildChunkingStrategy` sets `SECTION` metadata for TOC derivation.

## Project impact

Book-scale corpora with `parent_child` chunking can now build a TOC index and retrieve via hierarchical expansion without Tier-3 custom wiring. Opt-in via `INTERGRAX_RAG_HIERARCHICAL_INDEX=1` or `retriever_id=hierarchical`.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/RAG.md` GAP-RAG-02, 03 closed |
| Plan | `docs/plan/RAG.md` M-RAG.24 **Done**, AUDIT-IDEAL-14.4 **Done** |

## Changed artifacts

- `intergrax/rag/bootstrap/hierarchical_bootstrap.py` — TOC store factory + profile gate
- `intergrax/rag/profiles/rag_profile.py` — `hierarchical_index_enabled`, `uses_hierarchical_index()`
- `intergrax/rag/bootstrap/rag_stack_bootstrap.py` — `toc_vectorstore_manager` on `RagStack`
- `intergrax/rag/retrievers/bootstrap/retriever_bootstrap.py` — `toc_vector_store` propagation
- `intergrax/rag/ingest/ingest_pipeline.py` — `DualIndexStrategy` via `IndexingManager`
- `intergrax/rag/indexing/strategies/dual_index_strategy.py` — TOC metadata + batch insert fix
- `intergrax/rag/retrievers/providers/hierarchical_retriever.py` — `MetadataFilter` on parent expansion
- `intergrax/tools/registry/wiring.py`, `ingest_service.py`, Tier-3 wiring — TOC store propagation
- `tests/unit/rag/ingest/test_hierarchical_dual_index_wiring.py` — 3 unit tests

## Verification

```bash
uv run pytest tests/unit/rag/ingest/test_hierarchical_dual_index_wiring.py tests/unit/rag/ -m gate -q
uv run pytest -m gate -q
python scripts/check_harness_no_getattr.py
uv run python scripts/check_observability_gates.py
uv run python scripts/check_docs_domain_pairs.py
```

Result: 3 hierarchical tests passed; full gate green; harness scripts OK.

## Risks and follow-ups

- Multi-GB corpora still need async ingest (M-RAG.26) — sync path loads full document into RAM.
- Integration test with real vector backend (qdrant/pgvector) for TOC + chunk retrieve deferred to Wave 2 soak work.
- Next Wave 2 item: **M-RAG.26** (async ingest job contract).
