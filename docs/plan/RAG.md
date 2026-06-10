# RAG — Implementation Plan

**Architecture (1:1):** [`architecture/RAG.md`](../architecture/RAG.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and this plan doc for the domain.

---

## Phase AUDIT-IDEAL — RAG gap register (layer 14)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.6, §7.7  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-14.1 | §14 RAG | Graph RAG production profile (shared with MEMORY) | P1 | **Done** |
| AUDIT-IDEAL-14.3 | §14 RAG | Wire `RagProfile.query_expansion` to retrieval path | P0 | **Planned** |
| AUDIT-IDEAL-14.4 | §14 RAG | Dual-index + hierarchical retriever default bootstrap | P1 | **Planned** |
| AUDIT-IDEAL-14.5 | §14 RAG | Retrieval poisoning defense on `rag.retrieve` catalog path | P1 | **Planned** |
| AUDIT-IDEAL-14.6 | §14 RAG | Large-corpus async ingest (stream / job orchestration) | P1 | **Planned** |
| AUDIT-IDEAL-14.7 | §14 RAG | OpenTelemetry spans on RAG retrieve + ingest hot path | P2 | **Planned** |

**Note:** AUDIT-IDEAL-14.2 (retrieval poisoning on product hosts) is owned by [`plan/MEMORY.md`](MEMORY.md) + UAEP security wiring — Nexus `RagStep` path.

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

**Engine audit (2026-06-10):** Canon in [`architecture/RAG.md`](../architecture/RAG.md). Maturity **L2.5 implementation / L3 control plane**. Closeout queue: [Phase M-RAG-DEPTH](#phase-m-rag-depth--production-hardening-post-audit-2026-06-10).

---

### 6.1e Harness implementation queue — RAG closeout (closed)

**Purpose:** Single ordered list for **Phase RAG** (Band 2m). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **RAG-DOC.1** | Docs | **Done** | Appendix K §K.5 + AUDIT_MAP §14 | Author map complete |
| 2 | **RAG-1** | Code | **Done** | `rag_runtime_bridge` + environment wire | `test_rag_runtime_bridge.py` |

---

## Phase RAG — RAG retrieval control plane closeout

**Status:** **Done** (2026-06-02) — **3/3** deliverables Done (RAG-DOC.* + RAG-1); gate **612 passed**

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §14; author map: **Appendix K** §K.5.

**Priority ladder:** **Band 2m** (§4.0) — closed; default queue = **§6.1** maintenance.

**Execution order:** [§6.2be](#62be-phase-rag-execution-order-band-2m--closed) · queue: [§6.1e](#61e-harness-implementation-queue--rag-closeout-closed)

### RAG — Master register

| ID | Area | Deliverable | Status | Priority | Modules | Acceptance |
|----|------|-------------|--------|----------|---------|------------|
| RAG-DOC.1 | RAG0 | **Appendix K** §K.5 + AUDIT_MAP §14 cross-ref | **Done** | High | `docs/*` | RAG bridge documented |
| RAG-1 | RAG1 | **`rag_runtime_bridge.py`** + RAG stack on `wire_application_environment` | **Done** | **Critical** | `rag_runtime_bridge.py`, `environment_wiring.py`, `runtime_config_bridge.py` | `test_rag_runtime_bridge.py` |

### RAG — Paydown log

| Date | RAG ID | Summary |
|------|--------|---------|
| 2026-06-02 | RAG-DOC.1 | Appendix K §K.5 + plan sync |
| 2026-06-02 | RAG-1 | RAG runtime bridge + environment wire; gate **600** |
| 2026-06-10 | RAG-DOC.2 | Dedicated domain pair `architecture/RAG.md` ↔ `plan/RAG.md`; migrated M-RAG canon from INTEGRATIONS |

**Phase RAG complete when:** RAG-1 + RAG-DOC.* **Done**; §6.1e queue closed. **Status: complete (2026-06-02).**

---

### 6.2be Phase RAG execution order (Band 2m — closed 2026-06-02)

**Status:** **Done** · register: [Phase RAG](#phase-rag--rag-retrieval-control-plane-closeout) · queue: [§6.1e](#61e-harness-implementation-queue--rag-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | RAG-1 | `rag_runtime_bridge` + environment wire | Critical |
| 2 | RAG-DOC.1 | Appendix K §K.5 + plan sync | Low |

---

## Phase M-RAG — RAG Engine (Tier-0)

**Canon:** [`architecture/RAG.md`](../architecture/RAG.md) · PLATFORM_FOUNDATION §5.2.2  
**Goal:** One configurable retrieval path for `rag.retrieve`, Nexus `ContextBuilder`, and ingest — no duplicate dense-only shortcuts; parsers/chunkers/rerankers selected via profile and Integration Library slugs (never hardcoded to a single vendor).

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| M-RAG.1 | `RagProfile` + env (`INTERGRAX_RAG_*`) | **Done** | `intergrax/rag/profiles/rag_profile.py` |
| M-RAG.2 | `RetrievalService` (route → retrieve → rerank) | **Done** | `intergrax/rag/retrieval/`; wired to `rag.retrieve` + Nexus |
| M-RAG.3 | Adaptive `QueryRouter` (fast / standard / deep) | **Done** | `intergrax/rag/routing/query_router.py` |
| M-RAG.4 | `IngestPipeline` + configurable chunking strategy | **Done** | `intergrax/rag/ingest/`; `rag.ingest_document` |
| M-RAG.5 | Contextual chunk enricher (optional LLM) | **Done** | `INTERGRAX_RAG_CONTEXTUAL_ENRICH`; injected `LLMAdapter` |
| M-RAG.6 | Query expansion (`deterministic` / `llm`) | **Partial** | `MultiQueryRetriever` + `query_expander.py` exist; `RagProfile.query_expansion` **not wired** to retrieval path (audit 2026-06-10) → M-RAG.23 |
| M-RAG.7 | Evaluation metrics (`recall@k`, MRR) | **Done** | `intergrax/rag/evaluation/metrics.py` |
| M-RAG.8 | `create_default_rag_stack()` bootstrap | **Done** | `intergrax/rag/bootstrap/rag_stack_bootstrap.py` |
| M-RAG.9 | Tool/Nexus wiring (`retrieval_service`, profile on `ToolWiringContext`) | **Done** | `RuntimeConfig.retrieval_service` |
| M-RAG.10 | Native sparse / BM25 in vector backends | **Done** | `LexicalHybridSupport` + `query_hybrid` on InMemory/Qdrant/Weaviate; RRF fusion |
| M-RAG.11 | RAG eval CI gate + golden datasets | **Done** | `tests/fixtures/rag_golden/`, `golden_harness.py`, `rag-guard.yml` |
| M-RAG.12 | GraphRAG (`GraphStore` contract) | **Done** (beta) | `graph/` + `graph_rag` retriever + heuristic indexer |
| M-RAG.13 | Platform agentic retrieval loop (budgeted) | **Done** | `AgenticRetrievalLoop` on deep tier + `INTERGRAX_RAG_AGENTIC_*` |
| M-RAG.14 | Qdrant native sparse vectors + RRF fusion | **Done** | `INTERGRAX_RAG_QDRANT_SPARSE`, `bm25_sparse_encoder.py` |
| M-RAG.15 | Weaviate native `query.hybrid` | **Done** | Live client + `INTERGRAX_RAG_WEAVIATE_NATIVE_HYBRID`; fallback to in-memory |
| M-RAG.16 | LLM graph indexer (optional adapter) | **Done** | `INTERGRAX_RAG_GRAPH_INDEXER_MODE=llm\|heuristic_then_llm` |
| M-RAG.17 | LLM agentic query refinement | **Done** | `INTERGRAX_RAG_AGENTIC_QUERY_MODE=llm` + injected `LLMAdapter` |
| M-RAG.18 | Neo4j GraphRAG backend | **Done** | `Neo4jRagGraphStore` + `INTERGRAX_RAG_GRAPH_STORE=neo4j` |
| M-RAG.19 | SPLADE / learned sparse encoder | **Done** | `sparse_encoder.py`; `INTERGRAX_RAG_SPARSE_ENCODER=splade` (optional `fastembed`) |
| M-RAG.20 | Weaviate prod hardening | **Done** | `schema.py` — migration, multi-tenant, metadata filters |
| M-RAG.21 | Extended golden datasets | **Done** | graph_rag, multi_hop, agentic scenarios in `retrieval_cases.json` |
| M-RAG.22 | RAG observability metrics | **Done** | `INTERGRAX_RAG_METRICS_ENABLED`, runtime plugin on `TASK_COMPLETED` |

---

## Phase M-RAG-DEPTH — Production hardening (post audit 2026-06-10)

**Source:** RAG engine depth audit vs production RAG systems · canon [`architecture/RAG.md`](../architecture/RAG.md)  
**Status:** **Planned** — runs in parallel with §6.1 gate maintenance ([`plan/PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md))  
**Policy:** One M-RAG.\* ID per PR (or one cohesive harden wave ≤3 IDs); map to AUDIT-IDEAL-14.3–14.7 where applicable

| # | ID | Deliverable | Priority | Status | Acceptance |
|---|-----|-------------|----------|--------|------------|
| 1 | M-RAG.23 | Wire `RagProfile.query_expansion` (`off` \| `deterministic` \| `llm`) to `MultiQueryRetriever` / deep-tier fusion path | **P0** | **Planned** | `test_rag_profile_query_expansion_wiring.py`; M-RAG.6 → **Done** |
| 2 | M-RAG.24 | Bootstrap `DualIndexStrategy` + `HierarchicalRetriever` when profile flag or `parent_child` + `hierarchical` retriever selected | **P1** | **Planned** | Integration test: TOC + chunk retrieve; AUDIT-IDEAL-14.4 |
| 3 | M-RAG.25 | Optional poisoning filter on `perform_rag_retrieve` behind `security_profile` | **P1** | **Planned** | Unit test mirrors `rag_step` filter; AUDIT-IDEAL-14.5 |
| 4 | M-RAG.26 | Async ingest job contract — batch/stream shards via `workflow_orchestrator` (no sync full-RAM path for huge files) | **P1** | **Planned** | Job tool or worker + ingest idempotency; AUDIT-IDEAL-14.6 |
| 5 | M-RAG.27 | OTel spans on `RetrievalService` + `IngestPipeline` stages | **P2** | **Planned** | Span names in observability gate script; AUDIT-IDEAL-14.7 |
| 6 | M-RAG.28 | Retriever fallback chain after `RetrieverEngine` retry exhaustion | **P2** | **Planned** | Degrade `fusion` → `hybrid` → `vector_similarity` with trace reason |
| 7 | M-RAG.29 | Formal `Citation` model on `RetrievalResult` + `rag.retrieve` output | **P2** | **Planned** | Gate test extends citation preservation to engine output |
| 8 | M-RAG.30 | Vector store **beta → stable** promotion — `qdrant` + `pgvector` soak gate first | **P1** | **Planned** | `test_vectorstore_real_backends.py` + ops runbook row in [`architecture/INTEGRATIONS.md`](../architecture/INTEGRATIONS.md) catalog |
| 9 | M-RAG.31 | Embedding model version reindex policy (`embedding_model_version` mismatch → warn / queue reindex) | **P2** | **Planned** | Unit test on ingest metadata + retrieve filter |

**Audit maturity target after M-RAG-DEPTH closeout:** **L3 implementation** for Tier-3 reference hosts; L4 adaptive routing deferred to AHI domain.

**Paydown log:**

| Date | ID | Summary |
|------|-----|---------|
| 2026-06-10 | AUDIT-RAG | Engine depth audit; M-RAG-DEPTH queue; M-RAG.6 corrected to **Partial** |
| 2026-06-10 | RAG-DOC.2 | Dedicated `RAG.md` domain pair; migrated from INTEGRATIONS |
