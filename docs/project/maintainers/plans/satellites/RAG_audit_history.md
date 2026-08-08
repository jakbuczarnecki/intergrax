# RAG — audit history + LC closeout

**Parent hub:** [`RAG.md`](../RAG.md)

## Phase RAG — RAG retrieval control plane closeout

**Status:** **Done** (2026-06-02) — **3/3** deliverables Done (RAG-DOC.* + RAG-1); gate **612 passed**

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §14; author map: **Appendix K** §K.5.

**Priority ladder:** **Band 2m** (§4.0) — closed; default queue = **§6.1** maintenance + **M-RAG-DEPTH**.

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
| 2026-06-10 | RAG-DOC.3 | Full engine depth audit register (GAP-RAG-01…23); M-RAG-DEPTH waves 1–3; traceability matrix |
| 2026-06-10 | RAG-DOC.4 | Code-verified audit sync; GAP-RAG-07 manifest correction; full task register |
| 2026-06-10 | M-RAG.23 | Wire `query_expansion` to bootstrap + deep-tier `effective_retriever`; closes GAP-RAG-01/17/23, AUDIT-IDEAL-14.3, M-RAG.6 |
| 2026-06-10 | M-RAG.25 | Catalog poisoning filter on `perform_rag_retrieve`; closes GAP-RAG-04, AUDIT-IDEAL-14.5 |

**Phase RAG complete when:** RAG-1 + RAG-DOC.* **Done**; §6.1e queue closed. **Status: complete (2026-06-02).**  
**Phase M-RAG-DEPTH:** **Complete** (2026-06-10) — M-RAG.23 … M-RAG.37 **Done**.

**Phase M-RAG-BACKLOG:** **Complete** (2026-06-13) — M-RAG.55–M-RAG.57 **Done**; M-RAG.49–M-RAG.51 **Done**; M-RAG.58 **Frozen** (GAP-RAG-15).

---

## Phase M-RAG — RAG Engine (Tier-0)

**Canon:** [`architecture/RAG.md`](../architecture/RAG.md) · PLATFORM_FOUNDATION §5.2.2  
**Goal:** One configurable retrieval path for `rag.retrieve`, Nexus `ContextBuilder`, and ingest — no duplicate dense-only shortcuts; parsers/chunkers/rerankers selected via profile and Integration Library slugs (never hardcoded to a single vendor).

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| M-RAG.1 | `RagProfile` + env (`INTERGRAX_RAG_*`) | **Done** | `intergrax/rag/profiles/rag_profile.py` |
| M-RAG.2 | `RetrievalService` (route → retrieve → rerank) | **Done** | `intergrax/rag/retrieval/`; wired to `rag.retrieve` + Nexus |
| M-RAG.3 | Adaptive `QueryRouter` (fast / standard / deep) | **Done** | `intergrax/rag/routing/query_router.py` — heuristic only; LLM tier: M-RAG.32 |
| M-RAG.4 | `IngestPipeline` + configurable chunking strategy | **Done** | `intergrax/rag/ingest/`; `rag.ingest_document` |
| M-RAG.5 | Contextual chunk enricher (optional LLM) | **Done** | `INTERGRAX_RAG_CONTEXTUAL_ENRICH`; injected `LLMAdapter` |
| M-RAG.6 | Query expansion (`deterministic` / `llm`) | **Done** | `MultiQueryRetriever` + `query_expander_from_profile()` wired via bootstrap; deep tier → `multiquery` when `query_expansion != off` (M-RAG.23) |
| M-RAG.7 | Evaluation metrics (`recall@k`, MRR) | **Done** | `intergrax/rag/evaluation/metrics.py` |
| M-RAG.8 | `create_default_rag_stack()` bootstrap | **Done** | `intergrax/rag/bootstrap/rag_stack_bootstrap.py` |
| M-RAG.9 | Tool/Nexus wiring (`retrieval_service`, profile on `ToolWiringContext`) | **Done** | `RuntimeConfig.retrieval_service` |
| M-RAG.10 | Native sparse / BM25 in vector backends | **Done** | `LexicalHybridSupport` + `query_hybrid` on InMemory/Qdrant/Weaviate; RRF fusion |
| M-RAG.11 | RAG eval CI gate + golden datasets | **Done** | `tests/fixtures/rag_golden/`, `golden_harness.py`, `rag-guard.yml` |
| M-RAG.12 | GraphRAG (`GraphStore` contract) | **Done** (stable) | `graph/` + `graph_rag` retriever **stable** (M-RAG.42); prod contract: M-RAG.33 |
| M-RAG.13 | Platform agentic retrieval loop (budgeted) | **Done** | `AgenticRetrievalLoop` + M-RAG.34 iteration schedule / latency budget trace |
| M-RAG.14 | Qdrant native sparse vectors + RRF fusion | **Done** | `INTERGRAX_RAG_QDRANT_SPARSE`, `bm25_sparse_encoder.py` |
| M-RAG.15 | Weaviate native `query.hybrid` | **Done** | Live client + `INTERGRAX_RAG_WEAVIATE_NATIVE_HYBRID`; fallback to in-memory |
| M-RAG.16 | LLM graph indexer (optional adapter) | **Done** | `INTERGRAX_RAG_GRAPH_INDEXER_MODE=llm\|heuristic_then_llm` |
| M-RAG.17 | LLM agentic query refinement | **Done** | `INTERGRAX_RAG_AGENTIC_QUERY_MODE=llm` + injected `LLMAdapter` |
| M-RAG.18 | Neo4j GraphRAG backend | **Done** | `Neo4jRagGraphStore` + `INTERGRAX_RAG_GRAPH_STORE=neo4j` |
| M-RAG.19 | SPLADE / learned sparse encoder | **Done** | `sparse_encoder.py`; `INTERGRAX_RAG_SPARSE_ENCODER=splade` (optional `fastembed`) |
| M-RAG.20 | Weaviate prod hardening | **Done** | `schema.py` — migration, multi-tenant, metadata filters |
| M-RAG.21 | Extended golden datasets | **Done** | graph_rag, multi_hop, agentic scenarios in `retrieval_cases.json` |
| M-RAG.22 | RAG observability metrics | **Done** | `INTERGRAX_RAG_METRICS_ENABLED`, runtime plugin; OTel: M-RAG.27 |

---

## Phase M-RAG-DEPTH — Production hardening (post audit 2026-06-10)

**Source:** Full engine depth audit · canon [`architecture/RAG.md`](../architecture/RAG.md) §Engine depth audit register  
**Status:** **Done** (2026-06-10) — runs in parallel with §6.1 gate maintenance ([`plan/PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md))  
**Policy:** See [Step-by-step rollout](#step-by-step-rollout--phase-m-rag-depth) waves 1–3

| # | ID | Deliverable | Priority | Status | GAP-RAG | Acceptance |
|---|-----|-------------|----------|--------|---------|------------|
| 1 | M-RAG.23 | Wire `RagProfile.query_expansion` (`off` \| `deterministic` \| `llm`) to `MultiQueryRetriever` / deep-tier path; close M-RAG.6 | **P0** | **Done** | 01, 17, 23 | `test_rag_profile_query_expansion_wiring.py`; AUDIT-IDEAL-14.3 |
| 2 | M-RAG.24 | Bootstrap `DualIndexStrategy` + `HierarchicalRetriever` + ingest routing when profile selects hierarchical | **P1** | **Done** | 02, 03 | `test_hierarchical_dual_index_wiring.py`; AUDIT-IDEAL-14.4 |
| 3 | M-RAG.25 | Optional poisoning filter on `perform_rag_retrieve` behind `security_profile` | **P1** | **Done** | 04 | Unit test mirrors `rag_step` filter; AUDIT-IDEAL-14.5 |
| 4 | M-RAG.26 | Async ingest job contract — batch/stream shards via `workflow_orchestrator` | **P1** | **Done** | 05, 06 | `rag.schedule_ingest_job` + sync size guard; AUDIT-IDEAL-14.6 |
| 5 | M-RAG.27 | OTel spans on `RetrievalService` + `IngestPipeline`; observability gate script | **P2** | **Done** | 08, 09 | `rag_spans.py` + `check_rag_otel_span_registry.py`; AUDIT-IDEAL-14.7 |
| 6 | M-RAG.28 | Retriever fallback chain; structured errors; retry alignment; optional circuit breaker | **P2** | **Done** | 10, 11, 12 | `test_retriever_engine_resilience.py`; trace `fallback_applied` |
| 7 | M-RAG.29 | Formal `Citation` model on `RetrievalResult` + `rag.retrieve` output | **P2** | **Done** | 13 | `test_rag_citation_engine_gate.py` |
| 8 | M-RAG.30 | Vector-store prod SLO — soak gate for stable slugs; promote `pinecone`/`milvus`/`vespa` from beta when soak passes | **P1** | **Done** | 07 | `test_vectorstore_prod_slo_soak.py` + integration soak; INTEGRATIONS runbook |
| 9 | M-RAG.31 | Embedding model version reindex policy (mismatch → warn / queue reindex) | **P2** | **Done** | 14 | `test_embedding_version_policy.py` + ingest/retrieve gate tests |
| 10 | M-RAG.32 | Optional LLM `QueryRouter` tier classifier (`llm_route_enabled`, default off) | **P2** | **Done** | 16 | `test_query_router_llm_tier.py` |
| 11 | M-RAG.33 | GraphRAG Tier-3 prod profile contract (neo4j required; harness preset documented) | **P1** | **Done** | 18 | `test_production_graph_rag_profile.py` + `test_graph_rag_neo4j_prod_contract.py` |
| 12 | M-RAG.34 | Agentic loop — per-iteration retriever override + cost budget trace fields | **P2** | **Done** | 19 | `test_agentic_loop_iteration_trace.py` |
| 13 | M-RAG.35 | Cross-backend tenant isolation contract tests | **P1** | **Done** | 20 | `tenant_isolation_contract.py` + gate tests per backend |
| 14 | M-RAG.36 | RAG load/soak gate (concurrent retrieve SLO) | **P2** | **Done** | 21 | `test_rag_load_soak_gate.py` + `rag-guard.yml` `-m gate` |
| 15 | M-RAG.37 | Semantic chunking ingest size guard + clear failure reason | **P2** | **Done** | 22 | `test_semantic_chunking_size_guard.py` |

**Audit maturity target after M-RAG-DEPTH closeout:** **L3 implementation** for Tier-3 reference hosts; L4 adaptive routing deferred to AHI domain (GAP-RAG-15).

**Paydown log:**

| Date | ID | Summary |
|------|-----|---------|
| 2026-06-10 | AUDIT-RAG | Engine depth audit; M-RAG-DEPTH queue M-RAG.23–31 |
| 2026-06-10 | RAG-DOC.2 | Dedicated `RAG.md` domain pair; migrated from INTEGRATIONS |
| 2026-06-10 | RAG-DOC.3 | Full GAP-RAG register (23 rows); M-RAG.32–37; wave rollout; traceability matrix |
| 2026-06-10 | RAG-DOC.4 | Code-verified audit sync: GAP-RAG-07 manifest correction; §Audit verification evidence; full task register |
| 2026-06-10 | M-RAG.23 | Wire `query_expansion` to bootstrap + deep-tier retrieval; `test_rag_profile_query_expansion_wiring.py` |
| 2026-06-10 | M-RAG.25 | Catalog poisoning filter on `perform_rag_retrieve`; `security_profile` on `ToolWiringContext` |
| 2026-06-10 | M-RAG.24 | `toc_vector_store` bootstrap + `DualIndexStrategy` ingest routing + `HierarchicalRetriever` wiring; `test_hierarchical_dual_index_wiring.py` |
| 2026-06-10 | M-RAG.26 | `rag.schedule_ingest_job` + sync ingest size guard (`INTERGRAX_RAG_SYNC_INGEST_MAX_BYTES`); idempotent orchestrator trigger |
| 2026-06-10 | M-RAG.30 | Vector-store prod SLO soak contract + gate tests; INTEGRATIONS runbook; stable vs beta promotion policy |
| 2026-06-10 | M-RAG.33 | `production_graph_rag_profile()` + product host wiring; harness `production_rag_profile()` documented in-memory only |
| 2026-06-10 | M-RAG.35 | `tenant_isolation_contract.py`; gate tests for inmemory/pgvector/weaviate/qdrant |
| 2026-06-10 | M-RAG.27 | `rag_spans.py` OTel on retrieve + ingest; `check_rag_otel_span_registry.py` in observability gates |
| 2026-06-10 | M-RAG.28 | `RetrievalError` taxonomy; retriever fallback chain; retry=2; optional vector circuit breaker |
| 2026-06-10 | M-RAG.29 | `Citation` model on `RetrievalResult` + `RagCitationResult` on `rag.retrieve` |
| 2026-06-10 | M-RAG.31 | `embedding_version_policy.py` — ingest warn, retrieve filter, reindex hook |
| 2026-06-10 | M-RAG.32 | `llm_tier_classifier.py` + `RagProfile.llm_route_enabled`; trace `route_classifier` |
| 2026-06-10 | M-RAG.34 | `agentic_policy.py` — per-iteration retriever schedule + latency budget trace on `AgenticRetrievalLoop` |
| 2026-06-10 | M-RAG.37 | `semantic_chunking_allowed()` — reject oversized docs before semantic O(n) embed |
| 2026-06-10 | M-RAG.36 | `load_soak.py` concurrent retrieve SLO; `rag-guard.yml` gate marker |
| 2026-06-12 | RAG-DOC.5 | GraphRAG architecture audit; GAP-RAG-24–36; Phase M-RAG-GRAPH waves G1–G4; architecture §GraphRAG architecture |
| 2026-06-12 | M-RAG.38–41 | Graph store backend registry; memgraph/falkordb adapters; delete/purge lifecycle sync; graph tenant isolation contract |
| 2026-06-12 | M-RAG.42–52 | GraphRAG retrieval hardening, prod slug list, golden scenarios, maintenance job, indexer plugins, community_report mode |
| 2026-06-13 | RAG-DOC.6 | Layer Completion audit; GAP-RAG-29/30 partial reclassification; G5 sprint plan |
| 2026-06-13 | M-RAG.53–54 | GraphRAG usage hardening — 3-channel fusion + structured provenance on RetrievalTrace |

---

## Layer Completion — sprint execution plan (2026-06-12)

Execute after documentation sync. One sprint = one commit unless operator splits PRs.

### Sprint G1 — Backend registry and graph lifecycle (M-RAG.38–M-RAG.41)

| Item | Scope | Files |
|------|-------|-------|
| M-RAG.38 | `RagGraphStoreBackend` registry; refactor bootstrap | `graph/bootstrap/backend_registry.py`, `graph/bootstrap/graph_store_bootstrap.py`, `tests/unit/rag/graph/test_rag_graph_store_backend_registry.py` |
| M-RAG.39 | Memgraph + FalkorDB Cypher adapters | `graph/providers/cypher_rag_graph_store.py`, `graph/bootstrap/graph_store_bootstrap.py`, `tests/unit/rag/graph/test_graph_rag_memgraph_adapter.py`, `test_graph_rag_falkordb_adapter.py` |
| M-RAG.40 | Delete/purge graph lifecycle sync | `graph/contracts/graph_store.py`, providers, `tools/providers/rag/lifecycle_service.py`, `index_lifecycle_service.py`, `tools/registry/wiring.py`, `applications/_shared/environment_wiring.py`, `tests/unit/rag/graph/test_graph_lifecycle_delete_sync.py` |
| M-RAG.41 | Graph tenant isolation contract | `graph/tenant/graph_isolation_contract.py`, graph providers (tenant scope), `tests/unit/rag/graph/test_graph_tenant_isolation.py` |

**DoD:** registry gate green; delete/purge removes graph artifacts; tenant mismatch returns empty on graph path.

### Sprint G2 — Retrieval hardening (M-RAG.42–M-RAG.44, M-RAG.48, M-RAG.52)

| Item | Scope | Files |
|------|-------|-------|
| M-RAG.42 | Harden `GraphRagRetriever`; promote stable | `retrievers/providers/graph_rag_retriever.py`, `tests/unit/rag/graph/test_graph_rag_retriever_hardening.py` |
| M-RAG.43 | Hybrid channel fusion | `retrievers/providers/graph_rag_retriever.py`, `runtime/architecture/hybrid_retrieval.py`, `tests/unit/rag/graph/test_hybrid_retrieval_graph_channel.py` |
| M-RAG.44 | Graph provenance on `RetrievalTrace` | `retrieval/retrieval_result.py`, `graph_rag_retriever.py`, `tests/unit/rag/graph/test_graph_provenance_retrieval_trace.py` |
| M-RAG.48 | Approved prod graph_store slugs | `profiles/rag_profile.py`, `applications/_shared/rag_runtime_bridge.py`, `tests/unit/rag/profiles/test_production_graph_rag_profile.py` |
| M-RAG.52 | Extended golden harness scenarios | `tests/fixtures/rag_golden/retrieval_cases.json`, `evaluation/golden_harness.py` |

**DoD:** graph_rag stable; `channel_contributions` on trace; prod validation accepts soaked Bolt backends; golden gate covers lifecycle + isolation.

### Sprint G3 — Maintenance and indexer plugins (M-RAG.45–M-RAG.47)

| Item | Scope | Files |
|------|-------|-------|
| M-RAG.45 | `rag.schedule_graph_maintenance_job` | `tools/providers/rag/graph_maintenance_*.py`, `bundle.py`, `profiles/rag_profile.py`, `tests/unit/tools/providers/rag/test_graph_maintenance_job.py` |
| M-RAG.46 | `GraphIndexer` plugin registry | `graph/indexer/plugin_registry.py`, `graph_indexer_factory.py`, `docs/project/technical/guides/EXTENSION_AUTHOR_GUIDE.md` |
| M-RAG.47 | Optional `community_report` indexer | `graph/indexer/community_report_graph_indexer.py`, `rag_profile.py`, `tests/unit/rag/graph/test_community_report_graph_indexer.py` |

**DoD:** maintenance job idempotent; third-party indexer registers via plugin; community mode opt-in only.

### Sprint G5 — GraphRAG usage hardening (M-RAG.53–M-RAG.54)

| Item | Scope | Files |
|------|-------|-------|
| M-RAG.53 | Full 3-channel fusion (vector + keyword + graph) | `retrieval/graph_channel_fusion.py`, `retrievers/providers/graph_rag_retriever.py`, `tests/unit/rag/graph/test_hybrid_retrieval_graph_channel.py` |
| M-RAG.54 | Structured `GraphTraceFieldBundle` on `RetrievalTrace` | `retrieval/graph_provenance_builder.py`, `retrieval/retrieval_result.py`, `retrievers/engine/retriever_execution.py`, `retrieval/retrieval_service.py`, `tests/unit/rag/graph/test_graph_provenance_retrieval_trace.py` |

**DoD:** `channel_contributions` includes `vector`, `keyword`, `graph`; `RetrievalTrace.graph_provenance_records` populated with structured path records.

### Sprint G4 — Additional integrations (M-RAG.49–M-RAG.51) — P3

Requires Integration catalog slugs (Neptune, OrientDB, ArangoDB) per [`plan/INTEGRATIONS.md`](INTEGRATIONS.md) H-INT-GRAPH.

### Phase M-RAG-BACKLOG — P2 hardening (2026-06-13)

| Sprint | ID | Priority | Scope | Files |
|--------|-----|----------|-------|-------|
| P2.1 | **M-RAG.55** | P2 | Graph store soak gate; promote `falkordb` to `APPROVED_PRODUCTION_GRAPH_STORE_SLUGS` after harness soak | 29 | **Done** |
| P2.2 | **M-RAG.56** | P2 | Beta vector slug soak harness — inject in-memory store through beta adapter factories | 29 | **Done** |
| P2.3 | **M-RAG.57** | P2 | RAG metrics default-on when OTel spine enabled (align with `INTERGRAX_RAG_OTEL_SPANS_ENABLED`) | 30 | **Done** |

### Phase M-RAG-BACKLOG — P3 vendor graph_store (2026-06-13)

| Sprint | ID | Priority | Scope | Depends |
|--------|-----|----------|-------|---------|
| P3.1 | **H-INT-GRAPH-1** + **M-RAG.49** | P3 | `neptune` integration + RAG `CypherRagGraphStore` adapter | **Done** |
| P3.2 | **H-INT-GRAPH-2** + **M-RAG.50** | P3 | `orientdb` integration + RAG adapter | **Done** |
| P3.3 | **H-INT-GRAPH-3** + **M-RAG.51** | P3 | `arangodb` integration + RAG AQL adapter | **Done** |

### Phase M-RAG-BACKLOG — P4 frozen boundary (2026-06-13)

| Sprint | ID | Priority | Scope | Status |
|--------|-----|----------|-------|--------|
| P4.1 | **M-RAG.58** | P4 | GAP-RAG-15 **Frozen** — autonomous retriever/chunker selection owned by [`plan/ADAPTIVE_HARNESS_INTELLIGENCE.md`](ADAPTIVE_HARNESS_INTELLIGENCE.md) | **Frozen** |

**Layer status:** RAG domain **Frozen** for harness iteration — no open P0–P3 items; future work = ops soak promotion (beta→stable manifests) and AHI adaptive routing.

### Phase M-RAG-CONVERGE — doc + diagnostics closeout (2026-06-13)

Iteration II on **Frozen** layer — sync stale architecture sections and close minor harness gaps.

| Sprint | ID | Priority | Scope | Status |
|--------|-----|----------|-------|--------|
| C1 | **M-RAG.59** | P2 | Architecture/plan audit evidence convergence (readiness verdict, maturity L3, GraphRAG matrix, duplicate GAP rows) | **Done** |
| C2 | **M-RAG.60** | P2 | Export `channel_contributions` + `graph_provenance_records` on `rag.retrieve` diagnostics | **Done** |
| C3 | **M-RAG.61** | P3 | Align `STABLE_PROD_SLO_SLUGS` with stable manifests (`lancedb`, `typesense`) | **Done** |

**DoD:** architecture §Production readiness / §Maturity / §GraphRAG / §Audit evidence aligned with code; tool diagnostics expose graph trace fields; stable soak tuple matches integration manifests.

---

## Phase M-RAG-ITERATION-III — Layer Completion (2026-06-17)

**Source:** Operator accepted strategic proposals A–H and L from Layer Completion audit (2026-06-17). Proposals I/J/K remain **Rejected** (Tier-0 stream ingest, ColBERT, AHI auto-selection).

| Sprint | ID | Priority | Deliverable | Status |
|--------|-----|----------|-------------|--------|
| S1 | **M-RAG.59b** | P2 | Documentation Convergence III + audit prompt regen | **Done** |
| S2 | **M-RAG.62** | P1 | Tenant isolation contract — `chroma`, `lancedb`, `typesense` | **Done** |
| S3 | **M-RAG.63** | P1 | `validate_rag_profile_wiring` / `assert_rag_profile_wiring` at bootstrap | **Done** |
| S4 | **M-RAG.67** | P2 | Reference async ingest shard planner (`reference_workflows/rag_async_ingest.py`) | **Done** |
| S5 | **M-RAG.68a** | P3 | Evaluation metrics — `precision@k`, `ndcg@k` | **Done** |
| S6 | **M-RAG.64** | P2 | `evaluate_beta_promotion_readiness()` harness gate | **Done** |
| S7 | **M-RAG.66** | P3 | `register_chunking_strategy_plugin()` registry | **Done** |
| S8 | **M-RAG.65** | P2 | `CollectionAccessPolicy` on `VectorstoreManager` | **Done** |
| S9 | **M-RAG.68** | P3 | Legacy `rag_answers` removal timeline (2026-12-31) | **Done** |

**Layer status:** RAG domain **Architecturally Mature** (2026-06-17) — zero open P0/P1 harness defects; ops beta→stable manifest promotion and AHI adaptive routing remain backlog.

**Verification:**

```bash
uv run pytest tests/unit/rag/ tests/unit/tools/providers/rag/ tests/unit/applications/test_rag_async_ingest_reference.py -m gate -q
```

---

## Phase RAG-LC — Full Harness Layer Completion closeout (2026-06-17)

**Status:** **Done** (2026-06-17) — re-validates M-RAG-ITERATION-III + M-RAG-CONVERGE; no open P0/P1  
**Prerequisites:** M-RAG-GRAPH **Done** · M-RAG.62–M-RAG.68 **Done**  
**Goal:** Formal Full Harness LC closeout — gate verification, journal, audit prompt sync  
**ADR:** **No ADR needed**

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| RAG-LC-S1 | **Re-audit** — GAP-RAG register + frozen verdict | **Done** | High | No P0/P1 |
| RAG-LC-S2 | **Plan/architecture sync** — Full Harness LC note | **Done** | High | Domain pair consistent |
| RAG-LC-S3 | **Gate verification** | **Done** | High | 108 gate tests · 2 CI scripts |
| RAG-LC-S4 | **Journal + progress tracker** | **Done** | High | `layer_completion_progress.json` mature |

**Deferred P2–P4:** beta→stable manifest promotion · M-RAG.58 AHI adaptive routing (Frozen) · ops soak gates

### 6.1av Harness implementation queue — RAG audit maintenance (planned)

**Source:** Layer 12 audit (2026-06-18) — `RAG` layer 14 · [`../audit_results/2026-06-18/RAG.md`](../audit_results/2026-06-18/RAG.md)  
**Priority ladder:** **Band 1** (§6.1) — ops honesty + prompt sync; **one ID per PR**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **RAG-MAINT-01** | CI/Ops | P2 | **Done** | Beta→stable RagProfile/manifest promotion criteria + gate | `scripts/maintenance/check_rag_maturity_labels.py` in rag-guard; STABLE slugs ↔ manifest |
| 2 | **RAG-MAINT-02** | CI | P3 | **Done** | Production SLO soak depth — nightly workflow extension beyond M-RAG.36 gate marker | `scripts/release/rag_load_soak_report.py` → `build/rag/load_soak_report.json` nightly |
| 3 | **RAG-MAINT-03** | Docs | P3 | **Done** | Audit prompt sync — GAP-RAG register **Closed** in known gaps | `docs/project/maintainers/audit/RAG.md` regenerated from LC closeout |
| 4 | **RAG-MAINT-04** | Cross-ref | P4 | **Done** | M-RAG.58 AHI adaptive routing — document **Frozen** owner (AHI domain) | Owner: [`AHI-MAINT-04`](ADAPTIVE_HARNESS_INTELLIGENCE.md#61av-harness-implementation-queue--adaptive-harness-intelligence-audit-maintenance-planned) |

**Suggested PR order:** RAG-MAINT-03 → RAG-MAINT-01 → RAG-MAINT-02 → RAG-MAINT-04.

**Environment note:** Windows `pytest tests/unit/rag/` teardown crash (`-1073741819`) — track under DX if reproducible; not blocking L3 verdict.

**Cross-domain:** INT-MAINT-01 — integration slug maturity · M-RAG.58 — [`plan/ADAPTIVE_HARNESS_INTELLIGENCE.md`](ADAPTIVE_HARNESS_INTELLIGENCE.md).

---

### Phase RAG-MAINT-vllm — vLLM embedding provider (2026-06-19)

**Source:** vLLM platform integration — RAG embeddings via OpenAI-compatible `/v1/embeddings`.  
**Goal:** `VllmEmbeddingProvider` registered in default bootstrap; optional Docker `vllm-embed` on host **8101**.

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **RAG-MAINT-vllm-1** | Code | P2 | **Done** | `vllm_embedding_provider.py` + default registry | `provider_id=vllm` in `EmbeddingProviderRegistry` |
| 2 | **RAG-MAINT-vllm-2** | Infra | P2 | **Done** | `infra/docker/vllm-embed` + integration profile `vllm` service | `INTERGRAX_DEFAULT_VLLM_EMBED_BASE_URL=http://127.0.0.1:8101/v1` |
| 3 | **RAG-MAINT-vllm-3** | Tests | P2 | **Done** | Unit mocks + optional integration pipeline test | `tests/unit/rag/embedding/test_vllm_embedding_provider.py` green |

**ADR:** no ADR needed — mirrors `OpenAIEmbeddingProvider` against self-hosted vLLM embed server.

---

### Phase RAG-MAINT-llama-cpp — llama.cpp embedding provider (2026-06-19)

**Source:** llama.cpp platform integration — RAG embeddings via OpenAI-compatible `/v1/embeddings`.  
**Goal:** `LlamaCppEmbeddingProvider` registered in default bootstrap; optional Docker `llama-cpp-embed` on host **8103**.

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **RAG-MAINT-llama-cpp-1** | Code | P2 | **Done** | `llama_cpp_embedding_provider.py` + default registry | `provider_id=llama_cpp` in `EmbeddingProviderRegistry` |
| 2 | **RAG-MAINT-llama-cpp-2** | Infra | P2 | **Done** | `infra/docker/llama-cpp-embed` + integration profile `llama-cpp` service | `INTERGRAX_DEFAULT_LLAMA_CPP_EMBED_BASE_URL=http://127.0.0.1:8103/v1` |
| 3 | **RAG-MAINT-llama-cpp-3** | Tests | P2 | **Done** | Unit mocks + E2E embed pipeline in `tests/e2e/llama_cpp/` (`no_ci`) | `test_llama_cpp_embedding_pipeline_documents` with live embed server |

**Phase status:** **Done** (2026-06-19) — 3/3 Done.

**ADR:** no ADR needed — mirrors `VllmEmbeddingProvider` against self-hosted llama.cpp embed server.

---

*End of RAG Implementation Plan.*
