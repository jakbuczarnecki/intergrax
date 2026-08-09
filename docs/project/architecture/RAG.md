# RAG and Retrieval

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](intergrax_runtime_architecture.md)
**Plan (1:1):** [`plan/RAG.md`](../maintainers/plans/RAG.md)
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)
**Audit layer:** 14 (RAG and Retrieval)  
**Audit instruction:** [`audit/RAG.md`](../maintainers/audit/RAG.md)
**Related:** [`architecture/INTEGRATIONS.md`](INTEGRATIONS.md) (vector_store, document_parser, rerank_provider slugs) · [`architecture/MEMORY.md`](MEMORY.md) (Knowledge store vs LTM) · [`guides/AGENT_CREATION_GUIDE.md`](../technical/guides/AGENT_CREATION_GUIDE.md) Appendix K §K.5
**Implementation:** `intergrax/rag`
**Last architecture audit:** 2026-06-17 — **Full Harness LC** (re-validates M-RAG-ITERATION-III); **Architecturally Mature** · 2026-06-13 (iteration II Frozen) · 2026-06-12 (GraphRAG G1–G5)

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (RAG canon).

- **Implement / audit default:** retrieval pipeline + index lifecycle (hub). Pipelines detail: [`satellites/RAG_pipelines_detail.md`](satellites/RAG_pipelines_detail.md).
- **Use** table of contents below — `Read` with offset/limit per §.
- **Plan hub:** [`plan/RAG.md`](../maintainers/plans/RAG.md) (scoped §6 only).
- **Audit slice:** [`guides/audit_slices/RAG.md`](../technical/guides/audit_slices/RAG.md).
- **Max reads:** at most **one** file >5k tokens per session unless RESUME cites more.

---


## Architecture satellites (read on demand)

Large § blocks moved out of the architecture hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited §.

| Satellite | Contents |
|-----------|----------|
| [`satellites/RAG_pipelines_detail.md`](satellites/RAG_pipelines_detail.md) | pipelines detail |
| [`../capabilities/architecture/satellites/LANGCHAIN_INDEPENDENCE_native_document_contract.md`](../capabilities/architecture/satellites/LANGCHAIN_INDEPENDENCE_native_document_contract.md) | **Native Knowledge Document Contract** (`KnowledgeDocument`) — LCI-1A source of truth |

## RAG-CLOSE-5 production qualification (2026-08-09)

**Final status:** **PRODUCTION_QUALIFIED_WITH_LIMITATIONS**

The canonical native path is qualified for offline production-gate behavior:
recursive native chunking, deterministic file-ingest E2E retrieval, dense
vector similarity, built-in in-memory hybrid retrieval, graph indexing and
retrieval in the harness graph store, graph unlink/delete lifecycle, tenant
isolation, same-filename identity isolation, and opt-in external chunker,
retriever, and reranker runtime wiring. The plugin closeout gate also proves
an entry-point chunker can run through `IngestPipeline`, embedding, vector
storage, and `RetrievalService` without changing RAG core.

Explicit limitations:

- hierarchical/parent-child mode is implemented and profile-selectable, but
  remains **IMPLEMENTED_BUT_UNQUALIFIED** for a complete final-content and
  parent-linkage production gate;
- repeated ingest has deterministic/provider upsert behavior, but there is no
  separate source-scoped reingest contract and gate;
- tenant filtering is qualified; namespace/workspace propagation and provider
  predicates are covered, but independent negative leakage gates are not;
- the Qdrant provider contract and ID normalization are covered by offline
  tests, and the live tenant-isolation probe passed; the broader backend
  lifecycle suite is **TEST_ASSUMPTION_DEFECT** because its integration
  wrapper calls methods outside the provider contract; live production graph
  services are likewise not claimed here;
- LangChain recursive chunking remains optional and is not required by core;
  the optional splitter dependency is present in this environment but is not
  used by the canonical native gate.

Evidence commands:
`uv run pytest tests/unit/rag/document_splitters/test_native_strategies.py -q -k recursive`,
`uv run pytest tests/e2e/rag/test_native_rag_retrieval_qualification.py -q`,
`uv run pytest tests/unit/rag/evaluation/test_golden_retrieval_gate.py tests/unit/rag/graph/test_hybrid_retrieval_graph_channel.py tests/unit/rag/graph/test_graph_rag_retriever_hardening.py tests/unit/rag/graph/test_graph_provenance_retrieval_trace.py tests/unit/rag/profiles/test_production_graph_rag_profile.py -q`,
and `uv run pytest tests/unit/rag/test_rag_plugin_discovery.py -q`.

## Native Knowledge Document Contract

RAG is the **functional owner** of the platform knowledge document ABI. The canonical type is **`KnowledgeDocument`**, implemented in neutral Tier-0 module `intergrax/knowledge/contracts/document.py` (LCI-1B), public import: `from intergrax.knowledge.contracts import KnowledgeDocument`.

**Source of truth:** [`../capabilities/architecture/satellites/LANGCHAIN_INDEPENDENCE_native_document_contract.md`](../capabilities/architecture/satellites/LANGCHAIN_INDEPENDENCE_native_document_contract.md)

**Policy:** No new local document models in RAG contracts, memory, modality, or integrations — all shared pipelines consume `KnowledgeDocument`. Replaces `langchain_core.documents.Document` in public contracts per LangChain Independence inventory.

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.
## Maturity score (audit map L0–L4)

| Dimension | Score | Notes |
|-----------|-------|-------|
| Control-plane architecture (single path, typed contracts) | **L3** | `RetrievalService`, `RagProfile`, runtime bridges, 12 `rag.*` catalog tools |
| Retrieval mode breadth | **L3** | hybrid, fusion, agentic, graph_rag **stable**; hierarchical + dual-index wired via profile (M-RAG.24) |
| GraphRAG platform (build / maintain / retrieve) | **L3** | Backend registry, lifecycle sync, tenant isolation, stable retriever, maintenance jobs (M-RAG-GRAPH G1–G3) |
| Strategy selection (autonomous) | **L1.5** | Tier/cost routing only; MIME/size/retriever auto-pick deferred to Tier-3 + AHI |
| Ingest — short / medium documents | **L3** | Parser catalog, 5 chunking strategies, optional contextual enrich |
| Ingest — very large corpora | **L2–L2.5** | Sync path size-guarded; `rag.schedule_ingest_job` triggers orchestrator with idempotent contract (M-RAG.26); shard/stream execution in workflow worker |
| Resilience (retry, fallback, circuit breaker) | **L2.5** | Retriever retry=2 aligned with embedding; fallback `fusion`→`hybrid`→`vector_similarity`; optional vector circuit breaker |
| Observability | **L3** | `RetrievalTrace`, parser trace, OTel spans default-on; metrics default-on when OTel spine enabled (M-RAG.57) |
| Security (poisoning) | **L3** | Nexus `rag.retrieve` (catalog) + catalog `rag.retrieve` when `security_profile` wired (M-RAG.25) |
| Citations | **L3** | Formal `Citation` on `RetrievalResult` + `RagRetrieveOutput.citations` (M-RAG.29) |
| Vector backends (prod SLO) | **L3** | Catalog **stable:** `qdrant`, `pgvector`, `chroma`, `weaviate`, `lancedb`, `typesense`; **beta:** `pinecone`, `milvus`, `vespa`, `inmemory`; soak gates M-RAG.30/56 |
| Multi-tenant isolation | **L3** | Cross-backend contract tests vector (`inmemory`, `pgvector`, `weaviate`, `qdrant`, `chroma`, `lancedb`, `typesense` — M-RAG.62) + graph (M-RAG.35, M-RAG.41) |
| Evaluation depth | **L3** | Golden harness + `run_retrieval_load_soak` CI gate (M-RAG.36) |

**Overall engine posture:** **L3 implementation / L3 control plane** — **Frozen** harness layer (2026-06-13). L4 adaptive retriever selection → [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](ADAPTIVE_HARNESS_INTELLIGENCE.md).

---

## Engine depth audit register (2026-06-10)

Full findings from architecture + implementation review. **Category:** `gap` = missing capability · `incomplete_wiring` = implemented but incomplete · `below_prod_bar` = works but below prod bar · `prod_blocker` = blocks prod class · `design_boundary` = intentional design boundary.

| ID | Category | Finding | Severity | Plan | AUDIT-IDEAL |
|----|----------|---------|----------|------|-------------|
| GAP-RAG-01 | incomplete_wiring | ~~`RagProfile.query_expansion` / `INTERGRAX_RAG_QUERY_EXPANSION` not wired~~ — **closed M-RAG.23**: `query_expander_from_profile()` in bootstrap; deep tier → `multiquery` when `query_expansion != off` | **P0** | M-RAG.23 **Done** | 14.3 |
| GAP-RAG-02 | incomplete_wiring | ~~`toc_vector_store` not passed in default bootstrap~~ — **closed M-RAG.24**: `hierarchical_bootstrap` + `RagStack.toc_vectorstore_manager` + retriever bootstrap | **P1** | M-RAG.24 **Done** | 14.4 |
| GAP-RAG-03 | incomplete_wiring | ~~`IngestPipeline` bypasses `DualIndexStrategy`~~ — **closed M-RAG.24**: `IndexingManager` + `DualIndexStrategy` when `hierarchical_index_enabled` or `hierarchical` retriever | **P1** | M-RAG.24 **Done** | 14.4 |
| GAP-RAG-04 | prod_blocker | ~~Catalog `perform_rag_retrieve` had no poisoning filter~~ — **closed M-RAG.25**: mirrors `rag_step` when `security_profile.retrieval_poisoning_defense_enabled` | **P1** | M-RAG.25 **Done** | 14.5 |
| GAP-RAG-05 | prod_blocker | ~~Sync ingest loads full document into RAM with no size guard~~ — **closed M-RAG.26**: `sync_ingest_max_bytes` rejects oversized sync path; stream shard ingest remains workflow worker | **P1** | M-RAG.26 **Done** | 14.6 |
| GAP-RAG-06 | prod_blocker | ~~No Tier-0 async ingest job contract~~ — **closed M-RAG.26**: `rag.schedule_ingest_job` + idempotent `workflow_orchestrator` trigger | **P1** | M-RAG.26 **Done** | 14.6 |
| GAP-RAG-07 | incomplete_wiring | ~~No soak gate or ops runbook~~ — **closed M-RAG.30**: `prod_slo.py` soak contract; gate unit tests; integration soak `-m vectorstore_soak`; INTEGRATIONS runbook; `pinecone`/`milvus`/`vespa` remain **beta** until ops soak passes | **P1** | M-RAG.30 **Done** | — |
| GAP-RAG-08 | incomplete_wiring | ~~No OpenTelemetry spans on retrieve/ingest hot path~~ — **closed M-RAG.27**: `rag_spans.py` + `check_rag_otel_span_registry.py` | **P2** | M-RAG.27 **Done** | 14.7 |
| GAP-RAG-09 | below_prod_bar | ~~RAG aggregated metrics opt-in~~ — **closed M-RAG.57**: metrics default follows OTel spine when env unset; explicit `INTERGRAX_RAG_METRICS_ENABLED=false` still disables | **P2** | M-RAG.57 **Done** | 14.7 |
| GAP-RAG-10 | incomplete_wiring | ~~No retriever fallback chain~~ — **closed M-RAG.28**: `retriever_fallback_chain()` in `RetrieverEngine` | **P2** | M-RAG.28 **Done** | — |
| GAP-RAG-11 | below_prod_bar | ~~No structured retrieval errors~~ — **closed M-RAG.28**: `RetrievalError` taxonomy + optional `RetrieverVectorCircuitBreaker` | **P2** | M-RAG.28 **Done** | — |
| GAP-RAG-12 | below_prod_bar | ~~Asymmetric retry~~ — **closed M-RAG.28**: `RetrieverEngine.DEFAULT_MAX_RETRIES=2` | **P2** | M-RAG.28 **Done** | — |
| GAP-RAG-13 | incomplete_wiring | ~~No formal `Citation` on engine output~~ — **closed M-RAG.29**: `retrieval/citation.py` + `RagCitationResult` | **P2** | M-RAG.29 **Done** | — |
| GAP-RAG-14 | incomplete_wiring | ~~No embedding version policy~~ — **closed M-RAG.31**: warn on ingest, optional retrieve filter, reindex queue hook | **P2** | M-RAG.31 **Done** | — |
| GAP-RAG-15 | design_boundary | No autonomous MIME/size-based chunking or retriever selection — **Frozen** (M-RAG.58); owner: [`AHI-MAINT-04`](../maintainers/plans/ADAPTIVE_HARNESS_INTELLIGENCE.md#61av-harness-implementation-queue--adaptive-harness-intelligence-audit-maintenance-planned) | **P4** | M-RAG.58 **Frozen** | — |
| GAP-RAG-16 | below_prod_bar | ~~Heuristic-only tier routing~~ — **closed M-RAG.32**: optional `llm_route_enabled` + `llm_tier_classifier.py` with heuristic fallback | **P2** | M-RAG.32 **Done** | — |
| GAP-RAG-17 | incomplete_wiring | ~~`multiquery` not activated by `query_expansion`~~ — **closed M-RAG.23**: `effective_retriever(deep)` returns `multiquery` when expansion enabled | **P0** | M-RAG.23 **Done** | 14.3 |
| GAP-RAG-18 | prod_blocker | ~~No Tier-3 GraphRAG prod preset~~ — **closed M-RAG.33**: `production_graph_rag_profile()` requires `neo4j`; `production_rag_profile()` documented harness-only (in-memory graph) | **P1** | M-RAG.33 **Done** | — |
| GAP-RAG-19 | incomplete_wiring | ~~No inter-iteration retriever switch or latency budget trace~~ — **closed M-RAG.34**: `agentic_iteration_retriever_ids`, `agentic_max_total_latency_ms`, per-iteration trace fields | **P2** | M-RAG.34 **Done** | — |
| GAP-RAG-20 | prod_blocker | ~~No cross-backend tenant isolation contract~~ — **closed M-RAG.35**: `tenant_isolation_contract.py` + gate tests per backend; live qdrant probe in integration soak | **P1** | M-RAG.35 **Done** | — |
| GAP-RAG-21 | prod_blocker | ~~No RAG load/soak gate~~ — **closed M-RAG.36**: `evaluation/load_soak.py` concurrent retrieve SLO + CI `rag-guard.yml` | **P2** | M-RAG.36 **Done** | — |
| GAP-RAG-22 | below_prod_bar | ~~No semantic chunking ingest size guard~~ — **closed M-RAG.37**: `semantic_chunking_max_chars` + `semantic_chunking_size_exceeded` before chunk | **P2** | M-RAG.37 **Done** | — |
| GAP-RAG-23 | below_prod_bar | ~~M-RAG.6 query expansion **Partial**~~ — **closed M-RAG.23**: M-RAG.6 **Done** | **P0** | M-RAG.23 **Done** | 14.3 |
| GAP-RAG-24 | incomplete_wiring | ~~`create_rag_graph_store` hardcodes backends~~ — **closed M-RAG.38**: `RagGraphStoreBackend` registry in `graph/bootstrap/backend_registry.py` | **P0** | M-RAG.38 **Done** | — |
| GAP-RAG-25 | incomplete_wiring | ~~memgraph/falkordb not wired~~ — **closed M-RAG.39**: `CypherRagGraphStore` + registry factories | **P1** | M-RAG.39 **Done** | — |
| GAP-RAG-26 | prod_blocker | ~~No graph lifecycle sync~~ — **closed M-RAG.40**: `unlink_chunks` / `purge_graph` + catalog delete/purge hooks | **P1** | M-RAG.40 **Done** | — |
| GAP-RAG-27 | prod_blocker | ~~No graph tenant isolation~~ — **closed M-RAG.41**: `graph/tenant/graph_isolation_contract.py` + tenant scope on stores | **P1** | M-RAG.41 **Done** | — |
| GAP-RAG-28 | below_prod_bar | ~~`graph_rag` retriever beta~~ — **closed M-RAG.42**: metadata/chunk-linked entity seeds; **stable** | **P1** | M-RAG.42 **Done** | — |
| GAP-RAG-29 | incomplete_wiring | ~~hybrid channel not fused~~ — **closed M-RAG.53**: vector+keyword+graph fusion via `graph_channel_fusion.py`; `channel_contributions` on trace | **P1** | M-RAG.53 **Done** | — |
| GAP-RAG-30 | incomplete_wiring | ~~graph provenance not on trace~~ — **closed M-RAG.54**: `graph_provenance_records` + summary on `RetrievalTrace` | **P1** | M-RAG.54 **Done** | — |
| GAP-RAG-31 | prod_blocker | ~~No graph maintenance job~~ — **closed M-RAG.45**: `rag.schedule_graph_maintenance_job` + workflow contract | **P2** | M-RAG.45 **Done** | — |
| GAP-RAG-32 | incomplete_wiring | ~~No GraphIndexer plugin registry~~ — **closed M-RAG.46**: `register_graph_indexer_plugin()` | **P2** | M-RAG.46 **Done** | — |
| GAP-RAG-33 | gap | ~~`graph_store` catalog lacks Neptune, OrientDB, ArangoDB~~ — **closed M-RAG.49–51** (TigerGraph/JanusGraph out of scope) | **P3** | M-RAG.49–51 **Done** | — |
| GAP-RAG-34 | design_boundary | No Microsoft GraphRAG vendoring — optional harness-native `community_report` indexer (M-RAG.47 **Done**, default off) | — | M-RAG.47 **Done** | — |
| GAP-RAG-35 | incomplete_wiring | ~~prod slug list neo4j-only~~ — **closed M-RAG.48/55**: `APPROVED_PRODUCTION_GRAPH_STORE_SLUGS` (`neo4j`, `memgraph`, `falkordb`) | **P2** | M-RAG.48/55 **Done** | — |
| GAP-RAG-35b | incomplete_wiring | ~~`falkordb` absent from prod slug list~~ — **closed M-RAG.55**: graph soak gate + `falkordb` in `APPROVED_PRODUCTION_GRAPH_STORE_SLUGS` | **P2** | M-RAG.55 **Done** | — |
| GAP-RAG-09b | below_prod_bar | ~~RAG metrics opt-in vs OTel default-on~~ — **closed M-RAG.57**: metrics default follows `INTERGRAX_RAG_OTEL_SPANS_ENABLED` when unset | **P2** | M-RAG.57 **Done** | 14.7 |
| GAP-RAG-07b | incomplete_wiring | ~~Beta vector slugs lack harness soak gate~~ — **closed M-RAG.56**: `run_beta_adapter_soak` + gate for pinecone/milvus/vespa | **P2** | M-RAG.56 **Done** | — |
| GAP-RAG-39 | incomplete_wiring | ~~`rag.retrieve` diagnostics omit graph trace fields~~ — **closed M-RAG.60**: `channel_contributions`, `graph_provenance_records` on tool diagnostics | **P2** | M-RAG.60 **Done** | — |
| GAP-RAG-40 | incomplete_wiring | ~~`STABLE_PROD_SLO_SLUGS` omits `lancedb`/`typesense`~~ — **closed M-RAG.61**: tuple aligned with stable manifests | **P3** | M-RAG.61 **Done** | — |

**Traceability rule:** no open GAP-RAG row without a **Planned** M-RAG.\* deliverable in [`plan/RAG.md`](../maintainers/plans/RAG.md). **GAP-RAG-15** and **GAP-RAG-34** are explicit architectural boundaries, not harness defects.

---

## Materialized knowledge visibility

Physical storage or vector indexing is not query visibility. Workspace documents now carry
immutable `KnowledgeMaterializationOwnershipV1` metadata and an explicit visibility authority.
Legacy local-file and web records use `LEGACY_IMMEDIATE`; connected-source records require the
exact tenant, workspace, source, indexed binding, binding reference, delivery and remote identity.

Connected-source retrieval is visible only when the durable delivery receipt is `COMPLETED`,
has `completed_at`, has zero failed items, and matches every ownership identity. A durable active
materialization pointer selects the current committed version for each remote item. A prepared
delivery leaves that pointer unchanged; a committed newer version replaces it; an aborted or
malformed version cannot replace the current version. Missing, malformed, cross-scope or
incompatible authority fails closed.

Filtering occurs in the shared search-evidence mapping boundary before Ask evidence, citations or
context assembly. Current vector backends use bounded candidate post-filtering rather than a
backend-specific predicate; hidden candidates are never substituted and the requested result
limit remains deterministic. The ownership keys also preserve tenant/workspace/source/binding/
delivery dimensions for later binding-scoped purge.

Historical records without the new metadata remain parseable only through explicit legacy
compatibility. A connected source cannot use that compatibility path and remains hidden until
authoritative ownership is repaired. Recovery relies on durable receipt and pointer state, not
in-memory state. The next task is fenced atomic materialization commit; the later lifecycle task
will implement binding-scoped purge orchestration.
