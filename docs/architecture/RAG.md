# RAG and Retrieval

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/RAG.md`](../plan/RAG.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Audit layer:** 14 (RAG and Retrieval)  
**Related:** [`architecture/INTEGRATIONS.md`](INTEGRATIONS.md) (vector_store, document_parser, rerank_provider slugs) · [`architecture/MEMORY.md`](MEMORY.md) (Knowledge store vs LTM) · [`guides/AGENT_CREATION_GUIDE.md`](../guides/AGENT_CREATION_GUIDE.md) Appendix K §K.5  
**Implementation:** `intergrax/rag/`  
**Last architecture audit:** 2026-06-10 (engine depth vs production RAG systems)

---

## Purpose

RAG is a **full Tier-0 retrieval layer**, not a vector-search shortcut. One canonical path serves catalog tools (`rag.retrieve`), Nexus `ContextBuilder` / `RagStep`, and diagnostics.

**Rules:**

- Agents (Tier-2) MUST NOT call `vectorstore.query` directly.
- Tier-3 selects backends via `IntegrationProfile` and tuning via `RagProfile`.
- Vendor SDKs for vector stores and parsers live in the Integration Library; orchestration stays in `intergrax/rag/`.

```text
Tier-3 IntegrationProfile + RagProfile
  → create_default_rag_stack() / rag_runtime_bridge
  → RetrievalService + IngestPipeline
  → rag.* catalog tools + Nexus RagStep
```

---

## Maturity score (audit map L0–L4)

| Dimension | Score | Notes |
|-----------|-------|-------|
| Control-plane architecture (single path, typed contracts) | **L3** | `RetrievalService`, `RagProfile`, runtime bridges, 12 `rag.*` catalog tools |
| Retrieval mode breadth | **L2.5–L3** | hybrid, fusion, graph, agentic — graph **beta**; hierarchical path not default-wired |
| Ingest — short / medium documents | **L3** | Parser catalog, 5 chunking strategies, optional contextual enrich |
| Ingest — very large corpora | **L1.5–L2** | Synchronous in-memory pipeline; no stream ingest or job orchestration in engine |
| Observability | **L2** | `RetrievalTrace`, parser trace, opt-in metrics; no OTel spans on retrieve hot path |
| Security (poisoning) | **L2.5** | Enforced on Nexus `RagStep`; **not** on direct `rag.retrieve` tool path |
| Vector backends (prod SLO) | **L2** | Lab `qdrant` stable; cloud bridges `pinecone` / `qdrant` / `chroma` **beta** |

**Overall engine posture:** **L2.5 implementation / L3 control plane** — production-ready as a **Harness foundation** when Tier-3 defines explicit profiles; not drop-in for multi-GB corpora without async ingest and backend hardening.

Remediation queue: [plan/RAG.md — Phase M-RAG-DEPTH](plan/RAG.md#phase-m-rag-depth--production-hardening-post-audit-2026-06-10).

---

## End-to-end pipelines

```text
INGEST (rag.ingest_document / IngestPipeline)
  source → DocumentsLoader → ParserPipeline (integration catalog)
        → ChunkingEngine (strategy_id from RagProfile)
        → optional ContextualChunkEnricher (LLM)
        → EmbeddingManager (batched, retried)
        → VectorstoreManager.add_documents
        → optional GraphIndexer (heuristic | llm | heuristic_then_llm)

RETRIEVE (rag.retrieve / RetrievalService)
  query → QueryRouter (fast | standard | deep)     [heuristic; route_mode=off → standard]
        → optional AgenticRetrievalLoop (deep tier, opt-in)
        → RetrieverEngine (registry) → optional RerankerManager
        → score_threshold filter → RetrievalResult + RetrievalTrace
        → format_rag_context_text (citations via chunk metadata: source, page, doc_id)
```

**Wiring entry:** `create_default_rag_stack()` → `RagStack` on `ToolWiringContext` / `RuntimeConfig` via `rag_runtime_bridge.py`.

---

## Module map (`intergrax/rag/`)

| Layer | Module | Role |
|-------|--------|------|
| Profile | `profiles/rag_profile.py` | Retriever, reranker, routing, chunking, graph/agentic/hybrid flags; env `INTERGRAX_RAG_*` |
| Bootstrap | `bootstrap/rag_stack_bootstrap.py` | `create_default_rag_stack()` for Tier-3 wiring |
| Ingest | `ingest/ingest_pipeline.py` | Load → chunk → embed → index (+ optional graph) |
| Retrieval | `retrieval/retrieval_service.py` | **Single retrieve entry** — route → retrieve → rerank → filter |
| Routing | `routing/query_router.py` | `fast` / `standard` / `deep` tiers (adaptive cost) |
| Retrievers | `retrievers/` | Registry: vector, hybrid, fusion (RRF), MMR, parent–child, hierarchical, multi-query, graph_rag |
| Rerankers | `rerankers/` | Registry + integration slugs (`cohere_rerank`, `jina_rerank`) |
| Chunking | `document_splitters/` | `recursive`, `langchain_recursive`, `semantic`, `parent_child`, `docling` |
| Loaders | `document_loaders/` | Handler registry + `ParserPipeline`; parsers via Integration catalog |
| Embeddings | `embedding/` | Provider registry, batched pipeline, retry |
| Vector store | `vectorstore/` | Manager + hybrid/sparse; backends via Integration bridges |
| GraphRAG | `graph/` | `GraphStore` contract; inmemory / neo4j; heuristic/LLM indexer |
| Agentic | `retrieval/agentic_loop.py`, `query_refiner.py` | Budgeted deep-tier loop |
| Evaluation | `evaluation/metrics.py`, `golden_harness.py` | `recall@k`, MRR; golden CI scenarios |
| Observability | `tracking/metrics.py`, `observability_bridge.py` | Opt-in metrics + runtime plugin |
| Indexing | `indexing/` | `SingleIndexStrategy`, `DualIndexStrategy` (TOC + chunks) |

---

## Core contracts

| Contract | Module | Role |
|----------|--------|------|
| `RagProfile` | `profiles/rag_profile.py` | Platform defaults for retrieval, rerank, ingest, routing |
| `RetrievalService` | `retrieval/retrieval_service.py` | Canonical retrieval orchestration |
| `RetrievalRequest` / `RetrievalResult` | `retrieval/` | I/O + `RetrievalTrace` |
| `IngestPipeline` | `ingest/ingest_pipeline.py` | Configurable ingest |
| `RagStack` | `bootstrap/rag_stack_bootstrap.py` | Composed managers + `RetrievalService` |

`ToolWiringContext` and `RuntimeConfig` expose `retrieval_service`, `rag_profile`, `retriever_manager`, `reranker_manager`, `vectorstore_manager`, `embedding_manager`.

---

## Retrieval modes (registered retrievers)

| `retriever_id` | Strategy | When to use |
|----------------|----------|-------------|
| `vector_similarity` | Dense ANN | Fast tier; factual short queries |
| `hybrid` | Dense + lexical (`query_hybrid` or in-process alpha blend) | Default standard tier |
| `mmr` | Maximal marginal relevance | Diversity-sensitive context |
| `parent_child` | Child search + parent dedup | After `parent_child` chunking ingest |
| `multiquery` | Query expansion + merge | Multi-aspect queries (expander: deterministic default) |
| `hierarchical` | TOC index + chunk index | Large structured docs — **requires `toc_vector_store` wiring** |
| `fusion` | RRF over vector + hybrid + parent_child | Deep tier default |
| `graph_rag` | Vector seed + graph traversal | When `graph_rag_enabled` + `GraphStore` configured |

**Adaptive routing:** `QueryRouter` classifies by word count and simple heuristics (no LLM). Tier-3 maps tiers via `RagProfile.fast_retriever_id`, `retriever_id`, `deep_retriever_id`, `effective_retriever()`.

**Agentic deep tier:** `AgenticRetrievalLoop` — budgeted iterations, query refine (`deterministic` \| `llm`), stop on `agentic_min_chunks` / `agentic_min_score`. **Default:** `agentic_enabled=false`.

---

## Chunking and document scale

| Document class | Supported strategies | Engine behaviour |
|----------------|---------------------|------------------|
| Short text / small files | `langchain_recursive` (default), `recursive` | Single-pass ingest |
| Structured office / PDF | `docling`, smart handlers | Parser trace in ingest metadata |
| Semantic boundaries | `semantic` | Sentence-embedding boundaries — **O(n) embed cost** |
| Hierarchical context | `parent_child` | Child chunks indexed; parent metadata for `parent_child` retriever |
| Book-scale / TOC | `DualIndexStrategy` + `hierarchical` retriever | **Implemented but not default-wired** in `create_default_rag_stack()` / `IngestPipeline` |

**Limitation (audit 2026-06-10):** ingest loads full parsed document into memory before chunking; no streaming shard ingest in Tier-0 engine. Very large corpora SHOULD use Tier-3 async jobs (`workflow_orchestrator` slugs: `prefect`, `airflow`, `temporal`) — see plan M-RAG.26.

---

## Hybrid search and sparse vectors

- **In-memory / fallback:** `LexicalHybridSupport` + BM25 hash (`INTERGRAX_RAG_SPARSE_ENCODER=bm25_hash`).
- **Learned sparse:** optional SPLADE via `fastembed` (`sparse_encoder=splade`).
- **Qdrant:** native sparse vectors + RRF (`INTERGRAX_RAG_QDRANT_SPARSE`).
- **Weaviate:** native `query.hybrid` (`INTERGRAX_RAG_WEAVIATE_NATIVE_HYBRID`).

Vector-store catalog slugs and env prefixes: [`architecture/INTEGRATIONS.md`](INTEGRATIONS.md) §Vector store (RAG).

---

## GraphRAG

- Contract: `graph/contracts/graph_store.py`
- Backends: `inmemory` (lab default), `neo4j` (`INTERGRAX_RAG_GRAPH_STORE=neo4j`)
- Indexer modes: `heuristic`, `llm`, `heuristic_then_llm`
- Retriever: `graph_rag` — **beta**; `production_rag_profile()` enables graph with in-memory store (harness preset, not multi-tenant prod)

**Boundary:** Document knowledge graphs are retrieval infrastructure — not user entity / episodic memory. See [`architecture/MEMORY.md`](MEMORY.md) §Graph RAG ≠ agent memory.

---

## Integration boundaries

RAG **consumes** Integration Library categories; it does not duplicate vendor adapters.

| Integration category | RAG usage |
|---------------------|-----------|
| `vector_store` | Embedding indexes — implementations in `rag/vectorstore/`, catalog bridges in `integrations/providers/vector_store/` |
| `document_parser` | Ingest parsing — `CatalogDocumentParser` + `INTERGRAX_RAG_DOCUMENT_PARSER_SLUG` |
| `rerank_provider` | Vendor rerank APIs — `rerankers/` resolves via profile |
| `graph_store` | GraphRAG backends — optional `INTERGRAX_RAG_GRAPH_STORE` |
| `workflow_orchestrator` | Large-corpus reindex / async ingest (target M-RAG.26) |

Bootstrap: `create_vectorstore_manager()` in `vectorstore/bootstrap/` resolves via integration catalog when `vector_store` is configured on `IntegrationProfile`.

---

## Tenant scope and metadata

Retrieve and ingest accept scope fields (`tenant_id`, `session_id`, `user_id`, `workspace_id`) → `MetadataFilter` on vector query. `InMemoryVectorStore` enforces `tenant_id` mismatch as `ValueError`. Production isolation depends on vector backend namespace design (Weaviate multi-tenant schema supported).

---

## Security boundaries

| Control | Location | Scope |
|---------|----------|-------|
| Retrieval poisoning (trust score / quarantine) | `runtime/nexus/runtime_steps/rag_step.py` + `retrieval_security_wiring.py` | Nexus context build when `security_profile.retrieval_poisoning_defense_enabled` |
| Tool policy / risk levels | `rag.purge_collection` = CRITICAL | Catalog governance |
| Direct `rag.retrieve` | `tools/providers/rag/service.py` | **No poisoning filter** — Tier-3 must not expose ungoverned retrieve in untrusted surfaces (GAP-RAG-03 → M-RAG.25) |

---

## Observability

| Signal | Mechanism |
|--------|-----------|
| Per-request trace | `RetrievalResult.trace` — latencies, retriever, reranker, agentic stop reason |
| Tool diagnostics | `RagRetrieveOutput.diagnostics` |
| Parser ingest | `parser_trace` metadata; export Langfuse/Sentry via `parser_trace_exporter.py` |
| Nexus summary | `RagSummaryDiagV1` (PII-safe) |
| Aggregated metrics | `INTERGRAX_RAG_METRICS_ENABLED` → `RagMetricsCollector` + runtime plugin on `TASK_COMPLETED` |

Enable metrics: `INTERGRAX_RAG_METRICS_ENABLED=true` or `register_rag_observability_plugin(plugins)` from `tracking/observability_bridge.py`.

Per `(tenant_id, retriever_id, route_tier)`: `calls`, `retrieval_latency_ms`, `rerank_latency_ms`, `hybrid_calls`, `agentic_iterations`, `recall_at_k_avg`.

**Gap:** no OpenTelemetry spans on `RetrievalService.retrieve` / `IngestPipeline.run` — see plan M-RAG.27.

---

## Evaluation

- Metrics: `recall@k`, MRR — `evaluation/metrics.py`
- Golden harness: `tests/fixtures/rag_golden/retrieval_cases.json` — scenarios `retrieval`, `graph_rag`, `multi_hop`, `agentic`
- CI: `.github/workflows/rag-guard.yml`
- Citation preservation at **response composer** level (`FinalResponseComposer`); chunk metadata carries `source` / `page` for `format_rag_context_text`

---

## Strategy selection — architectural rule

The engine exposes **registries and profiles**, not a fully autonomous algorithm picker. Tier-3 MUST define:

1. `IntegrationProfile` — `vector_store`, `document_parser`, `rerank_provider`, optional `graph_store`
2. `RagProfile` — retriever per tier, chunking strategy, agentic/graph toggles
3. Optional presets — e.g. `production_rag_profile()` (graph on, in-memory graph store)

Automatic tier routing (`QueryRouter`) covers **cost/latency tiers only**, not MIME-based chunking or retriever auto-selection. L4 adaptive retriever selection is deferred to [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](ADAPTIVE_HARNESS_INTELLIGENCE.md).

---

## Catalog tools (`rag.*` bundle)

Registered in `tools/providers/rag/bundle.py`:

`rag.retrieve`, `rag.ingest_document`, `rag.rerank`, `rag.preview_retrieval`, `rag.list_collections`, `rag.describe_collection`, `rag.delete_documents`, `rag.purge_collection`, `rag.list_documents`, `rag.get_document`, `rag.check_index_status`, `rag.search_by_metadata`

Tool authoring: [`architecture/TOOLS.md`](TOOLS.md) · wiring: Appendix K §K.5.

---

## Known gaps (audit 2026-06-10)

| ID | Gap | Severity | Plan |
|----|-----|----------|------|
| GAP-RAG-01 | `RagProfile.query_expansion` not wired to `MultiQueryRetriever` / `RetrievalService` | P0 | M-RAG.23 |
| GAP-RAG-02 | `DualIndexStrategy` + `HierarchicalRetriever` not in default bootstrap | P1 | M-RAG.24 |
| GAP-RAG-03 | Poisoning defense only on Nexus path, not `rag.retrieve` | P1 | M-RAG.25 |
| GAP-RAG-04 | No stream / async ingest for multi-GB corpora | P1 | M-RAG.26 |
| GAP-RAG-05 | Vector cloud bridges remain **beta** | P1 | M-RAG.30 |
| GAP-RAG-06 | No OTel spans on retrieve/ingest hot path | P2 | M-RAG.27 |
| GAP-RAG-07 | `RetrieverEngine` raises after 1 retry — no retriever fallback chain | P2 | M-RAG.28 |
| GAP-RAG-08 | No formal `Citation` type on `RetrievalResult` | P2 | M-RAG.29 |
| GAP-RAG-09 | No embedding-model-version reindex policy | P2 | M-RAG.31 |

---

## Verification

```bash
uv run pytest tests/unit/rag/ tests/unit/tools/providers/rag/ -m gate -q
uv run pytest tests/integration/rag/ -q
uv run pytest tests/unit/applications/test_rag_runtime_bridge.py -m gate -q
```

Golden gate: `.github/workflows/rag-guard.yml` · fixtures: `tests/fixtures/rag_golden/`.
