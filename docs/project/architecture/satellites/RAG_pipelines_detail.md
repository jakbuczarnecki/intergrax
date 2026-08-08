# RAG — pipelines detail

**Parent hub:** [`RAG.md`](../RAG.md)

## End-to-end pipelines

```text
INGEST (rag.ingest_document / IngestPipeline)
  source → DocumentsLoader → ParserPipeline (integration catalog)
        → ChunkingEngine (strategy_id from RagProfile)
        → optional ContextualChunkEnricher (LLM)
        → EmbeddingManager (batched, retried)
        → IndexingManager (DualIndexStrategy when hierarchical profile) → VectorstoreManager
        → optional GraphIndexer (heuristic | llm | heuristic_then_llm)

RETRIEVE (rag.retrieve / RetrievalService)
  query → QueryRouter (fast | standard | deep)     [heuristic; route_mode=off → standard]
        → optional AgenticRetrievalLoop (deep tier, opt-in)
        → RetrieverEngine (registry) → optional RerankerManager
        → score_threshold filter → RetrievalResult + RetrievalTrace
        → `Citation` list + format_rag_context_text (source, page, doc_id)
        [poisoning filter: Nexus RagStep + catalog when security_profile enabled — M-RAG.25]
```

**Wiring entry:** `create_default_rag_stack()` → `RagStack` on `ToolWiringContext` / `RuntimeConfig` via `rag_runtime_bridge.py`.

---

## Module map (`intergrax/rag`)

| Layer | Module | Role |
|-------|--------|------|
| Profile | `profiles/rag_profile.py` | Retriever, reranker, routing, chunking, graph/agentic/hybrid flags; env `INTERGRAX_RAG_*` |
| Bootstrap | `bootstrap/rag_stack_bootstrap.py` | `create_default_rag_stack()` for Tier-3 wiring |
| Ingest | `ingest/ingest_pipeline.py` | Load → chunk → embed → index (+ optional graph) |
| Retrieval | `retrieval/retrieval_service.py` | **Single retrieve entry** — route → retrieve → rerank → filter |
| Routing | `routing/query_router.py` | `fast` / `standard` / `deep` tiers (adaptive cost) |
| Retrievers | `retrievers` | Registry: vector, hybrid, fusion (RRF), MMR, parent–child, hierarchical, multi-query, graph_rag |
| Resilience | `retrievers/resilience`, `retrieval/retrieval_errors.py` | Fallback chain, `RetrievalError`, optional vector circuit breaker |
| Governance | `governance/embedding_version_policy.py` | Embedding version warn/filter + reindex queue hooks |
| Rerankers | `rerankers` | Registry + integration slugs (`cohere_rerank`, `jina_rerank`) |
| Chunking | `document_splitters` | `recursive`, `langchain_recursive`, `semantic`, `parent_child`, `docling` |
| Loaders | `document_loaders` | Handler registry + `ParserPipeline`; parsers via Integration catalog |
| Embeddings | `embedding` | Provider registry (`hf`, `openai`, `ollama`, `vllm`, `llama_cpp`), batched pipeline, retry |
| Vector store | `vectorstore` | Manager + hybrid/sparse; backends via Integration bridges |
| GraphRAG | `graph` | RAG `GraphStore` ABC; bootstrap backends; indexers; adapters to Integration `graph_store` |
| Agentic | `retrieval/agentic_loop.py`, `query_refiner.py` | Budgeted deep-tier loop |
| Evaluation | `evaluation/metrics.py`, `golden_harness.py` | `recall@k`, MRR; golden CI scenarios |
| Observability | `tracking/metrics.py`, `observability_bridge.py` | Opt-in metrics + runtime plugin |
| Indexing | `indexing` | `SingleIndexStrategy`, `DualIndexStrategy` (TOC + chunks) |
| Governance | `vectorstore/governance`, `profiles/rag_profile_validator.py` | Collection ACL (M-RAG.65); profile bootstrap validation (M-RAG.63) |
| Reference workflows | `applications/_shared/reference_workflows/rag_async_ingest.py` | Tier-3 async ingest shard planner (M-RAG.67) |

---

## Core contracts

| Contract | Module | Role |
|----------|--------|------|
| `RagProfile` | `profiles/rag_profile.py` | Platform defaults for retrieval, rerank, ingest, routing |
| `RetrievalService` | `retrieval/retrieval_service.py` | Canonical retrieval orchestration |
| `RetrievalRequest` / `RetrievalResult` | `retrieval` | I/O + `RetrievalTrace` + `Citation` list |
| `Citation` | `retrieval/citation.py` | Structured provenance from chunk metadata |
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
| `multiquery` | Query expansion + merge | Deep tier when `query_expansion != off` (M-RAG.23); or explicit `retriever_id` |
| `hierarchical` | TOC index + chunk index | Large structured docs — `toc_vector_store` via `hierarchical_index_enabled` or `retriever_id=hierarchical` (M-RAG.24) |
| `fusion` | RRF over vector + hybrid + parent_child | Deep tier default |
| `graph_rag` | Vector seed + graph traversal + channel fusion | When `graph_rag_enabled` + `GraphStore` configured (**stable**) |

**Adaptive routing:** `QueryRouter` classifies by word-count heuristics; optional LLM classifier when `llm_route_enabled` (default off, env `INTERGRAX_RAG_LLM_ROUTE_ENABLED`). Trace field `route_classifier` = `heuristic` \| `llm`.

**Agentic deep tier:** `AgenticRetrievalLoop` — budgeted iterations, query refine (`deterministic` \| `llm`), stop on `agentic_min_chunks` / `agentic_min_score` or `agentic_max_total_latency_ms` (`latency_budget`). Optional per-iteration retriever schedule via `agentic_iteration_retriever_ids` (env `INTERGRAX_RAG_AGENTIC_ITERATION_RETRIEVERS`). Trace: `agentic_per_iteration_retriever_ids`, `agentic_per_iteration_latency_ms`, `agentic_refine_calls`, `agentic_latency_budget_ms`. **Default:** `agentic_enabled=false`.

---

## Chunking and document scale

| Document class | Supported strategies | Engine behaviour |
|----------------|---------------------|------------------|
| Short text / small files | `langchain_recursive` (default), `recursive` | Single-pass ingest |
| Structured office / PDF | `docling`, smart handlers | Parser trace in ingest metadata |
| Semantic boundaries | `semantic` | Sentence-embedding boundaries — **O(n) embed cost**; reject when loaded text exceeds `semantic_chunking_max_chars` (env `INTERGRAX_RAG_SEMANTIC_CHUNKING_MAX_CHARS`, default 100k) with `semantic_chunking_size_exceeded` |
| Hierarchical context | `parent_child` | Child chunks indexed; parent metadata for `parent_child` retriever |
| Book-scale / TOC | `DualIndexStrategy` + `hierarchical` retriever | **Wired** when `hierarchical_index_enabled` or `retriever_id=hierarchical` (M-RAG.24) |

**Limitation:** sync ingest loads full parsed document into memory before chunking; files above `RagProfile.sync_ingest_max_bytes` (env `INTERGRAX_RAG_SYNC_INGEST_MAX_BYTES`, default 50MB) are rejected with `sync_ingest_size_exceeded` and `async_job_recommended=true`. Schedule via `rag.schedule_ingest_job` (requires `workflow_orchestrator`). Shard/stream execution is workflow-worker responsibility.

---

## Hybrid search and sparse vectors

- **In-memory / fallback:** `LexicalHybridSupport` + BM25 hash (`INTERGRAX_RAG_SPARSE_ENCODER=bm25_hash`).
- **Learned sparse:** optional SPLADE via `fastembed` (`sparse_encoder=splade`) — optional dependency.
- **Qdrant:** native sparse vectors + RRF (`INTERGRAX_RAG_QDRANT_SPARSE`).
- **Weaviate:** native `query.hybrid` (`INTERGRAX_RAG_WEAVIATE_NATIVE_HYBRID`).

Vector-store catalog slugs and env prefixes: [`architecture/INTEGRATIONS.md`](INTEGRATIONS.md) §Vector store (RAG).

---

## GraphRAG architecture

GraphRAG indexes **document knowledge** (entity–relation graphs linked to retrieval chunks). It is **not** user episodic / entity memory — see [`architecture/MEMORY.md`](MEMORY.md) §Graph RAG ≠ agent memory and [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.7.1.

**Posture (audit 2026-06-13):** **L3 platform — Frozen** — backend registry (9 slugs), lifecycle sync, tenant isolation, stable `graph_rag` retriever with 3-channel fusion, maintenance jobs, indexer plugins. Vendor adapters Neptune/OrientDB/ArangoDB **Done** (M-RAG.49–51).

### Three-layer contract model

```text
Integration Library (Tier-0 catalog)
  integrations/contracts/graph_store.py — GraphStore Protocol
    run_query(statement, parameters) · get_node(node_id)
  Slugs: neo4j · memgraph · falkordb · neptune · orientdb · arangodb (see INTEGRATIONS §graph_store)
        ↓ adapter (e.g. Neo4jRagGraphStore)
RAG engine (Tier-0 intergrax/rag/graph/)
  graph/contracts/graph_store.py — GraphStore ABC
    upsert_node · upsert_edge · neighbors · find_nodes · link_chunk · chunk_ids_for_nodes
  graph/indexer/ — HeuristicGraphIndexer · LlmGraphIndexer · GraphIndexer Protocol
  retrievers/providers/graph_rag_retriever.py — GraphRagRetriever
        ↓ governance / explainability
Runtime architecture (Tier-1 contracts)
  runtime/architecture/graph_rag.py — GraphRagArchitectureContract · node/edge enums
  runtime/architecture/graph_provenance.py — GraphTraceFieldBundle
  runtime/architecture/hybrid_retrieval.py — vector + keyword + graph channel merge (reference)
```

| Layer | Contract | Extension point |
|-------|----------|-----------------|
| Integration | `integrations.contracts.graph_store.GraphStore` | `IntegrationManifest` + factory or `IntegrationPlugin` — [`EXTENSION_AUTHOR_GUIDE.md`](../guides/EXTENSION_AUTHOR_GUIDE.md) §2 |
| RAG graph | `rag.graph.contracts.graph_store.GraphStore` | Implement ABC or wrap integration store; inject via `create_default_rag_stack(graph_store=...)` |
| Indexer | `GraphIndexer` Protocol + `register_graph_indexer_plugin()` (M-RAG.46) | Custom class + profile mode or plugin id |
| Retriever | `BaseRetriever` | Register in retriever bootstrap; `retriever_id=graph_rag` |

**Distinction:** Integration `GraphStore` is a **vendor query facade** (Cypher-oriented). RAG `GraphStore` is a **document-graph semantic contract** (chunk linkage, neighborhood traversal). Adapters translate between them — pattern: `Neo4jRagGraphStore`.

### Plugin and backend matrix

| Backend | Integration slug | RAG bootstrap today | Prod Tier-3 (M-RAG.33) | Plan |
|---------|------------------|---------------------|------------------------|------|
| In-memory | — | **Default** (`inmemory`) | Harness / lab only | — |
| Neo4j | `neo4j` | **`neo4j`** via `Neo4jRagGraphStore` | **Required** today | Harden (M-RAG.38) |
| Memgraph | `memgraph` | **`memgraph`** via `CypherRagGraphStore` | Approved (M-RAG.48) | — |
| FalkorDB | `falkordb` | **`falkordb`** via `CypherRagGraphStore` | Approved (M-RAG.55) | — |
| Amazon Neptune | `neptune` | **`neptune`** via `CypherRagGraphStore` | Beta (M-RAG.49) | — |
| OrientDB | `orientdb` | **`orientdb`** via `CypherRagGraphStore` | Beta (M-RAG.50) | — |
| ArangoDB | `arangodb` | **`arangodb`** via `CypherRagGraphStore` | Beta (M-RAG.51) | — |

**Env / profile:**

| Knob | Role |
|------|------|
| `INTERGRAX_RAG_GRAPH_ENABLED` | Register `graph_rag` retriever + ingest graph indexing |
| `INTERGRAX_RAG_GRAPH_STORE` | Backend id — `inmemory` \| `neo4j` \| `memgraph` \| `falkordb` \| `neptune` \| `orientdb` \| `arangodb` (registry: M-RAG.38+) |
| `INTERGRAX_RAG_GRAPH_INDEXER_MODE` | `heuristic` \| `llm` \| `heuristic_then_llm` |
| `RagProfile.graph_rag_hops` | Max hops for `GraphRagRetriever` (default 1) |
| `production_graph_rag_profile()` | Tier-3 prod preset — neo4j backend + validation (M-RAG.33) |
| `IntegrationProfile.graph_store` | Integration slug resolved at bootstrap → passed to `create_rag_graph_store` |

**Microsoft GraphRAG:** not integrated. `ms365_graph` is Microsoft **Graph API** (mail/calendar) — unrelated. Optional harness-native **community-report indexer** (M-RAG.47) may approximate global/local search patterns without vendoring MS GraphRAG.

### Lifecycle — build · update · maintain · use

| Phase | Current behaviour | Gap | Plan |
|-------|-------------------|-----|------|
| **Build** | `IngestPipeline` → optional `resolve_graph_indexer()` (heuristic/LLM/community_report plugins) | — | — |
| **Update** | Re-ingest upsert + `rag.delete_documents` graph unlink (M-RAG.40) | — | — |
| **Maintain** | `purge_graph` on `rag.purge_collection` (M-RAG.40); `rag.schedule_graph_maintenance_job` (M-RAG.45) | — | — |
| **Use** | `GraphRagRetriever`: vector seed + metadata-linked entities + hop expansion + 3-channel fusion | **Stable** — vector+keyword+graph fusion (M-RAG.53); structured provenance records on trace (M-RAG.54) | — |
| **Isolate** | Vector tenant contract (M-RAG.35) + graph tenant contract (M-RAG.41) | — | — |

```text
BUILD (ingest)
  chunks indexed → GraphIndexer.index_documents(docs, chunk_ids)
    → upsert_node/edge · link_chunk(chunk_id)

USE (retrieve)
  deep tier when graph_rag_enabled → GraphRagRetriever
    → vector ANN seeds → graph neighbors → chunk_ids_for_nodes → merge candidates

UPDATE / MAINTAIN (planned)
  rag.delete_documents → unlink/remove graph artifacts (M-RAG.40)
  rag.schedule_graph_maintenance_job → workflow worker prune/reindex (**Done** M-RAG.45)
```

### Consumption surfaces (beyond RAG engine)

| Surface | Role |
|---------|------|
| `rag.retrieve` / `RetrievalService` | Canonical retrieval when `graph_rag_enabled` |
| `graph.run_query` / `graph.get_node` tools | Direct integration `graph_store` access — [`architecture/TOOLS.md`](TOOLS.md) |
| Skills `graph.entity_explorer`, `graph.path_finder`, `graph.knowledge_linker` | Agent packs combining graph tools + `rag.retrieve` |
| Maturity gate | `runtime/architecture/graph_rag.py` contract validation in governance evidence |

### GraphRAG production checklist (Tier-3)

1. `RagProfile.graph_rag_enabled=true` + `production_graph_rag_profile()` on product hosts.
2. `IntegrationProfile.graph_store=neo4j` (today); after M-RAG.48 — approved durable slug list.
3. Neo4j ops ready (Bolt `:7687`, infra profile `rag`) — [`infra/PORTS.md`](../../infra/PORTS.md).
4. Run graph tenant isolation gates when M-RAG.41 lands (alongside vector M-RAG.35).
5. Enable `graph_indexer_mode=llm` only with injected `LLMAdapter` on ingest path.

Remediation for all GraphRAG gaps: Phase **M-RAG-GRAPH** — [`plan/RAG.md`](../plan/RAG.md).

---

## Integration boundaries

RAG **consumes** Integration Library categories; it does not duplicate vendor adapters.

| Integration category | RAG usage |
|---------------------|-----------|
| `vector_store` | Embedding indexes — implementations in `rag/vectorstore`, catalog bridges in `integrations/providers/vector_store` |
| `document_parser` | Ingest parsing — `CatalogDocumentParser` + `INTERGRAX_RAG_DOCUMENT_PARSER_SLUG` |
| `rerank_provider` | Vendor rerank APIs — `rerankers` resolves via profile |
| `graph_store` | Integration backends — adapted to RAG `GraphStore` via `create_rag_graph_store` backend registry (M-RAG.38+) |
| `workflow_orchestrator` | Large-corpus reindex / async ingest via `rag.schedule_ingest_job` (M-RAG.26) |

Bootstrap: `create_vectorstore_manager()` in `vectorstore/bootstrap` resolves via integration catalog when `vector_store` is configured on `IntegrationProfile`.

---

## Tenant scope and metadata

Retrieve and ingest accept scope fields (`tenant_id`, `session_id`, `user_id`, `workspace_id`) → `MetadataFilter` on vector query. `InMemoryVectorStore` enforces `tenant_id` mismatch as `ValueError`. Cross-backend contract: `intergrax/rag/vectorstore/tenant/tenant_isolation_contract.py` — gate tests for `inmemory`, `pgvector`, `weaviate`, `qdrant`, `chroma`, `lancedb`, `typesense` (M-RAG.35, M-RAG.62).

Optional **collection-level ACL:** `CollectionAccessPolicy` on `VectorstoreManager` (M-RAG.65) — pair with UAEP at Tier-3 wiring.

**Graph store:** tenant namespace enforced via `GraphStore.tenant_id` + `graph/tenant/graph_isolation_contract.py` gate tests (M-RAG.41).

---

## Security boundaries

| Control | Location | Scope |
|---------|----------|-------|
| Retrieval poisoning (trust score / quarantine) | `plan_context_invocation.run_rag_context` + `retrieval_security_wiring.py` | Catalog `rag.retrieve` when `security_profile.retrieval_poisoning_defense_enabled` |
| Tool policy / risk levels | `rag.purge_collection` = CRITICAL | Catalog governance |
| Direct `rag.retrieve` | `tools/providers/rag/service.py` | Poisoning filter when `ToolWiringContext.security_profile` enabled (M-RAG.25) |

---

## Observability

| Signal | Mechanism | Gap |
|--------|-----------|-----|
| Per-request trace | `RetrievalResult.trace` | — |
| Tool diagnostics | `RagRetrieveOutput.diagnostics` | — |
| Parser ingest | `parser_trace` metadata; export Langfuse/Sentry | — |
| Nexus summary | `RagSummaryDiagV1` (PII-safe) | — |
| Aggregated metrics | `INTERGRAX_RAG_METRICS_ENABLED` → collector + runtime plugin | Default **on** when OTel spans enabled and env unset (M-RAG.57) |
| OpenTelemetry spans | `tracking/rag_spans.py` on `RetrievalService` + `IngestPipeline` stages | Default on; disable `INTERGRAX_RAG_OTEL_SPANS_ENABLED=false` |

Enable metrics: `INTERGRAX_RAG_METRICS_ENABLED=true` or `register_rag_observability_plugin(plugins)` from `tracking/observability_bridge.py`.

OTel span names (tracer `intergrax.rag`): `rag.retrieve`, `rag.retrieve.single_pass`, `rag.ingest`, `rag.ingest.load`, `rag.ingest.chunk`, `rag.ingest.index`, `rag.ingest.graph_index`. Gate: `scripts/maintenance/check_rag_otel_span_registry.py` (wired in `check_observability_gates.py`).

---

## Evaluation

- Metrics: `recall@k`, MRR, `precision@k`, `ndcg@k` — `evaluation/metrics.py`
- Golden harness: `tests/fixtures/rag_golden/retrieval_cases.json` — scenarios `retrieval`, `graph_rag`, `multi_hop`, `agentic`
- CI: `.github/workflows/rag-guard.yml`
- Load/soak SLO gate: `run_retrieval_load_soak()` — concurrent workers, p95 latency budget, per-query recall regression (`test_rag_load_soak_gate.py`; CI `.github/workflows/rag-guard.yml`)
- Citation preservation at **response composer** level; engine emits `RetrievalResult.citations` and `rag.retrieve` returns `RagCitationResult`

---

## Strategy selection — architectural rule

The engine exposes **registries and profiles**, not a fully autonomous algorithm picker (GAP-RAG-15). Tier-3 MUST define:

1. `IntegrationProfile` — `vector_store`, `document_parser`, `rerank_provider`, optional `graph_store`
2. `RagProfile` — retriever per tier, chunking strategy, agentic/graph toggles
3. Optional presets — `production_rag_profile()` (harness/lab, in-memory graph) or `production_graph_rag_profile()` (Tier-3 prod, neo4j)

Automatic tier routing (`QueryRouter`) covers **cost/latency tiers only**, not MIME-based chunking or retriever auto-selection. L4 adaptive retriever selection is deferred to [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](ADAPTIVE_HARNESS_INTELLIGENCE.md).

---

## Tier-3 production checklist (minimum)

1. **`IntegrationProfile`:** non-inmemory `vector_store`; `rerank_provider` when quality-critical; `graph_store=neo4j` if GraphRAG enabled.
2. **`RagProfile`:** explicit chunking per corpus type; `route_mode=auto`; set `query_expansion=off` to use `deep_retriever_id` (e.g. `fusion`) instead of `multiquery`.
3. **Ingest:** sync `rag.ingest_document` only below `sync_ingest_max_bytes`; oversized sources → `rag.schedule_ingest_job` with configured `async_ingest_workflow_id`.
4. **Security:** enable `retrieval_poisoning_defense_enabled` on `ApplicationSecurityProfile` so Nexus `rag.retrieve` (catalog) and catalog `rag.retrieve` share the same filter.
5. **Observability:** enable RAG metrics (`INTERGRAX_RAG_METRICS_ENABLED`); OTel spans on by default — disable only when needed.
6. **Isolation:** run gate `test_vectorstore_cross_tenant_isolation.py`; validate tenant namespace per chosen vector backend in ops (M-RAG.35).
7. **GraphRAG (when enabled):** neo4j durable backend; after M-RAG-GRAPH — graph delete-sync, graph tenant gates, extended golden scenarios.

---

## Catalog tools (`rag.*` bundle)

Registered in `tools/providers/rag/bundle.py`:

`rag.retrieve`, `rag.ingest_document`, `rag.schedule_ingest_job`, `rag.schedule_graph_maintenance_job`, `rag.rerank`, `rag.preview_retrieval`, `rag.list_collections`, `rag.describe_collection`, `rag.delete_documents`, `rag.purge_collection`, `rag.list_documents`, `rag.get_document`, `rag.check_index_status`, `rag.search_by_metadata`

Tool authoring: [`architecture/TOOLS.md`](TOOLS.md) · wiring: Appendix K §K.5.

---

## Verification

```bash
uv run pytest tests/unit/rag/ tests/unit/tools/providers/rag/ -m gate -q
uv run pytest tests/integration/rag/ -q
uv run pytest tests/unit/applications/test_rag_runtime_bridge.py -m gate -q
```

Golden gate: `.github/workflows/rag-guard.yml` · fixtures: `tests/fixtures/rag_golden`.

Implementation plan and step-by-step rollout: [`plan/RAG.md`](../plan/RAG.md).

---

## Audit verification evidence (2026-06-13)

Code-backed re-audit (iteration II on **Frozen** layer). Key confirmations:

| Check | Result | Evidence |
|-------|--------|----------|
| Unit + tools RAG gate | **108 passed** | `uv run pytest tests/unit/rag/ tests/unit/tools/providers/rag/ -m gate -q` |
| GraphRAG 3-channel fusion | **Closed M-RAG.53** | `graph_channel_fusion.py` + `test_hybrid_retrieval_graph_channel.py` |
| Graph provenance on trace | **Closed M-RAG.54** | `graph_provenance_builder.py` + `test_graph_provenance_retrieval_trace.py` |
| Graph store soak + falkordb prod | **Closed M-RAG.55** | `graph/soak/prod_slo.py` + `test_graph_store_prod_slo_soak.py` |
| Vendor graph_store adapters | **Closed M-RAG.49–51** | `test_graph_rag_vendor_adapters.py` |
| Load/soak gate | **Closed M-RAG.36** | `test_rag_load_soak_gate.py` |
| Tool graph diagnostics | **Closed M-RAG.60** | `test_rag_retrieve.py::test_rag_retrieve_diagnostics_include_graph_trace_fields` |
| Stable soak slug tuple | **Closed M-RAG.61** | `STABLE_PROD_SLO_SLUGS` includes `lancedb`, `typesense` |

**Posture:** RAG layer **Architecturally Mature** (2026-06-17) — M-RAG-ITERATION-III complete (M-RAG.62–M-RAG.68). Proposals I/J/K remain explicit rejections (Tier-0 stream ingest, ColBERT, AHI auto-selection).
