# RAG and Retrieval

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/RAG.md`](../plan/RAG.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Audit layer:** 14 (RAG and Retrieval)  
**Audit instruction:** [`guides/audit/RAG.md`](../guides/audit/RAG.md)  
**Related:** [`architecture/INTEGRATIONS.md`](INTEGRATIONS.md) (vector_store, document_parser, rerank_provider slugs) · [`architecture/MEMORY.md`](MEMORY.md) (Knowledge store vs LTM) · [`guides/AGENT_CREATION_GUIDE.md`](../guides/AGENT_CREATION_GUIDE.md) Appendix K §K.5  
**Implementation:** `intergrax/rag/`  
**Last architecture audit:** 2026-06-10 (full engine depth vs production RAG systems; code-verified — see §Audit verification evidence)

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

## Production readiness verdict (audit 2026-06-10)

| Question | Answer |
|----------|--------|
| Is the engine **absolutely production-ready** for every workload? | **No** — L2.5 implementation / L3 control plane. |
| Is it a **solid Harness foundation** when Tier-3 defines profiles? | **Yes** — single path, typed contracts, broad retriever registry, CI golden gate. |
| Does it **auto-select** parsers, chunkers, retrievers per document? | **No** — Tier-3 policy + optional L4 AHI (deferred). |
| Short / medium documents? | **Ready** with explicit `RagProfile`. |
| Multi-GB corpora / book-scale without Tier-3 jobs? | **Partial** — sync ingest rejected above `sync_ingest_max_bytes`; use `rag.schedule_ingest_job` + orchestrator worker (M-RAG.26). Streaming shard ingest remains Tier-3 workflow responsibility. |
| Untrusted surfaces via raw `rag.retrieve`? | **Ready** when `security_profile.retrieval_poisoning_defense_enabled` on `ToolWiringContext` (M-RAG.25). |

**Remediation:** every audit finding maps 1:1 to [`plan/RAG.md`](../plan/RAG.md) Phase M-RAG-DEPTH (GAP-RAG-01 … GAP-RAG-21 → M-RAG.23 … M-RAG.37).

---

## Maturity score (audit map L0–L4)

| Dimension | Score | Notes |
|-----------|-------|-------|
| Control-plane architecture (single path, typed contracts) | **L3** | `RetrievalService`, `RagProfile`, runtime bridges, 12 `rag.*` catalog tools |
| Retrieval mode breadth | **L2.5–L3** | hybrid, fusion, graph, agentic — graph **beta**; hierarchical + dual-index wired via profile (M-RAG.24) |
| Strategy selection (autonomous) | **L1.5** | Tier/cost routing only; MIME/size/retriever auto-pick deferred to Tier-3 + AHI |
| Ingest — short / medium documents | **L3** | Parser catalog, 5 chunking strategies, optional contextual enrich |
| Ingest — very large corpora | **L2–L2.5** | Sync path size-guarded; `rag.schedule_ingest_job` triggers orchestrator with idempotent contract (M-RAG.26); shard/stream execution in workflow worker |
| Resilience (retry, fallback, circuit breaker) | **L2** | Embedding retry=2; retriever retry=1; no fallback chain or circuit breaker |
| Observability | **L2** | `RetrievalTrace`, parser trace, opt-in metrics; no OTel spans on retrieve/ingest hot path |
| Security (poisoning) | **L3** | Nexus `RagStep` + catalog `rag.retrieve` when `security_profile` wired (M-RAG.25) |
| Citations | **L2** | Metadata in chunks + composer; no formal `Citation` on `RetrievalResult` |
| Vector backends (prod SLO) | **L2.5–L3** | Catalog **stable:** `qdrant`, `pgvector`, `chroma`, `weaviate`, `lancedb`, `typesense`; **beta:** `pinecone`, `milvus`, `vespa`, `inmemory`; soak gate `prod_slo.py` + gate tests (M-RAG.30) |
| Multi-tenant isolation | **L2.5–L3** | Cross-backend contract tests for `inmemory`/`pgvector`/`weaviate`/`qdrant` (M-RAG.35); prod namespace design still required per backend |
| Evaluation depth | **L2.5** | Golden harness (lab scenarios); no load/soak SLO gate |

**Overall engine posture:** **L2.5 implementation / L3 control plane** — production-ready as a **Harness foundation** when Tier-3 defines explicit profiles; not drop-in for multi-GB corpora, untrusted catalog retrieve, or autonomous algorithm selection without M-RAG-DEPTH closeout.

**Target after M-RAG-DEPTH:** **L3 implementation** for Tier-3 reference hosts. L4 adaptive retriever selection remains in [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](ADAPTIVE_HARNESS_INTELLIGENCE.md).

---

## Engine depth audit register (2026-06-10)

Full findings from architecture + implementation review. **Category:** `gap` = missing capability · `niedoróbka` = implemented but incomplete · `niska jakość` = works but below prod bar · `niegotowość` = blocks prod class · `ograniczenie` = intentional design boundary.

| ID | Category | Finding | Severity | Plan | AUDIT-IDEAL |
|----|----------|---------|----------|------|-------------|
| GAP-RAG-01 | niedoróbka | ~~`RagProfile.query_expansion` / `INTERGRAX_RAG_QUERY_EXPANSION` not wired~~ — **closed M-RAG.23**: `query_expander_from_profile()` in bootstrap; deep tier → `multiquery` when `query_expansion != off` | **P0** | M-RAG.23 **Done** | 14.3 |
| GAP-RAG-02 | niedoróbka | ~~`toc_vector_store` not passed in default bootstrap~~ — **closed M-RAG.24**: `hierarchical_bootstrap` + `RagStack.toc_vectorstore_manager` + retriever bootstrap | **P1** | M-RAG.24 **Done** | 14.4 |
| GAP-RAG-03 | niedoróbka | ~~`IngestPipeline` bypasses `DualIndexStrategy`~~ — **closed M-RAG.24**: `IndexingManager` + `DualIndexStrategy` when `hierarchical_index_enabled` or `hierarchical` retriever | **P1** | M-RAG.24 **Done** | 14.4 |
| GAP-RAG-04 | niegotowość | ~~Catalog `perform_rag_retrieve` had no poisoning filter~~ — **closed M-RAG.25**: mirrors `rag_step` when `security_profile.retrieval_poisoning_defense_enabled` | **P1** | M-RAG.25 **Done** | 14.5 |
| GAP-RAG-05 | niegotowość | ~~Sync ingest loads full document into RAM with no size guard~~ — **closed M-RAG.26**: `sync_ingest_max_bytes` rejects oversized sync path; stream shard ingest remains workflow worker | **P1** | M-RAG.26 **Done** | 14.6 |
| GAP-RAG-06 | niegotowość | ~~No Tier-0 async ingest job contract~~ — **closed M-RAG.26**: `rag.schedule_ingest_job` + idempotent `workflow_orchestrator` trigger | **P1** | M-RAG.26 **Done** | 14.6 |
| GAP-RAG-07 | niedoróbka | ~~No soak gate or ops runbook~~ — **closed M-RAG.30**: `prod_slo.py` soak contract; gate unit tests; integration soak `-m vectorstore_soak`; INTEGRATIONS runbook; `pinecone`/`milvus`/`vespa` remain **beta** until ops soak passes | **P1** | M-RAG.30 **Done** | — |
| GAP-RAG-08 | niedoróbka | ~~No OpenTelemetry spans on retrieve/ingest hot path~~ — **closed M-RAG.27**: `rag_spans.py` + `check_rag_otel_span_registry.py` | **P2** | M-RAG.27 **Done** | 14.7 |
| GAP-RAG-09 | niska jakość | RAG aggregated metrics remain opt-in (`INTERGRAX_RAG_METRICS_ENABLED`); OTel spans on default spine (`INTERGRAX_RAG_OTEL_SPANS_ENABLED`, default on) — documented M-RAG.27 | **P2** | M-RAG.27 **Done** | 14.7 |
| GAP-RAG-10 | niedoróbka | `RetrieverEngine` raises after 1 retry — no retriever fallback chain (`fusion` → `hybrid` → `vector`) | **P2** | M-RAG.28 | — |
| GAP-RAG-11 | niska jakość | No structured retrieval error taxonomy (retryable vs fatal); no circuit breaker on vector backend query | **P2** | M-RAG.28 | — |
| GAP-RAG-12 | niska jakość | Asymmetric resilience: `EmbeddingEngine` max_retries=2 vs `RetrieverEngine` max_retries=1 | **P2** | M-RAG.28 | — |
| GAP-RAG-13 | niedoróbka | No formal `Citation` model on `RetrievalResult`; citations only via chunk metadata + `FinalResponseComposer` | **P2** | M-RAG.29 | — |
| GAP-RAG-14 | niedoróbka | `embedding_model_version` on profile/metadata with no mismatch warn, filter, or reindex queue policy | **P2** | M-RAG.31 | — |
| GAP-RAG-15 | ograniczenie | No autonomous MIME/size-based chunking or retriever selection — Tier-3 must define `RagProfile` | — | Tier-3 + AHI | — |
| GAP-RAG-16 | niska jakość | `QueryRouter` tier selection is word-count heuristic only — no LLM intent / complexity classifier | **P2** | M-RAG.32 | — |
| GAP-RAG-17 | niedoróbka | ~~`multiquery` not activated by `query_expansion`~~ — **closed M-RAG.23**: `effective_retriever(deep)` returns `multiquery` when expansion enabled | **P0** | M-RAG.23 **Done** | 14.3 |
| GAP-RAG-18 | niegotowość | ~~No Tier-3 GraphRAG prod preset~~ — **closed M-RAG.33**: `production_graph_rag_profile()` requires `neo4j`; `production_rag_profile()` documented harness-only (in-memory graph) | **P1** | M-RAG.33 **Done** | — |
| GAP-RAG-19 | niedoróbka | `AgenticRetrievalLoop` cannot switch retriever between iterations; no RAG-level token/cost budget in trace | **P2** | M-RAG.34 | — |
| GAP-RAG-20 | niegotowość | ~~No cross-backend tenant isolation contract~~ — **closed M-RAG.35**: `tenant_isolation_contract.py` + gate tests per backend; live qdrant probe in integration soak | **P1** | M-RAG.35 **Done** | — |
| GAP-RAG-21 | niegotowość | No RAG load/soak gate for production SLO (latency, recall regression under concurrency) | **P2** | M-RAG.36 | — |
| GAP-RAG-22 | niska jakość | `semantic` chunking has O(n) embed cost per document — no ingest size guard or profile warning | **P2** | M-RAG.37 | — |
| GAP-RAG-23 | niska jakość | ~~M-RAG.6 query expansion **Partial**~~ — **closed M-RAG.23**: M-RAG.6 **Done** | **P0** | M-RAG.23 **Done** | 14.3 |

**Traceability rule:** no open GAP-RAG row without a **Planned** M-RAG.\* deliverable in [`plan/RAG.md`](../plan/RAG.md). GAP-RAG-15 is an explicit architectural boundary, not a harness defect.

---

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
        → format_rag_context_text (citations via chunk metadata: source, page, doc_id)
        [poisoning filter: Nexus RagStep + catalog when security_profile enabled — M-RAG.25]
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
| `multiquery` | Query expansion + merge | Deep tier when `query_expansion != off` (M-RAG.23); or explicit `retriever_id` |
| `hierarchical` | TOC index + chunk index | Large structured docs — `toc_vector_store` via `hierarchical_index_enabled` or `retriever_id=hierarchical` (M-RAG.24) |
| `fusion` | RRF over vector + hybrid + parent_child | Deep tier default |
| `graph_rag` | Vector seed + graph traversal | When `graph_rag_enabled` + `GraphStore` configured (**beta**) |

**Adaptive routing:** `QueryRouter` classifies by word count and simple heuristics (no LLM). Tier-3 maps tiers via `RagProfile.fast_retriever_id`, `retriever_id`, `deep_retriever_id`, `effective_retriever()`. LLM tier routing: M-RAG.32.

**Agentic deep tier:** `AgenticRetrievalLoop` — budgeted iterations, query refine (`deterministic` \| `llm`), stop on `agentic_min_chunks` / `agentic_min_score`. **Default:** `agentic_enabled=false`. Inter-iteration retriever switch: M-RAG.34.

---

## Chunking and document scale

| Document class | Supported strategies | Engine behaviour |
|----------------|---------------------|------------------|
| Short text / small files | `langchain_recursive` (default), `recursive` | Single-pass ingest |
| Structured office / PDF | `docling`, smart handlers | Parser trace in ingest metadata |
| Semantic boundaries | `semantic` | Sentence-embedding boundaries — **O(n) embed cost** (M-RAG.37 guard) |
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

## GraphRAG

- Contract: `graph/contracts/graph_store.py`
- Backends: `inmemory` (lab default), `neo4j` (`INTERGRAX_RAG_GRAPH_STORE=neo4j`)
- Indexer modes: `heuristic`, `llm`, `heuristic_then_llm`
- Retriever: `graph_rag` — **beta** retriever flag; Tier-3 prod uses `production_graph_rag_profile()` + `graph_store=neo4j` (M-RAG.33)

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
| `workflow_orchestrator` | Large-corpus reindex / async ingest via `rag.schedule_ingest_job` (M-RAG.26) |

Bootstrap: `create_vectorstore_manager()` in `vectorstore/bootstrap/` resolves via integration catalog when `vector_store` is configured on `IntegrationProfile`.

---

## Tenant scope and metadata

Retrieve and ingest accept scope fields (`tenant_id`, `session_id`, `user_id`, `workspace_id`) → `MetadataFilter` on vector query. `InMemoryVectorStore` enforces `tenant_id` mismatch as `ValueError`. Cross-backend contract: `intergrax/rag/vectorstore/tenant/tenant_isolation_contract.py` — gate tests for `inmemory`, `pgvector`, `weaviate`, `qdrant` (M-RAG.35).

---

## Security boundaries

| Control | Location | Scope |
|---------|----------|-------|
| Retrieval poisoning (trust score / quarantine) | `runtime/nexus/runtime_steps/rag_step.py` + `retrieval_security_wiring.py` | Nexus context build when `security_profile.retrieval_poisoning_defense_enabled` |
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
| Aggregated metrics | `INTERGRAX_RAG_METRICS_ENABLED` → collector + runtime plugin | Opt-in by design (GAP-RAG-09 documented) |
| OpenTelemetry spans | `tracking/rag_spans.py` on `RetrievalService` + `IngestPipeline` stages | Default on; disable `INTERGRAX_RAG_OTEL_SPANS_ENABLED=false` |

Enable metrics: `INTERGRAX_RAG_METRICS_ENABLED=true` or `register_rag_observability_plugin(plugins)` from `tracking/observability_bridge.py`.

OTel span names (tracer `intergrax.rag`): `rag.retrieve`, `rag.retrieve.single_pass`, `rag.ingest`, `rag.ingest.load`, `rag.ingest.chunk`, `rag.ingest.index`, `rag.ingest.graph_index`. Gate: `scripts/check_rag_otel_span_registry.py` (wired in `check_observability_gates.py`).

---

## Evaluation

- Metrics: `recall@k`, MRR — `evaluation/metrics.py`
- Golden harness: `tests/fixtures/rag_golden/retrieval_cases.json` — scenarios `retrieval`, `graph_rag`, `multi_hop`, `agentic`
- CI: `.github/workflows/rag-guard.yml`
- Load/soak SLO gate: **not present** (GAP-RAG-21) — M-RAG.36
- Citation preservation at **response composer** level; formal engine citations: M-RAG.29

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
4. **Security:** enable `retrieval_poisoning_defense_enabled` on `ApplicationSecurityProfile` so Nexus `RagStep` and catalog `rag.retrieve` share the same filter.
5. **Observability:** enable RAG metrics (`INTERGRAX_RAG_METRICS_ENABLED`); OTel spans on by default — disable only when needed.
6. **Isolation:** run gate `test_vectorstore_cross_tenant_isolation.py`; validate tenant namespace per chosen vector backend in ops (M-RAG.35).

---

## Catalog tools (`rag.*` bundle)

Registered in `tools/providers/rag/bundle.py`:

`rag.retrieve`, `rag.ingest_document`, `rag.schedule_ingest_job`, `rag.rerank`, `rag.preview_retrieval`, `rag.list_collections`, `rag.describe_collection`, `rag.delete_documents`, `rag.purge_collection`, `rag.list_documents`, `rag.get_document`, `rag.check_index_status`, `rag.search_by_metadata`

Tool authoring: [`architecture/TOOLS.md`](TOOLS.md) · wiring: Appendix K §K.5.

---

## Verification

```bash
uv run pytest tests/unit/rag/ tests/unit/tools/providers/rag/ -m gate -q
uv run pytest tests/integration/rag/ -q
uv run pytest tests/unit/applications/test_rag_runtime_bridge.py -m gate -q
```

Golden gate: `.github/workflows/rag-guard.yml` · fixtures: `tests/fixtures/rag_golden/`.

Implementation plan and step-by-step rollout: [`plan/RAG.md`](../plan/RAG.md).

---

## Audit verification evidence (2026-06-10)

Code-backed audit (`guides/audit/RAG.md`, mode `audit-only`). Key confirmations:

| Check | Result | Evidence |
|-------|--------|----------|
| Unit RAG suite | **114 passed**, 1 skipped (`fastembed` optional) | `uv run pytest tests/unit/rag/ -q` |
| Golden + tools + bridge gate | **24 passed** | `test_golden_retrieval_gate.py`, `tests/unit/tools/providers/rag/`, `test_rag_runtime_bridge.py` |
| `query_expansion` wired | **Closed M-RAG.23** | `effective_retriever(deep)` + `query_expander_from_profile()` in `retriever_bootstrap.py` |
| DualIndex not on default ingest | **Confirmed open** | `ingest_pipeline.py` → `vectorstore.add_documents`; bootstrap omits `toc_vector_store` |
| Catalog poisoning filter | **Closed M-RAG.25** | `perform_rag_retrieve` calls `filter_retrieved_chunks_for_poisoning` when `security_profile` enabled |
| Agents bypass vectorstore | **No violation** | No `vectorstore.query` in `agents/` |
| Vector manifest stability | **Partial** | `integrations/providers/vector_store/*/manifest.py` — stable vs beta per GAP-RAG-07 row above |

**Posture unchanged:** L2.5 implementation / L3 control plane. All GAP-RAG rows remain open except GAP-RAG-15 (architectural boundary). Closeout queue: M-RAG.23 … M-RAG.37 in [`plan/RAG.md`](../plan/RAG.md).
