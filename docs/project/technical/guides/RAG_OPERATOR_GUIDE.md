# RAG Operator Guide

**Status:** canonical operator / deployment surface · **RAG-ENT-2**
**Audience:** platform engineer · SRE · production operator · deployment engineer · incident responder
**Architecture:** [`docs/project/architecture/RAG.md`](../../architecture/RAG.md)
**Developer extensions:** [`RAG_EXTENSION_GUIDE.md`](RAG_EXTENSION_GUIDE.md) — not this document
**Production handoff:** [`RAG_PRODUCTION_HANDOFF.md`](../../maintainers/qualification/RAG_PRODUCTION_HANDOFF.md)

Intergrax RAG is a **library subsystem** composed into Tier-3 applications and
agent runtimes. It is not a single standalone microservice with one universal
`/health` endpoint. Operators wire vector backends, optional graph backends,
embedding providers, scope, coordination and observability at deployment time.

---

## 1. Purpose and supported production boundary

| Dimension | Contract |
|---|---|
| **Global** | `PRODUCTION_QUALIFIED_WITH_LIMITATIONS` |
| **Deployment** | `APPROVED WITH EXPLICIT LIMITATIONS` |
| **RAG-LIVE track** | `CLOSED` — not reopened by this guide |

`WITH_LIMITATIONS` is the **final production contract**, not an unfinished
qualification state. Limitations are explicit architectural and deployment
boundaries documented below and in the handoff.

**Approved native surface** (summary): `KnowledgeDocument` ABI, native ingest →
retrieval, dense/hybrid/hierarchical retrieval, dual-index lifecycle,
source-scoped reingest and exact ownership, portable logical vector IDs,
tenant/namespace/workspace isolation, same-source serialization,
publication-generation visibility fencing, canonical GraphRAG harness lifecycle,
and stable vector contracts for Qdrant, PgVector and Chroma.

**Not in scope of this guide:** developer plugin authoring, historical
qualification ledgers, architecture internals — link to canonical sources.

---

## 2. Production architecture at a glance

```text
authoritative source files / upstream systems
  → loader / parser
  → KnowledgeDocument (scope + provenance.source_id)
  → chunking → embedding
  → vector store (+ optional TOC index)
  → optional graph index (GraphRAG)
  → RetrievalService / rag.retrieve
```

**Three coordination controls** (distinct):

1. **Source operation lease** — who owns a replacement lifecycle for one exact source key.
2. **Publication generation** — which prepared version is active.
3. **Retrieval visibility** — inactive or superseded generations are filtered before results are exposed.

**Not transactional:** vector, TOC and graph writes are **not** one distributed
transaction. Partial publication may remain after failure; recovery is operator
and application responsibility.

---

## 3. Provider / backend matrix

Final stable / live matrix after RAG-LIVE closeout:

| Provider / surface | Offline / harness | Live status | Operator notes |
|---|---|---|---|
| **Qdrant** | `QUALIFIED_OFFLINE_CONTRACT` | `LIVE_QUALIFIED` | Stable vector backend; source replacement supported |
| **PgVector** | `QUALIFIED_OFFLINE_CONTRACT` | `LIVE_QUALIFIED` | PostgreSQL + `vector` extension; explicit dimension required |
| **Chroma** | `QUALIFIED_OFFLINE_CONTRACT` | `LIVE_QUALIFIED` | HTTP/server path only for production evidence |
| **Neo4j GraphRAG baseline** | `CANONICAL_HARNESS_QUALIFIED` | `LIVE_QUALIFIED_BASELINE` | Scoped GraphRAG; does not alone claim generation fencing |
| **Neo4j publication-generation fencing** | harness + offline contract | `LIVE_QUALIFIED` | Generation visibility on live Neo4j |
| **Canonical GraphRAG** | `CANONICAL_HARNESS_QUALIFIED` | `LIVE_NEO4J_BASELINE + LIVE_NEO4J_GENERATION_FENCING` | Combined claim only with both live gates |

**Beta providers** (Weaviate, LanceDB, Typesense, Pinecone, Milvus, Vespa):
catalog `BETA` — **do not promote** to stable live qualification. Changed-source
replacement is unsupported or fail-closed for beta vector backends.

Detail and capability taxonomy: [`RAG.md`](../../architecture/RAG.md) §6–7.

---

## 4. Deployment checklist

- [ ] Select a **qualified** vector backend (Qdrant, PgVector or Chroma for stable live path).
- [ ] Configure explicit `tenant_id`, `namespace`, `workspace_id` on every ingest and query.
- [ ] Match **embedding model and dimension** to provider configuration (PgVector requires explicit dimension).
- [ ] For **multi-process / multi-worker** ingest or replacement: wire a **durable** `DocumentStoreSourceOperationCoordinator` over a shared `ConditionalDocumentStore` — not the default in-process coordinator.
- [ ] Confirm provider **health probes** succeed before marking workers ready.
- [ ] Enable **OTEL RAG spans** and optional retrieval metrics export.
- [ ] Define deployment-specific **alert thresholds** and SLOs (no universal values in repository evidence).
- [ ] If GraphRAG is enabled: validate Neo4j connectivity, qualified server/driver baseline, and generation fencing requirements.
- [ ] Run **provider-specific live isolation/lifecycle gate** in the target infrastructure before production SLO claims.
- [ ] Document backup, reingest and recovery ownership for authoritative sources.
- [ ] Treat plugins as **trusted installed Python code** (not sandboxed).
- [ ] Preserve qualification evidence references with the release record.

Full handoff checklist: [`RAG_PRODUCTION_HANDOFF.md`](../../maintainers/qualification/RAG_PRODUCTION_HANDOFF.md) § Production deployment checklist.

---

## 5. Configuration and environment

Group operator-relevant controls by function. Names below are verified in
current runtime and provider integration code. **Never commit real secrets.**

### 5.1 RAG runtime (`RagProfile` / `INTERGRAX_RAG_*`)

| Variable | Purpose |
|---|---|
| `INTERGRAX_RAG_RETRIEVER_ID` | Primary retriever (`hybrid` default) |
| `INTERGRAX_RAG_FAST_RETRIEVER_ID` / `INTERGRAX_RAG_DEEP_RETRIEVER_ID` | Route tiers |
| `INTERGRAX_RAG_RERANKER_ID` / `INTERGRAX_RAG_ENABLE_RERANK` | Reranking |
| `INTERGRAX_RAG_PREFETCH_TOP_K` / `INTERGRAX_RAG_FINAL_TOP_K` | Retrieval limits |
| `INTERGRAX_RAG_CHUNKING_STRATEGY` | Chunker selection |
| `INTERGRAX_RAG_HIERARCHICAL_INDEX` | Dual-index (main + TOC) |
| `INTERGRAX_RAG_NATIVE_HYBRID` | Native hybrid channel |
| `INTERGRAX_RAG_GRAPH_ENABLED` | GraphRAG retrieval |
| `INTERGRAX_RAG_GRAPH_STORE` | Graph backend (`inmemory`, `neo4j`, …) |
| `INTERGRAX_RAG_GRAPH_HOPS` | Graph traversal depth |
| `INTERGRAX_RAG_GRAPH_INDEXER_MODE` / `INTERGRAX_RAG_GRAPH_INDEXER_PLUGIN` | Graph indexing |
| `INTERGRAX_RAG_EMBEDDING_MODEL_VERSION` | Version governance on ingest/retrieve |
| `INTERGRAX_RAG_SPARSE_ENCODER` | Sparse/hybrid encoder mode |
| `INTERGRAX_RAG_QDRANT_SPARSE` | Qdrant sparse vectors (optional) |

Application composition also selects vector store, embedding provider and graph
store through `IntegrationProfile` — see provider sections and
[`INTEGRATIONS.md`](../../architecture/INTEGRATIONS.md).

### 5.2 Provider connectivity

**Qdrant:** `INTERGRAX_QDRANT_URL` or `INTERGRAX_QDRANT_HOST` + `INTERGRAX_QDRANT_PORT`; optional `INTERGRAX_QDRANT_API_KEY`, `INTERGRAX_QDRANT_COLLECTION`, `INTERGRAX_QDRANT_TENANT_ID`, `INTERGRAX_QDRANT_METRIC`, `INTERGRAX_QDRANT_BATCH_SIZE`.

**PgVector:** `INTERGRAX_PGVECTOR_DSN` or `INTERGRAX_PGVECTOR_CONNECTION_STRING`; **required** `INTERGRAX_PGVECTOR_DIMENSION` matching embeddings. Extra: `integrations-pgvector` (`psycopg`, `pgvector`).

**Chroma:** `INTERGRAX_CHROMA_MODE=http` (production path); `INTERGRAX_CHROMA_HOST`, `INTERGRAX_CHROMA_PORT`; optional `INTERGRAX_CHROMA_COLLECTION`, `INTERGRAX_CHROMA_TENANT_ID`, `INTERGRAX_CHROMA_METRIC`, `INTERGRAX_CHROMA_BATCH_SIZE`. `embedded` mode is dev/test only.

**Neo4j (GraphRAG):** `INTERGRAX_NEO4J_URL`, `INTERGRAX_NEO4J_USER`, `INTERGRAX_NEO4J_PASSWORD`; optional `INTERGRAX_NEO4J_API_KEY`, `INTERGRAX_NEO4J_TIMEOUT`. Set `INTERGRAX_RAG_GRAPH_STORE=neo4j` when using live graph backend.

Provider USAGE files: [Qdrant](../../../../intergrax/integrations/providers/vector_store/qdrant/USAGE.md) · [PgVector](../../../../intergrax/integrations/providers/vector_store/pgvector/USAGE.md) · [Chroma](../../../../intergrax/integrations/providers/vector_store/chroma/USAGE.md) · [Neo4j](../../../../intergrax/integrations/providers/graph_store/neo4j/USAGE.md).

### 5.3 Observability

| Variable | Purpose |
|---|---|
| `INTERGRAX_RAG_OTEL_SPANS_ENABLED` | RAG OTEL spans (default **on**) |
| `INTERGRAX_RAG_METRICS_ENABLED` | In-process retrieval metrics snapshot (defaults to span setting when unset) |
| `INTERGRAX_OTEL_ENDPOINT` / `INTERGRAX_OTEL_SERVICE_NAME` | Platform OTEL export (application-level) |

### 5.4 Source operation coordination

There is **no environment variable** for coordinator selection. Composition is explicit:

| Coordinator | Scope | When to use |
|---|---|---|
| `InProcessSourceOperationCoordinator` | **Process-local**, thread-safe | Single-process ownership domain, dev/harness |
| `DocumentStoreSourceOperationCoordinator` | **Durable** CAS leases + publication state | Multi-process / multi-worker concurrent ingest or replacement |

Wire via `IngestPipeline(source_coordinator=…)`, `VectorstoreManager.set_source_operation_coordinator(…)`, and graph store `set_source_operation_coordinator(…)` when applicable. Durable coordinator requires a shared `ConditionalDocumentStore` and a unique `owner_id` per worker.

Default when unset: `InProcessSourceOperationCoordinator()` — **not** safe for multi-process writers.

---

## 6. Health / readiness model

RAG readiness is **derived** from application composition, provider probes,
configuration validity and dependency reachability. The states below are
**operator concepts**, not a single runtime enum.

| Conceptual state | Typical signals |
|---|---|
| **HEALTHY** | Providers probe healthy; ingest/retrieve spans succeed; metrics within deployment baseline |
| **DEGRADED** | Elevated latency or partial backend slowness; retrieval still succeeds with fallback or reduced quality |
| **NOT_READY** | Workers starting; providers not yet opened; embeddings unavailable |
| **BACKEND_UNAVAILABLE** | Vector or graph backend unreachable; provider health `healthy=false` |
| **CONFIGURATION_ERROR** | Missing DSN, dimension mismatch, invalid mode, corrupt coordinator state |
| **DEPENDENCY_ERROR** | Embedding provider, document store for coordinator, or required SDK missing |

### Provider diagnostics (verified)

| Backend | Mechanism | Fail-closed behavior |
|---|---|---|
| **Qdrant** | Integration `health()` delegates to inner store probe; live qualification used reachable HTTP endpoint | Misconfiguration prevents open |
| **PgVector** | `SELECT 1` + `pg_extension` check for `vector` | No InMemory fallback; missing extension → unhealthy |
| **Chroma** | `/api/v2/heartbeat` via `HttpClient` at open | No embedded fallback in production path |
| **Neo4j** | `driver.verify_connectivity()` at open; `health()` via client probe | Connection failure raises at open |

Map probes into your platform readiness gates (load balancer, Kubernetes
readiness, synthetic checks). Query-only workers may be ready when vector
backend and embeddings are reachable even if ingest coordinators are contended.

---

## 7. Observability catalog

### 7.1 OTEL spans (tracer: `intergrax.rag`)

| Span | Operator meaning |
|---|---|
| `rag.retrieve` | End-to-end retrieval request |
| `rag.retrieve.single_pass` | Single-pass retrieval (non-agentic) |
| `rag.ingest` | Full ingest pipeline for one source |
| `rag.ingest.load` | Document load stage |
| `rag.ingest.chunk` | Chunking stage (`rag.ingest.chunking_strategy` attribute) |
| `rag.ingest.index` | Vector (and dual-index) publication (`rag.ingest.num_chunks`, `rag.ingest.dual_index`) |
| `rag.ingest.graph_index` | Graph indexing stage when GraphRAG enabled |

**Success pattern:** spans complete without ERROR status; ingest.index follows chunk; retrieve latency stable vs deployment baseline.

**Failure pattern:** ERROR status on span; exception recorded; often correlates with backend unavailable or configuration error.

Disable spans only deliberately: `INTERGRAX_RAG_OTEL_SPANS_ENABLED=false`.

### 7.2 Retrieval metrics (when `INTERGRAX_RAG_METRICS_ENABLED` or spans enabled)

Per `(tenant_id, retriever_id, route_tier)` snapshot fields:

| Field | Meaning |
|---|---|
| `calls` | Retrieval invocations |
| `retrieval_latency_ms` | Accumulated retrieval latency |
| `rerank_latency_ms` | Accumulated rerank latency |
| `hybrid_calls` | Hybrid retriever invocations |
| `hits_total` | Total hits returned |
| `recall_at_k_avg` | Average recall@k when ground truth supplied (evaluation/diagnostics) |
| `agentic_iterations` | Agentic retrieval iterations when enabled |

These are **in-process aggregates** exported via application diagnostics — not
universal Prometheus metric names unless your host re-exports them.

### 7.3 Complementing backend health

Combine RAG spans/metrics with provider health probes and infrastructure
monitoring (PostgreSQL, Qdrant, Chroma, Neo4j CPU/memory/disk). Span failures
with healthy infrastructure may indicate scope misconfiguration or embedding
errors.

---

## 8. Alerting guidance

Set **deployment-specific thresholds** from baseline soak methodology and SLO
template (§18). Do not use repository qualification latencies as universal alerts.

| # | Scenario | Condition (example) | Signal | Impact | Recommended response |
|---|---|---|---|---|---|
| 1 | Vector backend unavailable | Provider health unhealthy N minutes | PgVector `health()`, Chroma heartbeat failure, Qdrant probe | Queries return errors or empty; ingest fails | Fail over or restore backend; mark workers NOT_READY; see runbook §20.1 |
| 2 | Graph backend unavailable | Neo4j connectivity failures | Open/health errors; `rag.ingest.graph_index` errors | GraphRAG degraded or ingest failure when graph required | Restore Neo4j or disable graph path; see runbook §20.2 |
| 3 | Sustained retrieval failures | Error rate on `rag.retrieve` above baseline | OTEL span ERROR; `calls` stall | User-facing RAG unavailable | Check scope, embedding, backend; roll back bad deploy |
| 4 | Ingest failures | `rag.ingest` / `rag.ingest.index` ERROR rate high | Spans + application logs | New/updated knowledge not searchable | Inspect source path, dimension, lease conflict; retry canonical ingest |
| 5 | Source replacement failures | Replacement jobs incomplete; stale visible after timeout | Ingest spans stop before cleanup; ownership drift | Old and new content ambiguity (mitigated by generation fencing when promoted) | Retry canonical replacement; do not edit backend records manually |
| 6 | Coordinator / generation failure | Lease acquire conflicts; `promote_publication` false | Ingest errors; duplicate worker contention | Concurrent writers unsafe | Ensure durable coordinator; scale ingest workers with lease awareness |
| 7 | Latency degradation | p95 `retrieval_latency_ms` above deployment SLO | Metrics snapshot / APM | Slow answers | Capacity tune; check backend load; review top_k and rerank |
| 8 | Provider config / version mismatch | Dimension errors; schema errors; driver errors | CONFIGURATION_ERROR at open | Hard failures at startup or ingest | Align embedding dimension, Chroma 1.4.x pair, Neo4j 5.26 + driver 5.28.x per qualification |

---

## 9. Capacity / sizing methodology

The repository does **not** publish universal capacity tables. Size from:

| Variable | Affects |
|---|---|
| Source document count & size | Ingest duration, storage |
| Chunk count | Vector rows, TOC rows, graph evidence |
| Vector dimension | PgVector storage, Qdrant point size |
| Embedding model throughput | Ingest and query latency |
| Query QPS and `top_k` | Retrieval CPU and backend load |
| Graph evidence / topology | Neo4j memory and traversal cost |
| Retention and replacement churn | Stale physical record accumulation |
| Dual-index enabled | Second vector collection / table pressure |

**Qualification soak tests** (e.g. 50 records, 5 query rounds in live gates) are
**methodology references** for your own benchmarks — not production capacity claims.

### Deployment benchmark checklist

1. Baseline ingest throughput for representative document sizes.
2. Query p95/p99 at expected QPS with production embedding model.
3. Replacement lifecycle under concurrent updates to same `source_id`.
4. GraphRAG traversal latency if enabled.
5. Backend disk growth over retention window.
6. Observe stale physical record accumulation at provider storage layer.

---

## 10. Persistence model

| Class | Examples | Authority |
|---|---|---|
| **SOURCE / CANONICAL** | Upstream files, CMS exports, authoritative `KnowledgeDocument` sources | **Authoritative** for rebuild |
| **DERIVED** | Vector indexes, TOC/section index, graph nodes/edges/chunk evidence | Rebuildable from canonical sources when available |
| **OPERATIONAL CONTROL** | `DocumentStoreSourceOperationCoordinator` lease and publication records | Required for multi-process coordination; backup with deployment policy |

Logical vector IDs are portable; provider physical IDs are internal. Generation
metadata (`__intergrax_source_publication_generation`) controls visibility.

---

## 11. Backup, restore and DR

**Invariant:** there is **no distributed cross-store transaction**. Backup and
restore across vector, TOC, graph and control state is **not globally atomic**.

### Operator guidance

- Use **backend-native backups** (PostgreSQL, Qdrant snapshots, Chroma persistence volume, Neo4j backup tools) per deployment policy.
- Restoring stores from **inconsistent points in time** can yield derived-index mismatch vs canonical sources.
- **Safest recovery** when consistency is uncertain: restore authoritative sources and operational control state as appropriate, then run **canonical scoped reingest/rebuild** through `IngestPipeline` / `rag.ingest_document`.
- **Do not** manually fabricate vector ownership or generation metadata in provider backends.
- If upstream sources no longer exist, full rebuild may be **impossible** — plan retention accordingly.

---

## 12. Reindex / reingest recovery

Use **canonical ingest and replacement mechanisms** only. **Do not** edit vector
or graph records directly in provider storage.

| Procedure | When | Safe path |
|---|---|---|
| **A. One-source reingest** | Source content updated | Canonical `IngestPipeline` / `rag.ingest_document` with same `provenance.source_id` and scope |
| **B. Failed replacement retry** | Post-publication cleanup incomplete | Retry replacement; generation fencing hides stale evidence logically |
| **C. Partial publication** | Vector published; TOC or graph stage failed | Retry ingest/replacement; monitor spans per stage |
| **D. Full derived-index rebuild** | Corruption or DR; sources available | Scoped reingest per source; or controlled full reingest from authoritative catalog |
| **E. Provider migration** | Moving Qdrant → PgVector etc. | Reindex all sources on new backend; re-run live qualification in target infra |
| **F. Embedding model / dimension change** | Model swap | Full reindex required; PgVector dimension must match; update `INTERGRAX_PGVECTOR_DIMENSION` |

Tools: `rag.ingest_document`, `rag.delete_documents` (scoped), `rag.list_collections`, `rag.describe_collection` — see [`rag tools USAGE`](../../../../intergrax/tools/providers/rag/USAGE.md).

---

## 13. Stale physical record maintenance

**Invariant:** logical correctness ≠ immediate physical deletion.

Publication generation fencing may render stale evidence **invisible** while
physical records temporarily remain in vector or graph storage. This is
**logically safe**; reclamation is a maintenance concern.

| Topic | Guidance |
|---|---|
| **Why safe** | Retrieval filters by active publication generation; stale rows are non-queryable |
| **Normal cleanup** | Successful canonical replacement removes source-scoped stale IDs when cleanup completes |
| **Retry** | Re-run canonical replacement to complete cleanup after transient failure |
| **Purge / admin** | Use only provider operations intended for your deployment; broad purge is not normal per-source cleanup |
| **Full rebuild** | Safer when unsure of physical state |
| **Monitoring** | No dedicated stale-record count metric in current telemetry; observe provider storage/record counts until dedicated telemetry exists |

---

## 14. Multi-process / HA

### Exact supported boundary

```text
InProcessSourceOperationCoordinator = process-local synchronization ONLY
```

For **multi-process concurrent ingestion or replacement** on the same source
key, use `DocumentStoreSourceOperationCoordinator` backed by a shared
`ConditionalDocumentStore` (CAS leases + publication promotion).

There is **no** repository evidence qualifying **active-active ingestion without
durable coordination**. Default `IngestPipeline` uses in-process coordinator
when none is supplied.

### Query horizontal scaling vs ingest scaling

| Pattern | Evidence | Notes |
|---|---|---|
| **Query horizontal scaling** | Supported when workers are read-only against consistent derived indexes | Scale retrieval workers; ensure embedding and vector backend capacity |
| **Concurrent ingest / replacement** | Requires durable coordinator | Without it, same-source operations may conflict unsafely |

TTL default for durable leases: `300` seconds (`ttl_seconds` constructor parameter).

---

## 15. Rolling deployment / version skew

**Do not assume** mixed-version RAG workers participating in the same
publication/ingest lifecycle are safe unless explicitly requalified.

Treat as **unsafe across concurrent writers** when any of these change:

- `KnowledgeDocument` ABI
- `VectorStoreScope` semantics
- logical vector ID law
- ownership / replacement semantics
- publication generation semantics
- `GraphScope` / `RagEvidence` model
- `SourceOperationCoordinator` semantics

**Safe upgrade rule:** avoid semantic version skew across concurrent writers when
any reopening criterion applies. Prefer drain writers → upgrade → validate →
resume ingest.

**Query-only nodes:** rolling upgrade is lower risk when workers do not publish;
still validate read compatibility after ABI or scope changes.

Reopening criteria: [`RAG_LIVE_BACKEND_CLOSEOUT.md`](../../maintainers/qualification/RAG_LIVE_BACKEND_CLOSEOUT.md) §9 and handoff § Reopening criteria.

---

## 16. Upgrade / migration checklist

### Intergrax upgrade

- [ ] Review reopening criteria for RAG contract changes in release notes.
- [ ] Re-run affected qualification gates if ABI, scope, ownership or coordinator semantics changed.
- [ ] Validate `IntegrationProfile` and `RagProfile` env compatibility.

### Backend versions

| Backend | Qualified baseline (repository evidence) | Migration notes |
|---|---|---|
| Qdrant | Live-qualified real service (RAG-PROD-13) | Major upgrades → requalify; reindex if storage format changes |
| PgVector / PostgreSQL | `pgvector/pgvector:0.8.0-pg16`, PG 16 (15A-R2) | Legacy JSONB `intergrax_pgvector` tables **incompatible** — not auto-migrated |
| Chroma | `chromadb==1.4.1` / `chromadb/chroma:1.4.1` (15B-R2) | Server/client pair must match; HTTP mode for production |
| Neo4j | `neo4j:5.26-community`, driver `neo4j==5.28.4` (15C/15D) | Legacy graph schema **not** silently migrated |

### Embedding / document ABI

- [ ] Embedding model or dimension change → **full reindex**.
- [ ] `KnowledgeDocument` ABI change → **requalification** and likely full reindex.

---

## 17. Security and data-governance responsibilities

### System guarantees (verified contract)

| Guarantee | Boundary |
|---|---|
| Tenant / namespace / workspace routing isolation | `VectorStoreScope`; provider-enforced predicates where qualified |
| Exact source ownership | `provenance.source_id` + scope; not basename |
| Reserved routing metadata protection | User metadata cannot override routing fields |
| Scoped deletion semantics | Delete inputs are logical IDs in exact scope |

### Deployment responsibility

| Area | Owner |
|---|---|
| Credentials and secrets storage | Deployment |
| TLS / transport | Deployment |
| Encryption at rest | Deployment / backend |
| Network segmentation | Deployment |
| Backup encryption | Deployment |
| Access policy and audit | Deployment |

**No certification claims** in repository evidence.

**Plugin trust boundary:** plugins are **trusted installed Python code**, not
sandboxed. Global plugin architecture is a separate platform track.

---

## 18. Deletion / retention

| Operation | Behavior |
|---|---|
| **Logical source deletion** | Scoped delete via canonical API removes ownership-linked logical IDs |
| **Derived data** | Vector/TOC/graph entries for scoped IDs removed per provider contract when cleanup completes |
| **Stale physical records** | May remain temporarily after replacement; logically invisible under generation fencing |
| **Retention** | Deployment-owned; regulatory physical erasure requires verifying backend-level reclamation per compliance policy |

Do not claim immediate physical erasure where generation semantics allow temporary stale state.

---

## 19. Deployment SLO template

Populate with deployment-specific values. **No universal targets** in repository evidence.

| Dimension | Metric / signal | Objective (define locally) | Window | Action on breach |
|---|---|---|---|---|
| Query availability | `rag.retrieve` success rate | e.g. 99.9% | 30d | Page on-call; scale query workers |
| Query latency | p95 `retrieval_latency_ms` | baseline + margin | 7d | Tune top_k, rerank, backend |
| Ingest success rate | `rag.ingest` completion | e.g. 99% | 7d | Inspect failures; retry backlog |
| Replacement completion | replacement job success | define | 7d | Retry canonical replacement |
| Data freshness | time source updated → searchable | define | per source class | Scale ingest; fix coordinator contention |
| Backend availability | provider health | define | 30d | Failover / restore |
| RTO | time to restore query path | define | incident | Execute DR runbook |
| RPO | acceptable source re-ingest window | define | incident | Restore sources + reindex |

---

## 20. Incident runbooks

Compact format for each scenario.

### 20.1 Vector backend unavailable

| | |
|---|---|
| **SYMPTOM** | `rag.retrieve` / `rag.ingest.index` errors; provider health unhealthy |
| **IMPACT** | Retrieval down or ingest blocked |
| **CHECK** | Provider health probe; backend service/container; network; credentials |
| **SAFE ACTION** | Mark workers degraded; route traffic if multi-region; restore backend |
| **RECOVERY** | Restore backend; validate health; replay failed ingests from authoritative sources |
| **DO NOT** | Manually insert or patch vector records |

### 20.2 Neo4j unavailable

| | |
|---|---|
| **SYMPTOM** | Graph open/health failures; `rag.ingest.graph_index` errors |
| **IMPACT** | GraphRAG unavailable; graph-indexing ingest may fail |
| **CHECK** | Bolt connectivity; `verify_connectivity`; container health |
| **SAFE ACTION** | Disable graph ingest if optional; restore Neo4j service |
| **RECOVERY** | Restore Neo4j; re-run graph indexing for affected sources |
| **DO NOT** | Hand-edit graph ownership or generation in Cypher |

### 20.3 Embedding / provider dependency unavailable

| | |
|---|---|
| **SYMPTOM** | Ingest fails at embedding; zero-dimension or timeout errors |
| **IMPACT** | No new indexed content |
| **CHECK** | Embedding provider health; API quotas; model version env |
| **SAFE ACTION** | Pause ingest queue; fix provider |
| **RECOVERY** | Resume ingest; retry failed jobs |
| **DO NOT** | Store raw vectors without canonical pipeline |

### 20.4 Failed ingest

| | |
|---|---|
| **SYMPTOM** | `rag.ingest` span ERROR; partial `rag.ingest.index` |
| **IMPACT** | Source not searchable or partially published |
| **CHECK** | Span attributes; source path; dimension; scope; lease conflict |
| **SAFE ACTION** | Retry canonical ingest for one source |
| **RECOVERY** | Complete ingest; verify retrieval |
| **DO NOT** | Direct backend inserts |

### 20.5 Failed source replacement

| | |
|---|---|
| **SYMPTOM** | Replacement started; cleanup incomplete; old generation still visible incorrectly |
| **IMPACT** | Stale or duplicate logical visibility risk mitigated by generation fencing when promoted |
| **CHECK** | Coordinator ownership; `promote_publication`; per-stage spans |
| **SAFE ACTION** | Retry replacement with durable coordinator |
| **RECOVERY** | Complete canonical replacement lifecycle |
| **DO NOT** | Delete-by-guess in provider UI |

### 20.6 Coordinator / generation failure

| | |
|---|---|
| **SYMPTOM** | Lease acquire returns conflict; publication not promoted |
| **IMPACT** | Concurrent writers unsafe; stuck replacements |
| **CHECK** | Shared document store reachable; lease TTL; worker `owner_id` |
| **SAFE ACTION** | Reduce concurrent writers per source; fix document store |
| **RECOVERY** | Expire/retry leases; complete promotion |
| **DO NOT** | Run multi-process ingest with in-process coordinator |

### 20.7 Provider configuration mismatch

| | |
|---|---|
| **SYMPTOM** | Open failures; dimension mismatch; mode errors |
| **IMPACT** | Hard startup or ingest failure |
| **CHECK** | Env vars vs embedding dimension; Chroma mode=http; PgVector extension |
| **SAFE ACTION** | Fix configuration; roll back deploy |
| **RECOVERY** | Redeploy with corrected config |
| **DO NOT** | Disable fail-closed checks |

### 20.8 Provider / server version mismatch

| | |
|---|---|
| **SYMPTOM** | Driver/protocol errors; schema errors |
| **IMPACT** | Intermittent or total failure |
| **CHECK** | Chroma 1.4.1 pair; Neo4j 5.26 + driver 5.28.x; pgvector 0.8 on PG16 |
| **SAFE ACTION** | Align versions to qualified baseline or requalify new pair |
| **RECOVERY** | Upgrade/downgrade to supported matrix |
| **DO NOT** | Assume backward compatibility without evidence |

### 20.9 Stale physical evidence accumulation

| | |
|---|---|
| **SYMPTOM** | Storage growth; logical queries correct |
| **IMPACT** | Cost; ops noise — not logical corruption |
| **CHECK** | Provider record counts; incomplete replacements |
| **SAFE ACTION** | Retry canonical replacement cleanup |
| **RECOVERY** | Scoped rebuild if cleanup cannot complete |
| **DO NOT** | Broad unscoped purge without impact analysis |

### 20.10 Partial / inconsistent restore

| | |
|---|---|
| **SYMPTOM** | Queries return wrong/missing chunks after DR |
| **IMPACT** | Data integrity risk |
| **CHECK** | Compare restore timestamps across vector/TOC/graph/control store |
| **SAFE ACTION** | Stop writes; assess authoritative sources |
| **RECOVERY** | Restore sources + control state; scoped full reingest |
| **DO NOT** | Patch individual vectors to match guesses |

---

## 21. Troubleshooting

### Application-level

- Inspect OTEL traces for `intergrax.rag` spans and ERROR status.
- Snapshot retrieval metrics via host diagnostics (`hybrid_calls`, `hits_total`, `retrieval_latency_ms`).
- Verify `VectorStoreScope` on requests matches ingest scope.
- Confirm `provenance.source_id` stability across reingest.

### Provider-level

| Backend | Diagnostic | Label |
|---|---|---|
| PgVector | Provider `health()` — `SELECT 1` + `vector` extension | Production |
| Chroma | Heartbeat at open; HTTP reachability | Production |
| Qdrant | Integration health / service HTTP | Production |
| Neo4j | `verify_connectivity`; `health()` on graph store client | Production |

### REFERENCE / QUALIFICATION DIAGNOSTIC (not universal production commands)

Repository qualification environments — use only to reproduce evidence:

```text
# PgVector (repository compose — not universal production topology)
docker compose -f infra/docker/postgresql/docker-compose.yml up pgvector
uv run pytest tests/integration/rag/vectorstore/test_pgvector_live_qualification.py -q -s

# Chroma
INTERGRAX_RUN_CHROMA_LIVE=1 uv run pytest tests/integration/rag/vectorstore/test_chroma_live_qualification.py -q -s

# Neo4j baseline
INTERGRAX_RUN_NEO4J_LIVE=1 uv run pytest tests/integration/rag/test_neo4j_live_qualification.py -q -s
```

Qualification Compose credentials and ports are **reference environments** — see §22.

---

## 22. Qualification vs production environment

Repository Compose stacks used in RAG-LIVE gates are **repeatable
qualification/reference environments**. They prove adapter behavior under
controlled conditions; they are **not** automatic universal production topology
recommendations.

| Environment | What it proves | Operator may reuse |
|---|---|---|
| `infra/docker/postgresql/docker-compose.yml` (pgvector) | PgVector live lifecycle, isolation, soak | Pattern for PG+pgvector sizing tests |
| `infra/docker/chromadb/docker-compose.yml` | Chroma 1.4.1 HTTP lifecycle | Pattern for Chroma server pairing |
| `infra/docker/neo4j/docker-compose.yml` | Neo4j 5.26 GraphRAG baseline + fencing | Pattern for graph sizing tests |
| Qdrant (RAG-PROD-13) | Live Qdrant isolation and replacement | Pattern only — endpoint is deployment-specific |

**Production deployment decisions** (HA topology, cloud region, credentials,
encryption, multi-tenant isolation at infra layer) remain **deployment-owned**.
Repeat live qualification in target infrastructure before deployment-specific SLO
claims.

### Evidence references

1. [`RAG.md`](../../architecture/RAG.md)
2. [`RAG_PRODUCTION_HANDOFF.md`](../../maintainers/qualification/RAG_PRODUCTION_HANDOFF.md)
3. [`RAG_PRODUCTION_QUALIFICATION.md`](../../maintainers/qualification/RAG_PRODUCTION_QUALIFICATION.md)
4. [`RAG_PGVECTOR_LIVE_QUALIFICATION.md`](../../maintainers/qualification/RAG_PGVECTOR_LIVE_QUALIFICATION.md)
5. [`RAG_CHROMA_LIVE_QUALIFICATION.md`](../../maintainers/qualification/RAG_CHROMA_LIVE_QUALIFICATION.md)
6. [`RAG_NEO4J_LIVE_BASELINE_QUALIFICATION.md`](../../maintainers/qualification/RAG_NEO4J_LIVE_BASELINE_QUALIFICATION.md)
7. [`RAG_NEO4J_GENERATION_FENCING_QUALIFICATION.md`](../../maintainers/qualification/RAG_NEO4J_GENERATION_FENCING_QUALIFICATION.md)
8. [`RAG_LIVE_BACKEND_CLOSEOUT.md`](../../maintainers/qualification/RAG_LIVE_BACKEND_CLOSEOUT.md)

RAG live closeout ancestor: `79e3826fa8cddb97a73ca4ae2feca0ddb3966897`.

---

## 23. Known limitations

Explicit non-guarantees (must remain visible to operators):

| Limitation | Consequence |
|---|---|
| No distributed cross-store transaction | Vector, TOC, graph not atomically consistent |
| No exactly-once ingestion | Retries may repeat work |
| Default coordinator is process-local | Multi-process ingest requires durable coordinator |
| Stale physical records may remain | Reclamation / retry / rebuild may be needed |
| Live qualification is environment-specific | No universal capacity, latency or cloud SLO claim |
| Beta vector providers | No stable live promotion; replacement may be unsupported |
| PgVector legacy JSONB schema | Incompatible; not auto-migrated |
| Neo4j legacy graph schema | Not silently migrated or reinterpreted |
| Plugins not sandboxed | Supply-chain and install policy is deployment responsibility |
| Mixed-version concurrent writers | Not assumed safe across contract changes |

**RAG-LIVE:** `CLOSED` — next enterprise handoff track is RAG-ENT-3 (not started by this document).

---

## Quick navigation

| Need | Document |
|---|---|
| Architecture & qualification matrix | [`RAG.md`](../../architecture/RAG.md) |
| Production decision & handoff | [`RAG_PRODUCTION_HANDOFF.md`](../../maintainers/qualification/RAG_PRODUCTION_HANDOFF.md) |
| Live closeout | [`RAG_LIVE_BACKEND_CLOSEOUT.md`](../../maintainers/qualification/RAG_LIVE_BACKEND_CLOSEOUT.md) |
| Developer extensions | [`RAG_EXTENSION_GUIDE.md`](RAG_EXTENSION_GUIDE.md) |
| Catalog tools | [`rag/USAGE.md`](../../../../intergrax/tools/providers/rag/USAGE.md) |
