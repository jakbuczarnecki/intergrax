# RAG-PROD-13 — Production Qualification Record

**Qualification date:** 2026-08-10  
**Status:** `READY_FOR_REVIEW`  
**Global status:** `PRODUCTION_QUALIFIED_WITH_LIMITATIONS`  
**RAG-PROD-14:** `READY` (declaration only; not started)

This is an evidence record, not a marketing or universal performance claim.

## Repository and environment

- Required ancestor: `7c80868720fa4123e1b156a521afb10035e3f30b`
- Qualification tested commit/base: `7c80868720fa4123e1b156a521afb10035e3f30b`
- Synchronized development base: `6f2f2bb6257ee7bf5410f591fadb1325b4633d3d`
- Preflight HEAD and `origin/development`: `6f2f2bb6257ee7bf5410f591fadb1325b4633d3d`
- Branch: `development`; no branch/worktree/detached-HEAD/rewrite operation used.
- The unrelated concurrent commit
  `6f2f2bb6257ee7bf5410f591fadb1325b4633d3d`
  (`fix(workspace): preserve verified local search evidence`) was preserved.
- It changed `intergrax/rag/ingest/ingest_pipeline.py` and
  `intergrax/rag/retrievers/contracts/base_retriever.py`; the affected
  source-scoped reingest qualification test passed `2 passed`, and no
  qualification runtime defect was found.
- Concurrent uncommitted changes were preserved and not staged.

Environment inventory:

- Python: `.venv\Scripts\python.exe`, Python `3.12.11`
- pytest: `8.4.2`
- `.venv`: ready
- `qdrant-client`: installed; `chromadb`: installed
- `psycopg`: not installed; `neo4j`: not installed
- Qdrant configuration variables: not configured; default `localhost:6333` reachable, HTTP `200`
- PgVector DSN/configuration: not configured; `localhost:5432` accepted TCP, but no usable client/DSN
- Chroma HTTP configuration: not configured; `localhost:8000` unreachable
- Neo4j configuration: not configured; `localhost:7687` unreachable
- No secrets were printed.

## Canonical gates

Final deterministic bundle:

```text
.venv\Scripts\python.exe -m pytest -q --tb=short
  tests/unit/knowledge/contracts/test_document_conformance.py
  tests/unit/rag/document_splitters/test_native_strategies.py
  tests/integration/rag/document_splitters/test_chunking_integration.py
  tests/e2e/rag/test_native_rag_retrieval_qualification.py
  tests/unit/rag/retrievers/test_hybrid_retriever.py
  tests/unit/rag/retrievers/test_fusion_retriever.py
  tests/integration/rag/test_hierarchical_retrieval_qualification.py
  tests/integration/rag/test_source_scoped_reingest_qualification.py
  tests/integration/rag/test_dual_index_reingest_qualification.py
  tests/unit/rag/vectorstore/test_source_ownership_contract.py
  tests/integration/rag/test_namespace_workspace_isolation_qualification.py
  tests/integration/rag/test_same_source_reingest_serialization.py
  tests/integration/rag/test_graph_reingest_qualification.py
  tests/unit/rag/test_rag_plugin_discovery.py
  tests/unit/rag/graph/test_graph_rag_neo4j_prod_contract.py
  tests/unit/architecture/test_langchain_boundary.py
  -k "not docling_strategy_uses_private_handle_and_skips_empty_items"
```

Result: `108 passed, 1 deselected` in `11.63s`.

The unfiltered run produced `108 passed, 1 failed` in `12.61s`; the sole
failure was the optional Docling test with `ModuleNotFoundError:
docling_core`. The native recursive and canonical RAG gates passed. This
optional dependency absence is not a runtime failure and was not repaired.

Provider/offline/live harness:

```text
.venv\Scripts\python.exe -m pytest -q --tb=short
  tests/unit/rag/vectorstore/test_vectorstore_contract.py
  tests/unit/rag/vectorstore/test_source_ownership_contract.py
  tests/unit/rag/vectorstore/test_vectorstore_cross_tenant_isolation.py
  tests/unit/integrations/providers/vector_store/test_qdrant_chroma.py
  tests/unit/integrations/providers/vector_store/test_chroma_ownership.py
  tests/unit/integrations/providers/vector_store/test_qdrant_point_id_normalization.py
  tests/unit/rag/vectorstore/test_real_backend_harness_skip_semantics.py
  tests/unit/rag/vectorstore/test_vectorstore_prod_slo_soak.py
  tests/integration/rag/vectorstore/test_vectorstore_real_backends.py
```

Result: `57 passed, 4 skipped` in `27.62s`. All four skips were Chroma HTTP
availability skips. PgVector passes in this bundle are offline fallback
contract evidence only: a PgVector store without DSN explicitly reports an
in-memory fallback.

Graph and local SLO bundle:

```text
.venv\Scripts\python.exe -m pytest -q --tb=short
  tests/unit/rag/evaluation/test_rag_load_soak_gate.py
  tests/unit/rag/graph/test_graph_store_prod_slo_soak.py
  tests/unit/rag/vectorstore/test_vectorstore_prod_slo_soak.py
  tests/unit/rag/graph/test_graph_rag_neo4j_prod_contract.py
```

Result: `13 passed` in `0.96s`.

## Identity, isolation and concurrency

- Identity law: `ADD IDs == QUERY.vector_id == OWNERSHIP IDs == DELETE input
  IDs` passed for InMemory, Qdrant fake/contract, PgVector fallback/contract
  and Chroma fake/contract. The real Qdrant gate passed the same law.
- Ownership selector is exact:
  `tenant_id + namespace + workspace_id + provenance.source_id`.
  No basename, semantic search or top-k ownership was accepted.
- Canonical tenant, namespace, workspace and combined namespace/workspace
  isolation passed with adversarial higher-scoring forbidden records.
- Real Qdrant live isolation passed for tenant, namespace, workspace and
  combined namespace/workspace scopes across two tenants, including
  adversarial foreign-scope records.
- The live proof covered exact ownership lookup, logical-ID parity,
  replacement, scoped delete with foreign-scope preservation, and successful
  isolated-collection cleanup.
- Same-source lease serialization passed: one owner and one conflict while
  the lease was held.
- Stale vector publication passed: the newer generation remained visible and
  the stale publication was not query-visible.
- Stale graph topology publication passed: `OLD_ENTITY` was not visible,
  `NEW_ENTITY` was active, and shared evidence remained preserved.
- Different source/scope keys remained independent in the canonical gates.
- These results claim operation ownership and generation visibility only.
  They do not claim distributed transactions, exactly-once processing or
  cross-store atomic commit.

## Live provider results

### Qdrant

- Offline contract: `QUALIFIED_OFFLINE_CONTRACT`
- Live result: `LIVE_QUALIFIED`
- Live gate: real Qdrant service, isolated collection, tenant isolation,
  namespace isolation, workspace isolation, combined namespace/workspace
  isolation, adversarial scope filtering, exact ownership lookup,
  logical-ID parity, replacement, scoped delete and foreign-scope
  preservation.
- Cleanup: `PASS`; qualification collection deleted.
- Bounded soak: `PASS`; 50 documents, 5 query rounds, p95 `30.647 ms`,
  threshold `2000.000 ms`.

The live gate was an ephemeral PowerShell here-string invoking
`create_qdrant_vector_store` and the native `VectorStoreRecord`,
`VectorStoreScope`, `add_records`, `query`, `count`,
`list_source_record_ids` and `delete` contracts. No qualification code was
added to the repository.

### PgVector

- Offline contract: `QUALIFIED_OFFLINE_CONTRACT`
- Live result: `BLOCKED_ENVIRONMENT`
- Limitation: no DSN/configuration and no `psycopg` or `psycopg2`; the observed
  TCP listener is not sufficient evidence of a usable PostgreSQL/pgvector
  lifecycle.
- No live claim, replacement claim or live soak claim.

### Chroma

- Offline contract: `QUALIFIED_OFFLINE_CONTRACT`
- Live result: `BLOCKED_ENVIRONMENT`
- Limitation: Chroma HTTP service at the canonical local endpoint was
  unreachable.
- No live claim, replacement claim or live soak claim.

## GraphRAG claim split

- Canonical graph indexing: `CANONICAL_HARNESS_QUALIFIED`
- Canonical GraphRAG retrieval: `CANONICAL_HARNESS_QUALIFIED`
- Canonical source replacement: `CANONICAL_HARNESS_QUALIFIED`
- Canonical stale topology fencing: `CANONICAL_HARNESS_QUALIFIED`
- Canonical shared evidence preservation: `CANONICAL_HARNESS_QUALIFIED`
- Live Neo4j baseline: `BLOCKED_ENVIRONMENT`
- Live Neo4j replacement: `BLOCKED_ENVIRONMENT`
- Live Neo4j generation fencing: `NOT_QUALIFIED`; no live evidence exists and
  the canonical InMemory generation proof is not evidence for Neo4j.

The existing Neo4j tests are contract tests over a fake integration graph
store. No live Neo4j qualification was claimed.

## Plugins and LangChain boundary

- Chunker discovery: `CANONICAL_HARNESS_QUALIFIED`
- Retriever discovery: `CANONICAL_HARNESS_QUALIFIED`
- Reranker discovery: `CANONICAL_HARNESS_QUALIFIED`
- Exact public groups verified:
  `intergrax.rag.chunkers`, `intergrax.rag.retrievers`,
  `intergrax.rag.rerankers`.
- An external entry-point chunker executed through the canonical splitter /
  ingest path.
- Native core LangChain claim:
  `QUALIFIED_OFFLINE_CONTRACT` — the canonical native RAG ABI/path does not
  require LangChain.
- Optional LangChain compatibility/provider paths remain allowed.
- No claim was made that the repository contains zero LangChain imports.

## Qualification matrix

| Capability | Result | Evidence type | Live? | Blocker / limitation | PROD-14 eligibility |
|---|---|---|---|---|---|
| KnowledgeDocument ABI | QUALIFIED_OFFLINE_CONTRACT | ABI conformance | No | native contract evidence | Eligible |
| Native recursive chunking | CANONICAL_HARNESS_QUALIFIED | native chunking gate | No | optional strategies separate | Eligible |
| Native ingest → retrieval | CANONICAL_HARNESS_QUALIFIED | native E2E gate | No | not provider-live evidence | Eligible |
| Dense retrieval | CANONICAL_HARNESS_QUALIFIED | native retrieval/harness | No | provider SLO separate | Eligible |
| Hybrid retrieval | CANONICAL_HARNESS_QUALIFIED | native unit contracts | No | backend parity separate | Eligible |
| Hierarchical retrieval | CANONICAL_HARNESS_QUALIFIED | parent-child + TOC gate | No | no parent reconstruction claim | Eligible |
| Dual-index | CANONICAL_HARNESS_QUALIFIED | reingest qualification | No | TOC/vector publication non-atomic | Eligible |
| Dual-index source replacement | CANONICAL_HARNESS_QUALIFIED | dual-index replacement gate | No | retry/recovery may be required | Eligible |
| Source ownership | QUALIFIED_OFFLINE_CONTRACT | exact enumeration tests | Qdrant live | unsupported providers fail closed | Eligible |
| Logical vector ID parity | QUALIFIED_OFFLINE_CONTRACT | four provider contracts; Qdrant live | Qdrant | provider physical IDs stay internal | Eligible |
| Delete lifecycle | QUALIFIED_OFFLINE_CONTRACT | provider lifecycle contracts; Qdrant live | Qdrant | other live services unavailable | Eligible |
| Same-basename isolation | QUALIFIED_OFFLINE_CONTRACT | source ownership/replacement gates | Qdrant live | PgVector/Chroma live blocked | Eligible |
| Tenant isolation | LIVE_QUALIFIED | canonical adversarial gate + Qdrant live | Qdrant | other live providers unavailable | Eligible |
| Namespace isolation | LIVE_QUALIFIED | canonical adversarial gate + Qdrant live | Qdrant | other live providers unavailable | Eligible |
| Workspace isolation | LIVE_QUALIFIED | canonical adversarial gate + Qdrant live | Qdrant | other live providers unavailable | Eligible |
| Single-index reingest | CANONICAL_HARNESS_QUALIFIED | source replacement gate | No | live stable-provider scope split | Eligible |
| Same-source serialization | CANONICAL_HARNESS_QUALIFIED | Events/barriers race gate | No | default coordinator process-local | Eligible |
| Stale vector publication fencing | CANONICAL_HARNESS_QUALIFIED | generation race gate | No | stale physical records need reclamation | Eligible |
| Stale TOC publication fencing | CANONICAL_HARNESS_QUALIFIED | dual-index generation gate | No | cross-store publication non-atomic | Eligible |
| GraphRAG canonical retrieval | CANONICAL_HARNESS_QUALIFIED | InMemory graph harness | No | live backend not qualified | Eligible |
| GraphRAG source replacement | CANONICAL_HARNESS_QUALIFIED | graph replacement gate | No | live Neo4j unavailable | Eligible |
| GraphRAG stale topology fencing | CANONICAL_HARNESS_QUALIFIED | deterministic graph race | No | Neo4j generation semantics unproven | Eligible with limitation |
| GraphRAG shared evidence preservation | CANONICAL_HARNESS_QUALIFIED | graph replacement gate | No | harness scope only | Eligible with limitation |
| Stable Qdrant provider | QUALIFIED_OFFLINE_CONTRACT + LIVE_QUALIFIED | offline contract + real service gate | Yes | local default endpoint; no remote claim | Eligible |
| Stable PgVector provider | QUALIFIED_OFFLINE_CONTRACT + BLOCKED_ENVIRONMENT | offline contract; live service unavailable | No | DSN/client/service qualification absent | Eligible with limitation |
| Stable Chroma provider | QUALIFIED_OFFLINE_CONTRACT + BLOCKED_ENVIRONMENT | offline contract; live service unavailable | No | HTTP service unavailable | Eligible with limitation |
| Plugin chunker | CANONICAL_HARNESS_QUALIFIED | entry-point discovery/ingest | No | one external fixture, not all plugins | Eligible |
| Plugin retriever | CANONICAL_HARNESS_QUALIFIED | entry-point discovery contract | No | one external fixture | Eligible |
| Plugin reranker | CANONICAL_HARNESS_QUALIFIED | entry-point discovery contract | No | one external fixture | Eligible |
| LangChain optionality/native core | QUALIFIED_OFFLINE_CONTRACT | native boundary gate | No | optional compatibility paths remain | Eligible |
| GraphRAG live backend / Neo4j | BLOCKED_ENVIRONMENT | contract-only adapter tests | No | Neo4j unavailable | Eligible with limitation |
| Live stable-provider replacement | LIVE_QUALIFIED | real Qdrant lifecycle | Qdrant only | PgVector/Chroma unavailable | Eligible with limitation |
| Live GraphRAG generation fencing | NOT_QUALIFIED | no live evidence | No | do not infer from InMemory | Follow-up evidence required |

## Document claim audit

### Confirmed

- Global status remains `PRODUCTION_QUALIFIED_WITH_LIMITATIONS`.
- Native ABI, scope, ownership, logical ID and failure boundaries remain
  qualified at offline/canonical-harness level.
- Qdrant is now documented as `QUALIFIED_OFFLINE_CONTRACT + LIVE_QUALIFIED`.
- PgVector, Chroma and Neo4j are not promoted to live-qualified.
- Non-transactional and non-exactly-once limitations remain explicit.

### Upgrade possible and applied

- Qdrant live isolation, ownership, replacement and delete lifecycle were
  promoted in `RAG.md` based on the real-service gate.

### Downgrade required

- None. No canonical runtime defect was found.

## Remaining limitations and non-claims

- PgVector and Chroma live qualification is blocked by environment.
- Neo4j live baseline, replacement and generation fencing are not qualified.
- The default source coordinator is process-local; durable CAS-backed
  composition is required for multi-worker/process safety.
- Publication is not a distributed transaction and does not provide
  exactly-once semantics.
- Stale physical records may require reclamation.
- The bounded soak is one local Qdrant observation, not a universal
  production SLO or capacity claim.
- The missing optional Docling package remains an environment limitation for
  that optional strategy test only.

## Validation and change record

- Changed by PROD-13: `docs/project/architecture/RAG.md`
- Added by PROD-13: `docs/project/maintainers/qualification/RAG_PRODUCTION_QUALIFICATION.md`
- Runtime production files: unchanged by PROD-13
- Test files: unchanged by PROD-13
- Evidence commit: the Git commit containing this record
- Push: reported in the final task handoff
- `HEAD == origin/development`: checked before the evidence commit
- Concurrent/unrelated modifications: preserved, not staged or restored.

