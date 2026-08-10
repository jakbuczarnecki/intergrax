# RAG and Retrieval

**Status:** Canonical architecture · **PRODUCTION_QUALIFIED_WITH_LIMITATIONS**
**Scope:** native Intergrax RAG architecture and qualification boundary after RAG-FINAL-10A–10D
**Implementation:** `intergrax/rag`
**Plan/history:** [`../maintainers/plans/RAG.md`](../maintainers/plans/RAG.md)
**Next roadmap items:** RAG-DEV-12, RAG-PROD-13, RAG-PROD-14

This document is the current source of truth for RAG architecture, provider
taxonomy, qualification status and failure boundaries. It does not grant a
full production decision: live qualification remains owned by RAG-PROD-13/14.

## Navigation and documentation inventory

| Classification | File | Ownership |
|---|---|---|
| **CANONICAL** | `docs/project/architecture/RAG.md` | Current RAG architecture and qualification |
| **SATELLITE** | [`satellites/RAG_pipelines_detail.md`](satellites/RAG_pipelines_detail.md) | Pipeline/module detail; current status points here |
| **SATELLITE** | [`../capabilities/architecture/satellites/LANGCHAIN_INDEPENDENCE_native_document_contract.md`](../capabilities/architecture/satellites/LANGCHAIN_INDEPENDENCE_native_document_contract.md) | `KnowledgeDocument` ABI |
| **HISTORICAL / PLAN** | [`../maintainers/plans/RAG.md`](../maintainers/plans/RAG.md) | Implementation history and roadmap; not runtime truth |
| **HISTORICAL / PLAN** | `../maintainers/plans/satellites/RAG_audit_history.md` | Detailed historical audit register |
| **RELATED OWNER** | `../capabilities/architecture/LANGCHAIN_INDEPENDENCE.md` | LangChain boundary and optionality |
| **RELATED PLAN** | `../capabilities/plan/LANGCHAIN_INDEPENDENCE.md` | LangChain migration history |
| **CATALOG OWNER** | [`INTEGRATIONS.md`](INTEGRATIONS.md) | Provider catalog taxonomy, not RAG qualification |
| **NAVIGATION ONLY** | [`../technical/guides/audit_slices/RAG.md`](../technical/guides/audit_slices/RAG.md) | Read-scope and audit entry point |

Older current-state passages in the architecture hub and pipeline satellite
were superseded by RAG-FINAL-10A–10D and are not retained as competing truth.
Historical evidence remains in the plan and audit-history documents.

## 1. Canonical contracts and identity

### KnowledgeDocument ABI

`KnowledgeDocument` is the canonical portable document ABI. RAG owns its
semantics; the neutral Tier-0 implementation is
`intergrax/knowledge/contracts/document.py`, imported as:

```python
from intergrax.knowledge.contracts import KnowledgeDocument
```

The document carries immutable identity, scope, content, metadata and
provenance. `tenant_id`, `namespace` and `workspace_id` are system-owned
scope fields. `provenance.source_id` is the authoritative source identity;
same basenames, paths or display names do not establish ownership.
The full field and lineage contract is the linked `KnowledgeDocument` satellite.

### VectorStoreScope and native vector ABI

`VectorStoreScope` is the explicit routing boundary:

```text
tenant_id + namespace + workspace_id
```

The native vector ABI uses `VectorStoreRecord`, `VectorStoreHit` and the
provider boundary for add, query, ownership enumeration and delete. Scope is
validated before provider calls; user metadata cannot override it.

The frozen portable ID invariant is:

```text
VectorStoreRecord.vector_id
  == logical portable persisted vector ID
  == ADD result IDs
  == QUERY.vector_id
  == OWNERSHIP IDs
  == DELETE input IDs
```

Provider physical IDs are internal implementation details. They are never
ownership IDs, delete inputs or a fallback when the logical ID is absent.

## 2. Canonical architecture

The native path is:

```text
source
  → parser/loader
  → KnowledgeDocument normalization and provenance
  → native chunking
  → embedding provider
  → VectorStoreRecord / vector backend
  → RetrievalService
  → optional reranker and graph channel
```

- **Chunking:** native recursive chunking is the core baseline. Semantic,
  parent-child and provider/plugin strategies are selectable. Hierarchical
  retrieval uses child chunks plus a section/TOC index; it does not promise
  reconstruction of a parent document.
- **Embeddings:** a provider-neutral embedding contract returns vectors in
  input order and preserves document identity, scope and provenance.
- **Vector retrieval:** dense similarity is the native base. Hybrid retrieval
  may combine dense and lexical/sparse channels; fusion and reranking are
  optional strategy layers.
- **Hierarchical retrieval:** `DualIndexStrategy` maintains the main chunk
  index and a TOC/section index. The TOC is a section index, not a second
  full-document copy.
- **GraphRAG:** graph indexing links entities, relationships and chunk
  evidence to canonical vector IDs. Retrieval seeds from vector results,
  traverses graph relations and fuses graph evidence with retrieval channels.
- **Profiles and routing:** `RagProfile`, registries and `RetrievalService`
  compose the path. Tier routing is cost/latency routing, not autonomous
  MIME-based algorithm selection.
- **Extensibility:** supported surfaces are parser/loader, metadata enricher,
  chunker, embedding provider, vector backend, retriever, reranker and graph
  indexer where the selected graph path supports it. Detailed authoring is
  deferred to RAG-DEV-12.

The pipeline satellite contains module-level flow and extension detail; it
does not own current qualification.

## 3. Source ownership and replacement

The canonical ownership selector is:

```text
tenant_id + namespace + workspace_id + provenance.source_id
```

`list_source_record_ids(source_id, scope)` means exact persisted ownership
enumeration inside the supplied `VectorStoreScope`. It is not semantic
search, top-k retrieval or basename lookup. Equal basenames do not imply
equal source ownership.

### Single-index replacement

The qualified lifecycle is:

```text
old ownership snapshot
  → prepare
  → publish current version
  → determine current ownership
  → stale = old - current
  → cleanup stale
```

There is no delete-before-publish step. Preparation/embedding failure before
publication preserves the old visible version. A post-publication cleanup
failure is a failed/incomplete replacement requiring retry or recovery, not a
successful transaction.

### Dual-index replacement

Main-index and TOC ownership are snapshotted and handled coherently. Main
publication precedes TOC publication; after publication both stores are
enumerated and only source-scoped stale IDs are removed from each store.
TOC publication and cleanup are not distributed atomic operations.

### GraphRAG replacement

For the canonical harness, the new graph publication occurs before stale
graph chunk unlink. Shared entities and relationships remain when valid
active evidence from another source supports them. Stale-only evidence is
removed or made inactive before it can influence canonical retrieval.

## 4. Failure and atomicity semantics

The lifecycle is deliberately **not called transactional**. It does not
guarantee:

- a distributed transaction across vector, TOC and graph stores;
- exactly-once ingestion;
- automatic rollback of every partial publication; or
- a cross-store atomic commit.

Partial vector, TOC or GraphRAG publication can remain after failure and
requires explicit retry/recovery. A provider without exact ownership lookup
must fail closed for changed-source replacement rather than append blindly.

## 5. Concurrency, generations and visibility

### Three distinct controls

1. **Source operation lease** controls which operation owns the replacement
   lifecycle.
2. **Publication generation** controls which prepared version is active.
3. **Retrieval visibility** filters inactive or unresolved generations before
   results are exposed.

The canonical source operation key is exactly:

```text
tenant_id + namespace + workspace_id + source_id
```

The default coordinator is process-local and thread-safe. It does not provide
multi-worker or multi-process safety. Production composition uses a durable
`ConditionalDocumentStore` CAS-backed lease/coordinator.

A lease records owner/token/expiry for publish, release and cleanup. A lease
alone cannot fence a backend write that was already in flight when the lease
expired. Therefore every replacement receives a publication generation; a
newer generation can supersede an older one. Vector and TOC reads filter by
the exact active generation. Stale physical records may exist temporarily,
but remain inactive/non-queryable and reclaimable.

The same generation-aware evidence rule applies to canonical harness GraphRAG
nodes, edges and chunk evidence. Traversal ignores inactive or unresolved
versioned evidence and retains shared graph facts supported by another active
generation. This graph evidence fence is qualified for
`InMemoryGraphStore`; live Neo4j publication fencing and live backend
reingest remain outside DOCS-11.

## 6. Provider capability and qualification

The matrix follows the catalog taxonomy and describes the native ABI, not
live service proof.

| Provider | Catalog status | Source replacement | Evidence |
|---|---|---|---|
| Qdrant | **STABLE** | supported by native contract | **QUALIFIED_OFFLINE_CONTRACT** |
| PgVector | **STABLE** | supported by native contract | **QUALIFIED_OFFLINE_CONTRACT** |
| Chroma | **STABLE** | supported by native contract | **QUALIFIED_OFFLINE_CONTRACT** |
| Weaviate | **BETA** | **UNSUPPORTED_FOR_SOURCE_REPLACEMENT** | no live claim |
| LanceDB | **BETA** | **UNSUPPORTED_FOR_SOURCE_REPLACEMENT** | no live claim |
| Typesense | **BETA** | **UNSUPPORTED_FOR_SOURCE_REPLACEMENT** | no live claim |
| Pinecone | **BETA** | **UNSUPPORTED_FOR_SOURCE_REPLACEMENT** | no live claim |
| Milvus | **BETA** | **UNSUPPORTED_FOR_SOURCE_REPLACEMENT** | no live claim |
| Vespa | **BETA** | **UNSUPPORTED_FOR_SOURCE_REPLACEMENT** | no live claim |
| InMemory | **BETA** (catalog taxonomy) | harness use only | canonical test harness |

`QUALIFIED_OFFLINE_CONTRACT` means that native records, scope, logical IDs
and exact ownership behavior are covered by offline/fake-provider contract
evidence. `LIVE_QUALIFIED` means accepted live service evidence; **none of
the providers above is LIVE_QUALIFIED after RAG-FINAL-10D**.

`BETA` means the catalog supports an adapter or capability under qualification
limits; it does not promote the provider to stable or prove source
replacement. `UNSUPPORTED_FOR_SOURCE_REPLACEMENT` requires fail-closed
behavior for changed sources, not append behavior.

## 7. Canonical current-state qualification matrix

| Capability | Status | Evidence level | Remaining limitation |
|---|---|---|---|
| Native recursive chunking | Qualified | native harness | provider-specific strategies remain optional |
| Native E2E ingest → retrieval | Qualified | native offline gate | not a live provider qualification |
| Dense retrieval | Qualified | native contract/harness | provider SLOs remain outside this gate |
| Hybrid retrieval | Qualified | native in-memory/harness | backend-specific live parity is not claimed |
| Hierarchical retrieval | Qualified | explicit parent-child + TOC gate | no parent-content reconstruction promise |
| Dual-index reingest | Qualified with limitations | RAG-FINAL-10A | non-atomic TOC/vector publication |
| GraphRAG canonical harness | Qualified with limitations | canonical harness | `InMemoryGraphStore` scope |
| GraphRAG source replacement | Qualified with limitations | generation-aware harness | live Neo4j reingest/fencing not qualified |
| Same-source serialization | Qualified with limitations | source-key coordinator | default coordinator is process-local |
| Publication generation fencing | Qualified with limitations | vector/TOC + graph harness | stale physical records need reclamation |
| Source ownership | Qualified | exact scoped enumeration | providers without lookup fail closed |
| Stable vector providers | Offline contract only | Qdrant/PgVector/Chroma | no LIVE_QUALIFIED provider |
| Namespace/workspace isolation | Contract-qualified | native scope/harness and offline Qdrant proof | live Qdrant isolation not claimed |
| Plugins | Qualified extension surface | native registry/plugin gate | authoring guide is RAG-DEV-12 |
| LangChain optionality | Qualified architecture | native ABI and boundary docs | optional compatibility paths remain |

## 8. Live-claim boundary and roadmap

DOCS-11 explicitly does **not** claim:

- live Qdrant tenant/namespace/workspace isolation;
- a live stable-provider source-replacement lifecycle;
- live Neo4j GraphRAG publication fencing or reingest qualification;
- a live GraphRAG backend qualification;
- transactional or exactly-once source replacement.

The global status remains **PRODUCTION_QUALIFIED_WITH_LIMITATIONS**.
RAG-PROD-13/14 decide whether any live or full-production claim can be
raised. RAG-DEV-12 owns the future plugin/developer guide. No DOCS-11 text
starts that work.

## 9. LangChain boundary

The canonical core RAG ABI and native path are Intergrax-native and do not
require LangChain. LangChain may remain as an optional provider,
compatibility implementation, or specific loader/splitter/embedding adapter
behind explicit plugin/compatibility boundaries. The correct claim is not
that Intergrax contains no LangChain code; the correct claim is that core RAG
contracts and the canonical native path do not require it.

## 10. Qualification evidence boundary

The accepted evidence is offline/contract and canonical-harness evidence from
RAG-FINAL-10A–10D. Runtime suites are not repeated by this documentation-only
task. This document records what the system does, what is qualified, what is
offline-only, what is beta, and what remains for PROD-13/14.
