# RAG - pipelines detail

**Parent hub:** [`RAG.md`](../RAG.md)
**Authority:** Current architecture, provider status and qualification are
owned exclusively by the parent hub. This satellite is technical pipeline
detail and is not live-provider or full-production evidence.
**Current global status:** **PRODUCTION_QUALIFIED_WITH_LIMITATIONS**

## Pipeline shape

```text
INGEST
  source
  → parser/loader
  → scoped KnowledgeDocument
  → native chunker and metadata enrichment
  → embedding provider
  → VectorStoreRecord / vector backend
  → optional TOC index and canonical-harness GraphRAG publication

RETRIEVE
  query + VectorStoreScope
  → RetrievalService
  → vector, lexical/hybrid, hierarchical or graph strategy
  → optional reranker
  → visibility filtering and citations
```

The native path does not require LangChain. LangChain loaders, splitters or
embeddings may be selected only as explicit optional provider/compatibility
implementations behind the normal plugin boundaries.

## Module-level responsibilities

| Area | Responsibility |
|---|---|
| Profile/bootstrap | Compose `RagProfile`, providers and `RetrievalService` |
| Loader/parser | Produce scoped `KnowledgeDocument` values |
| Chunking | Produce native derivative documents and preserve lineage |
| Embedding | Return ordered vectors without changing document identity |
| Indexing | Validate scope, IDs and cardinality before provider writes |
| Vector store | Implement native records, hits and exact ownership enumeration |
| Retrieval | Apply vector/hybrid/hierarchical/graph strategies and visibility |
| GraphRAG | Index and retrieve graph evidence linked to canonical vector IDs |
| Plugins | Extend parser, enricher, chunker, embedding, vector, retriever,
  reranker and supported graph-indexer surfaces |

## Chunking and indexing

Native recursive chunking is the core baseline. Semantic and parent-child
strategies are selectable. Hierarchical retrieval uses `DualIndexStrategy`:
child chunks belong to the main index and section names belong to a TOC/section
index. The TOC is not a full parent-document reconstruction.

The native indexing boundary is:

```text
KnowledgeDocument chunks
  → scope/lineage validation
  → native embedding result
  → VectorStoreRecord
  → provider-native payload
```

The portable logical ID is always `VectorStoreRecord.vector_id`. The invariant
is `ADD IDs == QUERY.vector_id == OWNERSHIP IDs == DELETE INPUT IDs`.
Provider physical IDs are internal and cannot substitute for the logical ID.

## Retrieval modes

- `vector_similarity`: dense retrieval.
- `hybrid`: dense plus lexical/sparse evidence.
- `hierarchical`: TOC relevance combined with child retrieval.
- `fusion`: combines registered retrieval channels.
- `graph_rag`: vector seed, graph traversal and channel fusion.
- reranking and agentic/query-expansion strategies remain explicit profile
  choices, not autonomous MIME-based algorithm selection.

All retrieval paths receive an explicit `VectorStoreScope` containing
`tenant_id`, `namespace` and `workspace_id`. Scope is system-owned and cannot
be overridden by user metadata.

## Source ownership and replacement detail

The ownership selector is:

```text
tenant_id + namespace + workspace_id + provenance.source_id
```

`list_source_record_ids(source_id, scope)` enumerates exact persisted records
owned by that source in that scope. It is not semantic search, top-k retrieval
or basename lookup. Equal basenames do not imply equal ownership.

Single-index replacement is:

```text
old ownership snapshot
  → prepare
  → publish current version
  → determine current ownership
  → stale = old - current
  → cleanup stale
```

Dual-index replacement snapshots and reconciles both main and TOC ownership.
GraphRAG publication in the canonical harness occurs before stale graph
unlink. No lifecycle uses delete-before-publish.

These operations are not distributed transactions and do not guarantee
exactly-once ingestion or automatic rollback of every partial publication.
Retry/recovery is required after partial vector, TOC or graph publication.

## Lease, generation and visibility detail

A source operation key is exactly:

```text
tenant_id + namespace + workspace_id + source_id
```

A source operation lease controls lifecycle ownership. The default coordinator
is process-local and thread-safe; it does not coordinate multiple workers or
processes. Production composition uses a CAS-backed durable
`ConditionalDocumentStore` coordinator.

A lease cannot fence a backend write already in flight when the lease expires.
Each replacement therefore receives a publication generation. A newer
publication supersedes an older one; vector and TOC reads expose only the
active generation. Stale physical records can remain temporarily, but are
inactive/non-queryable and reclaimable.

Canonical-harness GraphRAG nodes, edges and chunk evidence carry generation
context. Traversal ignores inactive or unresolved versioned evidence while
preserving shared evidence supported by another active generation. This
fencing is qualified for `InMemoryGraphStore`; live Neo4j publication fencing
and live backend reingest are not claimed.

## Qualification boundary

The parent hub owns the single provider matrix and capability matrix. In
particular, after RAG-FINAL-10D:

- stable: Qdrant, PgVector, Chroma - `QUALIFIED_OFFLINE_CONTRACT`;
- beta: Weaviate, LanceDB, Typesense, Pinecone, Milvus, Vespa and InMemory
  according to catalog taxonomy;
- `LIVE_QUALIFIED`: none;
- beta providers are not qualified for source replacement and must fail
  closed rather than append a changed source.

This satellite contains no independent live-provider, tenant-isolation or
Neo4j qualification claim. Future plugin authoring belongs to RAG-DEV-12.
