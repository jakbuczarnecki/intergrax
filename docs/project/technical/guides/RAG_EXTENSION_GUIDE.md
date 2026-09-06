# RAG Extension Guide

**Status:** canonical developer guide · **RAG-DEV-12** · **PLATFORM-PLUGIN-DOCS-4**
**Architecture owner:** [`docs/project/architecture/RAG.md`](../../architecture/RAG.md)
**Document ABI:** [`LANGCHAIN_INDEPENDENCE_native_document_contract.md`](../../capabilities/architecture/satellites/LANGCHAIN_INDEPENDENCE_native_document_contract.md)
**Integration architecture:** [`INTEGRATIONS.md`](../../architecture/INTEGRATIONS.md)
**Platform catalog:** [`EXTENSION_AUTHOR_GUIDE.md`](EXTENSION_AUTHOR_GUIDE.md)

This guide describes how to extend Intergrax RAG without modifying RAG core.
The design is **batteries included, but not locked to Intergrax algorithms**:
an external implementation may use LangChain, another vendor SDK, or entirely
independent code, provided that it enters through the contract and composition
boundary documented here.

This is an authoring guide, not a provider qualification claim. Provider status
and live/offline evidence remain in [`RAG.md`](../../architecture/RAG.md).

---

## Developer journey - chunker / retriever / reranker (D1–D16)

The three public RAG entry-point surfaces share discovery semantics but differ in contracts and DI. Scores below apply to the **external-EP author path**.

### Chunker (`intergrax.rag.chunkers`)

| D | Topic | Status | Section |
|---|-------|--------|---------|
| D1 Purpose | COMPLETE | §0, §4 |
| D2 Public contract | COMPLETE | §4 - `BaseChunkingStrategy` |
| D3 Minimal implementation | COMPLETE | §4 |
| D4 External package | COMPLETE | §15 |
| D5 Local / host path | COMPLETE | §0.2 - advanced composition only |
| D6 Configuration | COMPLETE | §0.3 - `RagProfile.chunking_strategy_id` |
| D7 Secrets | N/A | |
| D8 DI | COMPLETE | No-arg EP constructor; `register_chunking_strategy_plugin` for composition |
| D9 Registration/discovery | COMPLETE | §1, §0.1 |
| D10 Qualification | COMPLETE | §0.4, §16 |
| D11 Runtime use | COMPLETE | §0.3 |
| D12 Lifecycle | N/A | |
| D13 Failure behavior | COMPLETE | §1, §17 |
| D14 Testing | COMPLETE | §16 |
| D15 Production checklist | COMPLETE | §16 |
| D16 Troubleshooting | COMPLETE | §17 |

**Overall: COMPLETE** (external-EP path)

### Retriever (`intergrax.rag.retrievers`)

| D | Topic | Status | Section |
|---|-------|--------|---------|
| D1–D4 | Same pattern as chunker | COMPLETE | §5, §15 |
| D5 Local / host path | COMPLETE | §0.2 |
| D6 Configuration | COMPLETE | `RagProfile.retriever_id`, `fast_retriever_id`, `deep_retriever_id` |
| D8 DI | COMPLETE | `BaseRetrieverPlugin.create(...)` |
| D9–D16 | Same pattern as chunker | COMPLETE | §1, §16–§17 |

**Overall: COMPLETE** (external-EP path)

### Reranker (`intergrax.rag.rerankers`)

| D | Topic | Status | Section |
|---|-------|--------|---------|
| D1–D4 | Same pattern as chunker | COMPLETE | §6, §15 |
| D5 Local / host path | COMPLETE | §0.2 |
| D6 Configuration | COMPLETE | `RagProfile.reranker_id`, `enable_rerank` |
| D8 DI | COMPLETE | `BaseRerankerPlugin.create(embedding_manager=...)` |
| D9–D16 | Same pattern as chunker | COMPLETE | §1, §16–§17 |

**Overall: COMPLETE** (external-EP path)

---

## 0. Shared RAG model and platform truths

### Pipeline stages (what each plugin owns)

```text
documents
  → chunker (BaseChunkingStrategy)     # index time
  → embed + vector index
  → retriever (BaseRetriever)          # query time - candidate set
  → reranker (BaseReranker)            # optional - ranked context
  → Context Engineering (builtin.rag)  # not a RAG EP surface
```

Chunkers transform `KnowledgeDocument` sequences. Retrievers map `RetrieverQuery` → `RetrievalHit`. Rerankers reorder `RerankerCandidate` → `RerankerResult`. Vector backends use Integration Library - not `intergrax.rag.*` EP groups.

### Discovery and trust (all Platform Plugin surfaces)

- `installed` ≠ `discovered` ≠ `enabled` ≠ `production-qualified`
- Third-party plugins are **trusted in-process Python**
- Qualification is **host-owned semantic approval**, not attestation - not automatic across every RAG component surface
- Secrets stay in host/integration configuration - not EP values
- No universal Platform Plugin lifecycle/unload manager
- Shared discovery primitives apply where EP groups exist; typed `DomainPluginLoadReport` / cross-surface RAG evidence in `ApplicationPlatformPluginEvidence` is a **future maturity** target - per-component registries and bootstrap remain domain-owned today

### 0.1 Discovery timing (chunker / retriever / reranker)

RAG discovery is opt-in: `discover_entry_points=True` on bootstrap functions or `INTERGRAX_DISCOVER_PLUGINS=true`. Built-in strategies register first; EP plugins append. Duplicate component IDs fail in native registries (`ValueError`). See §1 for conflict policy details.

### 0.2 Local / host path - classification

RAG local authoring is **not** equivalent to Tools scaffold + `register_tool_plugin()`. Classify as **external-EP-first** with optional **advanced host composition**:

| Surface | Advanced host composition (trusted code) | Canonical author path |
|---------|------------------------------------------|------------------------|
| Chunker | `register_chunking_strategy_plugin(strategy_id, factory)` then `create_default_chunking_engine(registry=custom)`; or `ChunkingStrategyRegistry.register(instance)` | External EP `intergrax.rag.chunkers` |
| Retriever | `RetrieverRegistry.register(retriever)` passed to `create_default_retriever_manager(registry=...)` | External EP `intergrax.rag.retrievers` |
| Reranker | `RerankerRegistry.register(reranker)` passed to `create_default_reranker_engine(registry=...)` | External EP `intergrax.rag.rerankers` |

Do not document these composition APIs as a portable local plugin framework. They require host-owned bootstrap code in the same process.

### 0.3 Runtime path - RagProfile to active component

```text
ApplicationEnvironmentProfile
  → resolve_rag_profile_for_environment(env)     # may return None when enable_rag=False
  → create_default_rag_stack(..., profile=rag_profile)
      → create_default_chunking_engine()           # selects chunking_strategy_id at ingest
      → create_default_retriever_manager()         # registers retriever_id / fast / deep
      → create_default_reranker_engine()           # registers reranker_id when enable_rerank
  → RagStack on ToolWiringContext / RuntimeConfig
  → RetrievalService.retrieve(RetrievalRequest)    # uses profile retriever + reranker ids
```

Key `RagProfile` selectors:

| Field | Selects |
|-------|---------|
| `chunking_strategy_id` | `ChunkingStrategyRegistry` entry (default `recursive`) |
| `retriever_id` | Primary retriever (default `hybrid`) |
| `fast_retriever_id` / `deep_retriever_id` | Adaptive routing pair |
| `reranker_id` | Reranker when `enable_rerank=True` (default `embedding_cosine`) |
| `enable_rerank` | Rerank stage on/off |
| `prefetch_top_k` / `final_top_k` | Retrieval limits |

Tier-3 hosts wire via `intergrax.applications._shared.rag_runtime_bridge.resolve_rag_stack_for_environment`.

### 0.4 Qualification layers (do not merge)

| Layer | Meaning |
|-------|---------|
| Contract validity | Plugin implements `BaseChunkingStrategy` / `BaseRetriever` / `BaseReranker` (or plugin bases) |
| Platform production qualification | Host `require_production_qualification` gates - compatible ≠ qualified |
| Domain / live backend qualification | Separate evidence for vector stores, live PgVector/Chroma/Neo4j, etc. - see [`RAG.md`](../../architecture/RAG.md) qualification records |

External EP registration does not grant `LIVE_QUALIFIED` or `STABLE` provider status.

---

## 1. Extension topology: audit result

The eight surfaces do not share one registration mechanism. The category in the
second column is intentional:

- **A - `PUBLIC_EXTERNAL_PLUGIN`**: an installed external package can register
  without changing Intergrax core.
- **B - `INTERNAL_REGISTRY_EXTENSION`**: the contract is pluggable, but the
  registration currently happens in composition or Intergrax-owned bootstrap.
- **C - `INTEGRATION_CATALOG_PROVIDER`**: the provider is resolved by
  Integration Library, not by a RAG component entry-point group.
- **D - `NOT_PUBLICLY_EXTENSIBLE`**: no supported external path was found.

| Surface | Category | Contract | Registration/discovery | Built-ins and selector | Runtime supplied by Intergrax |
|---|---|---|---|---|---|
| Loader | **B** | `BaseDocumentsLoader.load_document/load_documents` | Inject a loader into the ingest composition; no RAG EP | `create_default_documents_loader`; handler registry | authoritative `tenant_id`, optional `namespace`, metadata callback |
| Parser | **C** | Integration `DocumentParser`; parser output is normalized to `KnowledgeDocument` | Integration Library, category `DOCUMENT_PARSER`; external catalog plugins use `intergrax.integrations` | parser slug in `IntegrationProfile`/preset; `resolve_document_parser(slug)` | parser options; scope is supplied by the normalization/loader boundary |
| Metadata enricher | **B** | `BaseMetadataProvider.enrich(documents, source)` | `MetadataPipeline(providers=...)` or loader/ingest callback; no public RAG EP | `DefaultMetadataProvider` | native documents and source; no authority to change routing fields |
| Chunker | **A** | `BaseChunkingStrategy.strategy_id()` and `chunk(Sequence[KnowledgeDocument])` | `intergrax.rag.chunkers`; discovery is opt-in | built-ins plus `RagProfile.chunking_strategy_id` | no runtime infrastructure by default; constructor is no-argument for EP discovery |
| Embedding | **C** | `EmbeddingProvider.provider_name/dimension/embed` | Integrations catalog `embedding_provider` + `bind_embedding_provider()`; inject bound provider into `EmbeddingEngine`/`EmbeddingPipeline`; no `intergrax.rag.embeddings` EP | `IntegrationProfile.embedding_provider` / `EmbeddingProfile` env | embedding engine/manager is composed by the host |
| Vector backend | **C** | native `VectorStore` plus `VectorStoreRecord/Hit/Scope` ABI | Integration Library, category `VECTOR_STORE`; resolved and wrapped by `VectorstoreManager` | `IntegrationProfile.vector_store` / environment / preset | scope and lifecycle stay at the native manager boundary |
| Retriever | **A** | `BaseRetriever` or dependency-aware `BaseRetrieverPlugin`; `RetrieverQuery` → `RetrievalHit` | `intergrax.rag.retrievers`; discovery is opt-in | `RagProfile.retriever_id`, `fast_retriever_id`, `deep_retriever_id` | vector manager, embedding manager, optional TOC/graph store, profile and LLM |
| Reranker | **A** | `BaseReranker` or `BaseRerankerPlugin`; candidates → `RerankerResult` | `intergrax.rag.rerankers`; discovery is opt-in | `RagProfile.reranker_id`; some built-ins use Integration Library rerank providers | embedding manager is supplied to `BaseRerankerPlugin.create`; cache/engine are composed |
| Graph indexer | **B** | `GraphIndexer`/`GraphIndexerPlugin.index_documents` | process registry via `register_graph_indexer_plugin`; no public EP | `RagProfile.graph_indexer_plugin_id` or built-in mode | `GraphStore`, `RagProfile`, optional `LLMAdapter`, canonical `chunk_ids` |

The public RAG entry-point groups are exactly:

```text
intergrax.rag.chunkers
intergrax.rag.retrievers
intergrax.rag.rerankers
```

There is currently no public `intergrax.rag.embeddings` group. Do not invent
one in a plugin package.

### Discovery timing and conflicts

RAG discovery is opt-in. It is enabled with
`discover_entry_points=True` at the relevant bootstrap or with
`INTERGRAX_DISCOVER_PLUGINS=true`. The bootstrap first loads the entry-point
specification and imports its target. The target may be a class or a callable
that returns a plugin class. The component instance is then constructed by the
RAG bootstrap:

- chunkers: `plugin_type()`; therefore an EP chunker must have a no-argument
  constructor;
- retrievers: `BaseRetrieverPlugin.create(...)` receives runtime resources, or
  a `BaseRetriever` is constructed with no arguments;
- rerankers: `BaseRerankerPlugin.create(embedding_manager=...)`, or a
  no-argument `BaseReranker`.

Imports are not lazy at the RAG EP boundary: importing the selected plugin
module occurs during discovery. Optional vendor libraries should therefore be
owned by the plugin package and imported only when the plugin is selected or
constructed. A malformed target or import failure is wrapped in
`PluginLoadError`; the target must be a class or a factory returning one.

Duplicate EP names fail by default with `PluginConflictError`. A duplicate
component identifier then fails in the native registry, for example
`ValueError: Retriever already registered: ...`, `Reranker already registered:
...`, or `Chunking strategy already registered: ...`. Do not rely on overriding
a built-in ID.

## 2. Runtime dependency model

There is no generic service locator, hidden global container, or automatic
dependency injection. Dependencies enter through the bootstrap signatures and
the plugin construction protocols.

### Intergrax provides

- native `KnowledgeDocument`, scope, identity and provenance;
- `RetrieverQuery`, `RetrievalHit`, reranker candidate/result types;
- `VectorStoreScope`, native records/hits and the portable logical ID;
- the composed `VectorstoreManager`, `BaseEmbeddingManager`, optional TOC and
  graph stores, profile and optional LLM where the plugin contract exposes them;
- registry selection, lifecycle coordination, publication visibility filtering
  and normal error/trace boundaries.

### The plugin owns

- its algorithm and deterministic transformation/ranking behavior;
- vendor SDK imports and optional dependencies;
- configuration validation, credentials, endpoints, model selection and API
  limits;
- network timeouts, bounded retries and provider-specific error translation;
- any cache or client state that is not part of the supplied contract.

A chunker should remain document transformation logic. It should not reach into
the vector store, global application state or a secret manager. A retriever
that needs runtime resources must use `BaseRetrieverPlugin`, not a hidden
singleton. A reranker may use the supplied embedding manager when its strategy
needs it, but must accept the exact construction signature.

## 3. Non-negotiable native invariants

Every extension must preserve the following ABI, regardless of its algorithm.

### `KnowledgeDocument`

`KnowledgeDocument` is the shared document ABI. Preserve:

- `identity.document_id` and root/parent lineage;
- `scope.tenant_id`, `scope.namespace` and `scope.workspace_id`;
- `provenance.source_id` and other provenance semantics;
- content meaning and `schema_version` (currently `1`);
- reserved metadata rules.

System-owned scope and provenance must not be transported through arbitrary
user metadata. Metadata enrichment may add business fields, but it cannot
override canonical routing or source ownership. A plugin must not replace a
native document with a LangChain `Document` at a RAG contract boundary.

### `VectorStoreScope`

Every vector operation is scoped by the exact triple:

```text
tenant_id + namespace + workspace_id
```

Missing or invalid scope must fail closed. A user metadata filter is not a
scope substitute and cannot supply reserved routing keys as ordinary conditions.

### Portable logical vector ID

`VectorStoreRecord.vector_id` is the logical persisted ID:

```text
ADD IDs == QUERY.vector_id == OWNERSHIP IDs == DELETE INPUT IDs
```

Provider physical IDs may be mapped internally, but must never cross the native
ABI or become ownership/delete fallbacks.

### Source ownership and replacement

A provider claiming changed-source replacement must implement exact ownership:

```python
list_source_record_ids(source_id, scope)
```

The result must enumerate all owned logical IDs in that exact scope. Semantic
search, top-k results, basename matching or path heuristics are not ownership.
Without exact enumeration, the provider must fail closed for source replacement.

Visibility and same-source replacement remain governed by the shared
`VectorstoreManager` and `SourceOperationCoordinator` where applicable. Plugin
code must not invent a competing publication-generation model. The default
coordinator is process-local and thread-safe; it is not distributed safety.
Cross-store publication is not transactional or exactly-once.

## 4. Chunker authoring

### Contract

Implement `BaseChunkingStrategy`:

```python
from collections.abc import Sequence

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_splitters.chunk_document import build_derived_chunk
from intergrax.rag.document_splitters.contracts.base_chunking_strategy import (
    BaseChunkingStrategy,
)


class ParagraphChunker(BaseChunkingStrategy):
    def __init__(self, max_chars: int = 800) -> None:
        if type(max_chars) is not int or max_chars <= 0:
            raise ValueError("max_chars must be a positive int")
        self._max_chars = max_chars

    @classmethod
    def strategy_id(cls) -> str:
        return "my_paragraph"

    def chunk(
        self,
        documents: Sequence[KnowledgeDocument],
    ) -> Sequence[KnowledgeDocument]:
        output: list[KnowledgeDocument] = []
        for document in documents:
            paragraphs = [
                paragraph.strip()
                for paragraph in document.content.split("\n\n")
                if paragraph.strip()
            ]
            chunk_index = 0
            for paragraph in paragraphs:
                for offset in range(0, len(paragraph), self._max_chars):
                    part = paragraph[offset : offset + self._max_chars]
                    output.append(
                        build_derived_chunk(
                            document,
                            content=part,
                            strategy_id=self.strategy_id(),
                            chunk_index=chunk_index,
                        )
                    )
                    chunk_index += 1
        return tuple(output)
```

The example accepts and returns native documents. `build_derived_chunk` is the
identity/lineage helper: it creates deterministic derived identity while
preserving scope, provenance and source lineage. Keep input ordering stable,
do not silently rewrite content, and do not derive routing data from metadata.
The constructor validates its configuration. Because EP discovery calls
`plugin_type()` with no arguments, EP configuration must be represented by
safe package defaults or supplied through an explicit composition registry.

Register the class in the exact current group:

```toml
[project.entry-points."intergrax.rag.chunkers"]
my_paragraph = "my_intergrax_plugins.chunker:ParagraphChunker"
```

Select it with `RagProfile(chunking_strategy_id="my_paragraph")` or the normal
profile/environment wiring. Discovery must be enabled by the host.

### Optional LangChain wrapper

LangChain is optional. A wrapper may use a LangChain splitter internally, but
the boundary remains native:

```text
KnowledgeDocument
  → LangChain splitter (inside the plugin)
  → text parts
  → build_derived_chunk(...)
  → KnowledgeDocument chunks
```

Install `langchain-text-splitters` as a dependency of the plugin (or use the
Intergrax optional compatibility extra where appropriate). LangChain types must
not leak into `BaseChunkingStrategy`, ingest, vector or retrieval contracts.

## 5. Retriever authoring

Use `BaseRetriever` for a no-dependency strategy. Use `BaseRetrieverPlugin` when
the strategy needs the vector manager, embedding manager or another runtime
resource:

```python
from collections.abc import Sequence

from intergrax.rag.embedding.contracts.base_embedding_manager import (
    BaseEmbeddingManager,
)
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    BaseRetrieverPlugin,
    RetrievalHit,
    RetrieverQuery,
)
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import (
    BaseVectorstoreManager,
)


class DenseExternalRetriever(BaseRetriever):
    def __init__(
        self,
        vector_store: BaseVectorstoreManager,
        embedding_manager: BaseEmbeddingManager,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_manager = embedding_manager

    @classmethod
    def name(cls) -> str:
        return "my_dense"

    def retrieve(self, query: RetrieverQuery) -> Sequence[RetrievalHit]:
        if query.scope is None:
            raise ValueError("retriever requires an explicit VectorStoreScope")
        embedding = query.query_embedding
        if embedding is None:
            embedding = self._embedding_manager.embed_one(query.query_text)
        native_hits = self._vector_store.query(
            embedding,
            scope=query.scope,
            top_k=query.top_k,
            metadata_filter=query.metadata_filter,
            include_embeddings=query.include_embeddings,
        )
        return tuple(
            RetrievalHit.from_vector_store_hit(
                hit,
                channel="dense",
                retriever_name=self.name(),
            )
            for hit in native_hits
        )


class DenseExternalRetrieverPlugin(BaseRetrieverPlugin):
    @classmethod
    def create(
        cls,
        *,
        vector_store: BaseVectorstoreManager,
        embedding_manager: BaseEmbeddingManager,
        graph_store=None,
        profile=None,
        llm_for_query_expansion=None,
        toc_vector_store=None,
    ) -> BaseRetriever:
        del graph_store, profile, llm_for_query_expansion, toc_vector_store
        return DenseExternalRetriever(vector_store, embedding_manager)
```

The bootstrap supplies exactly the dependencies shown in
`BaseRetrieverPlugin.create`: vector manager, embedding manager, optional TOC
and graph managers, profile and optional query-expansion LLM. The plugin must
return a `BaseRetriever`.

`RetrievalHit` validates a finite score in `[0, 1]`, a non-negative rank, a
non-empty channel and native document identity. Use
`RetrievalHit.from_vector_store_hit` when starting from a native hit. Preserve
`hit.vector_id`; do not expose a provider physical ID.

Use the existing `VectorstoreManager` when the needed operation is already
available. Implement a retriever when the ranking/query strategy is new.
Implement a vector backend only when the storage and native add/query/delete
semantics are new; do not import provider internals from a retriever to avoid
the portable manager.

Register:

```toml
[project.entry-points."intergrax.rag.retrievers"]
my_dense = "my_intergrax_plugins.retriever:DenseExternalRetrieverPlugin"
```

Select with `RagProfile(retriever_id="my_dense")` or the corresponding fast/deep
selector. A plugin that implements only `BaseRetriever` is instantiated without
runtime dependencies and must therefore own no hidden vector/embedding state.

## 6. Reranker authoring

Candidates are native `RerankerCandidate` values. Their `document` is the
canonical identity/scope/provenance object; `original_score` is the normalized
retrieval score. Return `RerankerResult` values, ordered by final rank:

```python
from collections.abc import Sequence

from intergrax.rag.embedding.contracts.base_embedding_manager import (
    BaseEmbeddingManager,
)
from intergrax.rag.rerankers.contracts.base_reranker import (
    BaseReranker,
    BaseRerankerPlugin,
)
from intergrax.rag.rerankers.contracts.reranker_types import (
    RerankerCandidate,
    RerankerResult,
)


class LengthReranker(BaseReranker):
    @classmethod
    def name(cls) -> str:
        return "my_length"

    def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[RerankerCandidate],
        limit: int | None = None,
    ) -> Sequence[RerankerResult]:
        del query
        ordered = sorted(
            tuple(candidates),
            key=lambda candidate: len(candidate.document.content),
            reverse=True,
        )
        if limit is not None:
            ordered = ordered[:limit]
        return tuple(
            RerankerResult(
                candidate=candidate,
                rerank_score=float(len(candidate.document.content)),
                rank=rank,
            )
            for rank, candidate in enumerate(ordered)
        )


class LengthRerankerPlugin(BaseRerankerPlugin):
    @classmethod
    def create(cls, *, embedding_manager: BaseEmbeddingManager) -> BaseReranker:
        del embedding_manager
        return LengthReranker()
```

The supplied embedding manager is available to a dependency-aware reranker,
even though this example does not use it. Output ordering and scores are owned
by the reranker; scores must be finite, and the candidate object/document
identity must not be replaced. External API ownership, authentication and
timeouts belong to the plugin/provider configuration, never to `KnowledgeDocument`.

Register:

```toml
[project.entry-points."intergrax.rag.rerankers"]
my_length = "my_intergrax_plugins.reranker:LengthRerankerPlugin"
```

Select with `RagProfile(reranker_id="my_length")`. Built-in Cohere/Jina
providers are a separate Integration Library provider selection path; that
does not change the RAG reranker EP contract.

## 7. Vector backend authoring

Vector stores use a different extension model. They are not registered through
`intergrax.rag.retrievers`.

1. Implement the native `VectorStore` port:
   `add_records`, `query`, `delete`, and `count`, with exact `VectorStoreScope`.
2. Return native `VectorStoreHit` values and normalized similarity scores.
3. Preserve logical `VectorStoreRecord.vector_id`; map it to a physical ID only
   inside the provider.
4. Register an Integration Library manifest with category `VECTOR_STORE` and a
   provider factory/bundle. External catalog packages use the
   `intergrax.integrations` IntegrationPlugin protocol; shipped providers use
   manifest/register/bundle modules.
5. Select the provider through `IntegrationProfile.vector_store`, a registered
   slug or the normal environment/preset path.
6. Let RAG resolve it with `create_vectorstore_manager(...)`; this wraps the
   native provider in `VectorstoreManager`.
7. Implement exact `list_source_record_ids(source_id, scope)` before claiming
   source replacement.

The IntegrationPlugin shape is:

```python
class MyVectorPlugin:
    @classmethod
    def integration_manifest(cls) -> IntegrationManifest:
        return IntegrationManifest(
            slug="my_vector",
            categories=(IntegrationCategory.VECTOR_STORE,),
            status=IntegrationStatus.BETA,
        )

    @classmethod
    def create_integration(cls, **kwargs: object) -> object:
        return MyVectorStoreIntegration.from_config(**kwargs)
```

External catalog registration uses:

```toml
[project.entry-points."intergrax.integrations"]
my_vector = "my_intergrax_plugins.vector:MyVectorPlugin"
```

`IntegrationProfile` accepts a catalog slug, manifest, plugin class or
pre-built integration instance. Catalog discovery must be enabled by the host.
The provider bundle owns configuration and client construction; the RAG manager
owns scope validation, logical ID semantics and lifecycle composition.

A backend may support core retrieval while being unable to support exact source
replacement. Do not claim that capability until ownership enumeration is exact.
Use the repository taxonomy:

```text
STABLE
BETA
QUALIFIED_OFFLINE_CONTRACT
LIVE_QUALIFIED
UNSUPPORTED_FOR_SOURCE_REPLACEMENT
```

External plugins do not receive automatic `STABLE` or `LIVE_QUALIFIED` status.
Refer to the provider matrix in `RAG.md`.

## 8. Embedding provider authoring

Embedding providers currently use a composition registry, not a public RAG EP.
Implement:

```python
from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray

from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider


class MyEmbeddingProvider(EmbeddingProvider):
    def provider_name(self) -> str:
        return "my_embeddings"

    def dimension(self) -> int:
        return 384

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        vectors = ...  # return one finite vector per input text
        array = np.asarray(vectors, dtype=np.float32)
        if array.shape != (len(texts), self.dimension()):
            raise ValueError("embedding cardinality or dimension mismatch")
        return array
```

Bind the provider through Integrations (`IntegrationProfile` + `bind_embedding_provider()`)
or inject a typed `EmbeddingProvider` instance at composition time, then build the normal
`EmbeddingEngine`/`EmbeddingPipeline`. `EmbeddingEngine` executes the already-bound
provider; it does not resolve providers from a registry.

The provider must return exactly one vector per input, in input order, with a
stable positive dimension and finite values. It must not mutate document
identity, scope or provenance. No `intergrax.rag.embeddings` entry-point group
exists today; an external package must expose a composition helper or be
registered by the application.

## 9. Parser and loader authoring

The boundary is deliberately two-stage:

```text
provider parser/extraction result
  → scoped normalization layer
  → KnowledgeDocument
```

`BaseDocumentsLoader.load_document` receives the authoritative `tenant_id` and
optional `namespace`. A parser must not mint fake tenant/workspace context
before that authority exists. It returns the provider extraction shape (for
example `ParsedDocumentFragment`) to the loader/normalizer, which creates the
native document and provenance.

Parser providers belong to Integration Library category `DOCUMENT_PARSER`.
Use `IntegrationPlugin`/`IntegrationManifest` and the
`intergrax.integrations` catalog path, then resolve by parser slug through
`resolve_document_parser(slug, **options)` or the normal IntegrationProfile
composition. `RagProfile.document_parser_slug` is populated by environment
parsing, but the bounded audit found no independent RAG EP or direct resolver
for that field; do not document it as a standalone plugin registry. Use the
Integration Library profile/preset path as the authoritative provider selector.

A completely custom loader is a composition object implementing
`BaseDocumentsLoader` and injected into `IngestPipeline(loader=...)`. A custom
metadata callback is also composition-time. Neither path is a public RAG
entry-point group.

## 10. Metadata enrichment

`BaseMetadataProvider` is a real contract, but it is not a first-class external
plugin catalog. The current supported mechanisms are:

- pass providers to `MetadataPipeline(providers=...)` when composing a loader;
- pass the supported custom metadata callback to the loader;
- pass `metadata_callback` to `IngestPipeline` where that composition API is used.

An enricher receives native documents and the original source and returns native
documents. Preserve identity, lineage, scope and provenance. Add user/business
metadata only in the user metadata area. Never override `tenant_id`,
`namespace`, `workspace_id`, `provenance.source_id`, identity fields or reserved
system metadata. If a new enricher needs a durable runtime capability rather
than document-local enrichment, it is an architecture discussion, not a
metadata plugin workaround.

## 11. GraphIndexer is not GraphStore

`GraphIndexer` transforms native documents/chunks into graph semantic evidence.
`GraphStore` is the persistence/traversal contract. They are separate extension
points:

```text
GraphIndexer: KnowledgeDocument/chunks → graph semantic evidence
GraphStore:   graph evidence → persistence/traversal
```

The current GraphIndexer extension is an internal registry:

```python
from intergrax.rag.graph.indexer.plugin_registry import (
    register_graph_indexer_plugin,
)


def create_my_indexer(store, profile, llm):
    return MyGraphIndexer(store=store, profile=profile, llm=llm)


register_graph_indexer_plugin("my_graph", create_my_indexer)
```

The factory receives `(GraphStore, RagProfile, LLMAdapter | None)`. The plugin
implements:

```python
def index_documents(
    self,
    documents: Sequence[KnowledgeDocument],
    *,
    chunk_ids: Sequence[str] | None = None,
) -> int:
    ...
```

Select it with `RagProfile(graph_indexer_plugin_id="my_graph")` or
`INTERGRAX_RAG_GRAPH_INDEXER_PLUGIN`. There is no public graph-indexer EP.
Registration currently requires trusted composition code, and duplicate IDs in
this process registry replace the existing factory; choose globally unique IDs.

Use the canonical chunk IDs passed by ingest. Never mint a second graph chunk
identity. Preserve source evidence where generation-aware lifecycle requires
it, and use the graph lifecycle hooks rather than bypassing source replacement.
`InMemoryGraphStore` evidence does not guarantee live Neo4j publication or
distributed fencing.

## 12. Plugin or architecture change?

### A plugin is appropriate when

- behavior fits an existing contract;
- identity and scope semantics remain unchanged;
- required runtime resources already exist in the composition boundary;
- no new lifecycle or durable coordination capability is required.

### An architecture change is required when

- a new system-owned identity or routing field is needed;
- new cross-store lifecycle semantics are required;
- the existing exact source ownership contract is insufficient;
- new durable coordination or fencing is needed;
- a secret/runtime resource is required but not exposed by supported composition;
- a new capability changes the canonical ABI or serialized schema.

Do not work around a missing architectural capability by using user metadata for
routing, global singletons, monkey-patching registries, provider-specific
imports in core, changing logical IDs, or bypassing `VectorstoreManager`.

## 13. Security and operational safety

Production plugins should:

- validate configuration before making network calls;
- fail closed on invalid or missing scope;
- never log secrets;
- never put secrets in `KnowledgeDocument.metadata` or provenance URIs;
- use bounded retries and timeouts for network providers;
- avoid arbitrary dynamic imports from user-provided module names;
- load plugins only from trusted installed packages and trusted configuration;
- make plugin failures explicit, observable and attributable.

Intergrax does not provide a plugin sandbox in this boundary. Python plugins
execute with the privileges of the hosting process unless stronger isolation is
provided by the deployment platform.

## 14. Versioning and compatibility

Plugins should depend on public Intergrax contracts, not private underscore
modules or provider implementation details. Pin and test against the supported
Intergrax version used by the host. Contract, identity and serialized-schema
changes require compatibility review; `KnowledgeDocument.schema_version` must
be respected.

Optional libraries belong in the plugin package dependencies, not in core
Intergrax dependencies. The repository currently exposes compatibility extras
for selected LangChain paths, but that does not turn LangChain into a core
requirement or impose it on independent plugins.

## 15. Minimal external package skeleton

Only the three genuinely public RAG EP surfaces belong in this package:

```text
my-intergrax-rag-plugins/
├── pyproject.toml
└── src/
    └── my_intergrax_plugins/
        ├── __init__.py
        ├── chunker.py
        ├── retriever.py
        └── reranker.py
```

`chunker.py`, `retriever.py` and `reranker.py` contain the implementations from
sections 4–6. A minimal `__init__.py` can export their public classes:

```python
from .chunker import ParagraphChunker
from .reranker import LengthRerankerPlugin
from .retriever import DenseExternalRetrieverPlugin

__all__ = [
    "DenseExternalRetrieverPlugin",
    "LengthRerankerPlugin",
    "ParagraphChunker",
]
```

The package metadata is:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-intergrax-rag-plugins"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = ["Intergrax-ai==0.1.0"]

[tool.hatch.build.targets.wheel]
packages = ["src/my_intergrax_plugins"]

[project.entry-points."intergrax.rag.chunkers"]
my_paragraph = "my_intergrax_plugins.chunker:ParagraphChunker"

[project.entry-points."intergrax.rag.retrievers"]
my_dense = "my_intergrax_plugins.retriever:DenseExternalRetrieverPlugin"

[project.entry-points."intergrax.rag.rerankers"]
my_length = "my_intergrax_plugins.reranker:LengthRerankerPlugin"
```

Update the pinned Intergrax version only after testing the package against that
contract version. A vector, parser or embedding provider is deliberately not
included in this skeleton because those surfaces use Integration Library or
composition registration.

## 16. Qualification checklist

Use the native contract tests as the model. Relevant existing evidence includes
[`test_rag_plugin_discovery.py`](../../../../tests/unit/rag/test_rag_plugin_discovery.py),
[`test_retrieval_integration.py`](../../../../tests/integration/rag/retrievers/test_retrieval_integration.py),
[`test_semantic_reranker.py`](../../../../tests/integration/rag/rerankers/test_semantic_reranker.py),
and
[`test_graph_indexer_plugin_registry.py`](../../../../tests/unit/rag/graph/test_graph_indexer_plugin_registry.py).

### Chunker

- deterministic output for the same input/configuration;
- content integrity and deterministic ordering;
- correct derived identity and root/parent lineage;
- exact scope and provenance preservation;
- invalid configuration fails before processing.

### Retriever

- exact `tenant_id`/namespace/workspace scope;
- stable ranking, finite `[0, 1]` scores and valid ranks;
- portable logical vector IDs;
- no foreign-scope leakage;
- no direct dependence on provider physical internals.

### Reranker

- candidate/document identity preservation;
- bounded output and correct limit behavior;
- finite, explicit scores and zero-based final ranks;
- deterministic behavior or explicit failure semantics where possible;
- external API timeout/auth failures are observable.

### Vector backend

- ADD/QUERY/OWNERSHIP/DELETE logical ID parity;
- exact scope on every operation;
- negative namespace/workspace isolation gate;
- same-basename source isolation;
- exact source ownership if replacement is claimed;
- fail-closed behavior when ownership is unsupported.

### Graph indexer

- canonical ingest `chunk_ids`;
- native `KnowledgeDocument` inputs;
- scope/tenant/workspace consistency;
- source replacement compatibility and generation-aware evidence;
- shared evidence preservation;
- no writes when batch validation fails.

## 17. Troubleshooting

### Plugin is not discovered

Check that the package is installed in the same environment as the host, the
entry-point group is one of the three exact RAG groups, and discovery is enabled
with `discover_entry_points=True` or `INTERGRAX_DISCOVER_PLUGINS=true`.
Discovery is opt-in by default.

### Duplicate plugin ID/name

Duplicate EP names raise `PluginConflictError`. Duplicate retriever/reranker/
chunker component IDs raise the corresponding registry `ValueError`. Rename the
plugin; do not silently shadow a built-in.

### Optional dependency is absent

For built-in embedding lazy factories, expect
`EmbeddingProviderDependencyError`. For an external EP, `PluginLoadError` may
wrap the import failure. Put the optional library in the plugin's dependency
metadata and import it at selection/construction time.

### Constructor or factory mismatch

Chunker EPs and plain `BaseRetriever`/`BaseReranker` EPs are constructed without
arguments. Use the dependency-aware plugin base when resources are required.
A mismatch is reported as an explicit `TypeError`; do not add a global client.

### Scope or logical-ID error

Construct and pass `VectorStoreScope` from authoritative document/query scope.
Use `VectorStoreRecord.vector_id` as the portable ID and pass it unchanged to
delete/ownership operations. `VectorStoreContractError` indicates malformed
native scope, record, hit, filter or vector data.

### Source replacement is unsupported

`vectorstore_source_record_lookup_not_supported` means exact ownership lookup is
not available. The backend may still support retrieval, but it must not claim
changed-source replacement or append blindly.

### Graph plugin cannot be resolved

`unknown_graph_indexer_plugin:<id>` means the factory was not registered before
RAG bootstrap resolved the profile. Register it in trusted composition code and
use the exact normalized plugin ID.

### LangChain import fails

LangChain is optional. Install the plugin's declared LangChain extra/library,
then keep the adapter boundary native. Do not change core contracts to accept
LangChain objects.

## 18. When to stop and request architecture work

Stop plugin implementation and open an architecture task when qualification
requires a new system-owned field, cross-store lifecycle guarantee, durable
coordination, secret/runtime resource not exposed by composition, or a changed
canonical ABI. The correct DEV-12 outcome in that case is
`CHANGES_REQUIRED_ARCHITECTURE`, not a workaround hidden in a plugin.
