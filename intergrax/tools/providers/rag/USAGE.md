# RAG tool bundle

**Bundle id:** `rag`  
**Tools:** `rag.retrieve`, `rag.ingest_document`, `rag.list_collections`, `rag.delete_documents`, `rag.describe_collection`

## Dependencies (`ToolWiringContext`)

| Field | Required | Purpose |
|-------|----------|---------|
| `vectorstore_manager` | Yes | Vector index for chunks |
| `embedding_manager` | Yes (retrieve/ingest) | Embeddings for query + ingest |
| `rag_profile` | No | `RagProfile` or env `INTERGRAX_RAG_*` (retriever, reranker, routing) |
| `retrieval_service` | No | Pre-built `RetrievalService`; else composed from managers + profile |
| `retriever_manager` / `reranker_manager` | No | Override Tier-0 registries |

Local backends: `infra/integration` profile `rag` (Qdrant, Chroma, Weaviate, Neo4j, Ollama, Docling) - see [infra/PORTS.md](../../../../infra/PORTS.md).

Tier-3 example (full stack):

```python
from intergrax.rag.bootstrap.rag_stack_bootstrap import create_default_rag_stack
from intergrax.tools.registry import ToolProfile, ToolWiringContext, build_registry_from_profile, register_default_tools

register_default_tools()
stack = create_default_rag_stack(integration_profile=integration_profile)
ctx = ToolWiringContext(
    vectorstore_manager=stack.vectorstore_manager,
    embedding_manager=stack.embedding_manager,
    retriever_manager=stack.retriever_manager,
    reranker_manager=stack.reranker_manager,
    rag_profile=stack.profile,
    retrieval_service=stack.retrieval_service,
    extras={"contextual_enricher": stack.contextual_enricher},
)
registry = build_registry_from_profile(
    ToolProfile(enabled=["rag.retrieve", "rag.ingest_document", "rag.list_collections"]),
    ctx=ctx,
)
```

### `rag.retrieve`

Uses **`RetrievalService`**: adaptive route tier → registered retriever (default `hybrid`) → optional reranker → scoped metadata filter. Output includes `diagnostics` (`retriever_id`, `route_tier`, latencies).

Env examples:

| Variable | Purpose |
|----------|---------|
| `INTERGRAX_RAG_RETRIEVER_ID` | Default retriever (`hybrid`, `fusion`, `graph_rag`, …) |
| `INTERGRAX_RAG_ENABLE_RERANK` | Cross-encoder / cosine rerank after retrieval |
| `INTERGRAX_RAG_ROUTE_MODE` | `auto` → fast/standard/deep tiers |
| `INTERGRAX_RAG_NATIVE_HYBRID` | BM25+dense `query_hybrid` when store supports it |
| `INTERGRAX_RAG_AGENTIC_ENABLED` | Budgeted deep-tier query refinement loop |
| `INTERGRAX_RAG_GRAPH_ENABLED` | Register `graph_rag` retriever + ingest graph indexing |
| `INTERGRAX_RAG_GRAPH_INDEXER_MODE` | `heuristic` (default), `llm`, `heuristic_then_llm` - requires `llm_adapter` in extras |
| `INTERGRAX_RAG_AGENTIC_QUERY_MODE` | `deterministic` or `llm` for deep-tier refinement |
| `INTERGRAX_RAG_QDRANT_SPARSE` | Qdrant native sparse vectors + RRF hybrid query |
| `INTERGRAX_RAG_SPARSE_ENCODER` | `bm25_hash` (default) or `splade` (requires `fastembed`) |
| `INTERGRAX_RAG_WEAVIATE_NATIVE_HYBRID` | Weaviate `query.hybrid` when client is wired |
| `INTERGRAX_RAG_GRAPH_STORE` | `inmemory` (default) or `neo4j` for GraphRAG persistence |
| `INTERGRAX_RAG_METRICS_ENABLED` | Export retrieval latencies, hybrid/agentic stats, recall@k avg |

Pass optional adapters via `ToolWiringContext.extras`:

```python
extras={"llm_adapter": runtime_config.llm_adapter, "graph_store": graph_store}
```

### `rag.ingest_document`

Uses **`IngestPipeline`**: configurable loader/splitter (extras or defaults), chunking strategy from `RagProfile.chunking_strategy_id` or metadata `chunking_strategy_id`, optional contextual enrich when `INTERGRAX_RAG_CONTEXTUAL_ENRICH=on` and `contextual_enricher` is wired with an `LLMAdapter`.

Parsers are **not** fixed to Docling - handlers use smart parsers + `ParserPipeline` catalog fallback; set `INTERGRAX_RAG_DOCUMENT_PARSER_SLUG` to force an integration parser slug.

### `rag.list_collections`

Returns collection names from the active vector store (`VectorStore.list_collections()`).

### `rag.delete_documents`

Deletes indexed vector chunks by document id via `vectorstore_manager.delete(ids)`.

### `rag.describe_collection`

Returns document count and available collection names from the active vector store.

## Agent allow-list

```python
AgentContract(allowed_tools=["rag.retrieve", "rag.list_collections"], ...)
```
