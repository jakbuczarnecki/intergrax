# RAG tool bundle

**Bundle id:** `rag`  
**Tools:** `rag.retrieve`, `rag.ingest_document`, `rag.list_collections`

## Dependencies (`ToolWiringContext`)

| Field | Required | Purpose |
|-------|----------|---------|
| `vectorstore_manager` | Yes | Semantic search over indexed chunks |
| `embedding_manager` | Yes (retrieve/ingest) | Query embedding for vector search |

Tier-3 example:

```python
from intergrax.tools.registry import ToolProfile, ToolWiringContext, build_registry_from_profile, register_default_tools

register_default_tools()
ctx = ToolWiringContext(
    vectorstore_manager=runtime_config.vectorstore_manager,
    embedding_manager=runtime_config.embedding_manager,
)
registry = build_registry_from_profile(
    ToolProfile(enabled=["rag.retrieve", "rag.ingest_document", "rag.list_collections"]),
    ctx=ctx,
)
```

### `rag.ingest_document`

Loads a local file through the default RAG document handler pipeline (integration-backed parsers), splits, embeds, and stores chunks. Response includes `parser_id` and `integration_parser_trace` from `ParserPipeline`.

### `rag.list_collections`

Returns collection names from the active vector store (`VectorStore.list_collections()`). Useful before ingest/retrieve when multiple namespaces exist.

## Agent allow-list

```python
AgentContract(allowed_tools=["rag.retrieve", "rag.list_collections"], ...)
```
