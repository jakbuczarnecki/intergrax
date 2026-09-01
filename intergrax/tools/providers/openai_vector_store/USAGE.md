# OpenAI vector store tool bundle

**Bundle id:** `openai_vector_store`  
**Tools:** `openai.file_search.query`, `openai.vector_store.upload`, `openai.vector_store.clear`

Vendor-specific operations on **OpenAI managed** vector stores. Not interchangeable with harness `rag.*` tools (Chroma/Qdrant + `RetrievalService`).

## Dependencies (`ToolWiringContext`)

| Source | Purpose |
|--------|---------|
| `extras["openai_client"]` | Pre-built `openai.OpenAI` client (preferred) |
| `OPENAI_API_KEY` | Used when client not injected |
| `extras["openai_vector_store_id"]` | Default vector store id |
| `INTERGRAX_OPENAI_VECTOR_STORE_ID` | Env fallback for vector store id |
| `INTERGRAX_OPENAI_FILE_SEARCH_MODEL` | Default Responses model (default `gpt-4o-mini`) |
| `extras["prompt_registry"]` | Optional; loads `knowledge_openai_strict_system` instructions |

## Skill

`knowledge.openai_strict` - allows `openai.file_search.query` with strict citation prompt.

## Example

```python
from intergrax.tools.registry import ToolProfile, ToolWiringContext, build_registry_from_profile, register_default_tools

register_default_tools()
ctx = ToolWiringContext(
    extras={
        "openai_vector_store_id": "vs_abc123",
    },
)
registry = build_registry_from_profile(
    ToolProfile(enabled=list(OPENAI_VECTOR_STORE_TOOL_IDS)),
    ctx=ctx,
)
```
