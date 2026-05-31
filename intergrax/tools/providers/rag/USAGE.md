# RAG tool bundle

**Bundle id:** `rag`  
**Tools:** `rag.retrieve`

## Dependencies (`ToolWiringContext`)

| Field | Required | Purpose |
|-------|----------|---------|
| `vectorstore_manager` | Yes | Semantic search over indexed chunks |
| `embedding_manager` | Yes | Query embedding for vector search |

Tier-3 example:

```python
from intergrax.tools.registry import ToolProfile, ToolWiringContext, build_registry_from_profile, register_default_tools

register_default_tools()
ctx = ToolWiringContext(
    vectorstore_manager=runtime_config.vectorstore_manager,
    embedding_manager=runtime_config.embedding_manager,
)
registry = build_registry_from_profile(ToolProfile(enabled=["rag.retrieve"]), ctx=ctx)
```

## Invoke (via runtime)

```python
from intergrax.tools.providers.rag.contracts import RagRetrieveInput

# ToolExecutionRequest(tool_id="rag.retrieve", input=RagRetrieveInput(query="...", top_k=5), ...)
```

## Agent allow-list

```python
AgentContract(allowed_tools=["rag.retrieve"], ...)
```
