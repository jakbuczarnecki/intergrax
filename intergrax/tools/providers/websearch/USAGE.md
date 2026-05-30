# Websearch tool bundle

**Bundle id:** `websearch`  
**Tools:** `websearch.query`

## Dependencies (`ToolWiringContext`)

| Field | Priority | Purpose |
|-------|----------|---------|
| `websearch_executor` | 1 (preferred) | Full `WebSearchExecutor` pipeline (search + fetch) |
| `search_provider` | 2 (fallback) | Integration catalog `SearchProvider` (`google_cse`, `bing`) |

Tier-3 example:

```python
from intergrax.tools.registry import ToolProfile, ToolWiringContext, build_registry_from_profile, register_default_tools

register_default_tools()
ctx = ToolWiringContext(websearch_executor=runtime_context.websearch_executor)
registry = build_registry_from_profile(ToolProfile(enabled=["websearch.query"]), ctx=ctx)
```

Integration-only fallback:

```python
ctx = ToolWiringContext.from_integration_profile(
    integration_profile,
    # resolves SearchProvider when search_provider slug is set
)
```

## Agent allow-list

```python
AgentContract(allowed_tools=["websearch.query"], ...)
```
