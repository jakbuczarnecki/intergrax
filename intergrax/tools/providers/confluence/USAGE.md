# Confluence tool bundle

**Bundle id:** `confluence`  
**Tools:** `confluence.get_page`, `confluence.search_pages`, `confluence.search`

`confluence.search` is a stable alias of `confluence.search_pages` (same handler) — use the shorter id in large LLM tool catalogs.

## Dependencies (`ToolWiringContext`)

| Field | Required | Purpose |
|-------|----------|---------|
| `wiki_knowledge` | Yes | `WikiKnowledge` contract (typically `confluence` integration) |

Tier-3 example:

```python
from intergrax.integrations import IntegrationProfile, IntegrationSlug, register_default_integrations
from intergrax.tools.registry import ToolProfile, ToolWiringContext, build_registry_from_profile, register_default_tools

register_default_integrations()
register_default_tools()

profile = IntegrationProfile(wiki_knowledge=IntegrationSlug.CONFLUENCE)
ctx = ToolWiringContext.from_integration_profile(profile)
registry = build_registry_from_profile(ToolProfile(enabled_bundles=["confluence"]), ctx=ctx)
```

## Agent allow-list

```python
AgentContract(
    allowed_tools=["confluence.get_page", "confluence.search_pages", "confluence.search"],
    ...
)
```
