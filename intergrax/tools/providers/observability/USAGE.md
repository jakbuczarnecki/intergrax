# Observability tool bundle

**Bundle id:** `observability`  
**Tools:** `metrics.query_instant`, `logs.search`

## Dependencies (`ToolWiringContext`)

| Field | Required | Purpose |
|-------|----------|---------|
| `observability_backend` | Yes | `ObservabilityBackend` (`prometheus` for metrics, `elasticsearch` for logs) |

Tier-3 example:

```python
from intergrax.integrations import IntegrationProfile, IntegrationSlug, register_default_integrations
from intergrax.tools.registry import ToolProfile, ToolWiringContext, build_registry_from_profile, register_default_tools

register_default_integrations()
register_default_tools()

profile = IntegrationProfile(observability_backend=IntegrationSlug.PROMETHEUS)
ctx = ToolWiringContext.from_integration_profile(profile)
registry = build_registry_from_profile(ToolProfile(enabled_bundles=["observability"]), ctx=ctx)
```

## Notes

- `metrics.query_instant` accepts PromQL (Prometheus backend).
- `logs.search` uses Elasticsearch `_search` via the backend's `rest_client` when available.

## Agent allow-list

```python
AgentContract(allowed_tools=["metrics.query_instant", "logs.search"], ...)
```
