# Observability tool bundle

**Bundle id:** `observability`  
**Tools:** `metrics.query_instant`, `logs.search`, `observability.query_traces`, `errors.capture`

## Dependencies (`ToolWiringContext`)

| Field | Required | Purpose |
|-------|----------|---------|
| `observability_backend` | Yes | `ObservabilityBackend` (`prometheus`, `elasticsearch`, `langfuse`, …) |

Tier-3 example:

```python
from intergrax.integrations import IntegrationProfile, IntegrationSlug, register_default_integrations
from intergrax.tools.registry import ToolProfile, ToolWiringContext, build_registry_from_profile, register_default_tools

register_default_integrations()
register_default_tools()

profile = IntegrationProfile(observability_backend=IntegrationSlug.LANGFUSE)
ctx = ToolWiringContext.from_integration_profile(profile)
registry = build_registry_from_profile(ToolProfile(enabled_bundles=["observability"]), ctx=ctx)
```

## Notes

- `metrics.query_instant` accepts PromQL (Prometheus backend).
- `logs.search` uses Elasticsearch `_search` via the backend's `rest_client` when available.
- `observability.query_traces` calls `ObservabilityBackend.query_traces()` (Langfuse REST when configured).
- `errors.capture` reports an error event via `ObservabilityBackend.capture_message()` (Sentry when `observability_backend=sentry`).

## Agent allow-list

```python
AgentContract(
    allowed_tools=[
        "metrics.query_instant",
        "logs.search",
        "observability.query_traces",
        "errors.capture",
    ],
    ...
)
```
