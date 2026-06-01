# Observability tool bundle

**Bundle id:** `observability`  
**Tools:** `metrics.query_instant`, `logs.search`, `observability.query_traces`, `errors.capture`

## Dependencies (`ToolWiringContext`)

| Field | Required | Purpose |
|-------|----------|---------|
| `observability_backend` | Yes (single-backend) | Primary `ObservabilityBackend` |
| `observability_backends` | Optional (composite) | Map of slug → backend for role-based routing (Phase M.10 harness) |

### Single backend

```python
from intergrax.integrations import IntegrationProfile, IntegrationSlug, register_default_integrations
from intergrax.tools.registry import ToolProfile, ToolWiringContext, build_registry_from_profile, register_default_tools

register_default_integrations()
register_default_tools()

profile = IntegrationProfile(observability_backend=IntegrationSlug.LANGFUSE)
ctx = ToolWiringContext.from_integration_profile(profile)
registry = build_registry_from_profile(ToolProfile(enabled_bundles=["observability"]), ctx=ctx)
```

### Composite harness (Sentry errors + LangSmith traces)

When `IntegrationProfile.harness_lab()` sets `observability_backend=sentry` and `options={langsmith: …}`,
`ToolWiringContext.from_integration_profile()` fills `observability_backends`. Tools resolve by role:

| Role | Tool | Preferred slug |
|------|------|----------------|
| `errors` | `errors.capture` | `sentry` |
| `traces` | `observability.query_traces` | `langsmith`, `langfuse`, … |
| `logs` | `logs.search` | `elasticsearch`, `opensearch` |
| `eval` | `braintrust.log_eval` | `braintrust` |

```python
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.tools.registry import ToolWiringContext

ctx = ToolWiringContext.from_integration_profile(IntegrationProfile.harness_lab())
# errors.capture → Sentry; observability.query_traces → LangSmith
```

Implementation: `intergrax/tools/providers/observability/resolve.py`.

## Notes

- `metrics.query_instant` accepts PromQL (Prometheus backend).
- `logs.search` uses Elasticsearch `_search` via the backend's `rest_client` when available.
- `observability.query_traces` calls `ObservabilityBackend.query_traces()` (Langfuse/LangSmith REST when configured).
- `errors.capture` reports an error event via `ObservabilityBackend.capture_message()` (Sentry when role resolves to `sentry`).

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
