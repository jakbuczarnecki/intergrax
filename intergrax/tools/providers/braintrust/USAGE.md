# Braintrust tool bundle

**Bundle id:** `braintrust`  
**Tools:** `braintrust.log_eval`

## Dependencies (`ToolWiringContext`)

| Field | Required | Purpose |
|-------|----------|---------|
| `observability_backend` | Yes | Braintrust backend with `log_eval()` |

Tier-3 example:

```python
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.tools.registry import ToolProfile, ToolWiringContext, build_registry_from_profile, register_default_tools

register_default_integrations()
register_default_tools()

profile = IntegrationProfile(observability_backend="braintrust")
ctx = ToolWiringContext.from_integration_profile(profile)
registry = build_registry_from_profile(ToolProfile(enabled_bundles=["braintrust"]), ctx=ctx)
```

Env: `INTERGRAX_BRAINTRUST_API_KEY`, optional `INTERGRAX_BRAINTRUST_PROJECT`.

## Agent allow-list

```python
AgentContract(allowed_tools=["braintrust.log_eval"], ...)
```
