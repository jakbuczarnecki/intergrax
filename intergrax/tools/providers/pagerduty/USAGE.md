# PagerDuty tool bundle

**Bundle id:** `pagerduty`  
**Tools:** `pagerduty.trigger_incident`

## Dependencies (`ToolWiringContext`)

| Field | Required | Purpose |
|-------|----------|---------|
| `notification_channel` | Yes | PagerDuty channel with `trigger_incident()` (Events API v2) |

Tier-3 example:

```python
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.tools.registry import ToolProfile, ToolWiringContext, build_registry_from_profile, register_default_tools

register_default_integrations()
register_default_tools()

profile = IntegrationProfile(notification_channel="pagerduty")
ctx = ToolWiringContext.from_integration_profile(profile)
registry = build_registry_from_profile(ToolProfile(enabled_bundles=["pagerduty"]), ctx=ctx)
```

Env: `INTERGRAX_PAGERDUTY_ROUTING_KEY`.

## Agent allow-list

```python
AgentContract(allowed_tools=["pagerduty.trigger_incident"], ...)
```
