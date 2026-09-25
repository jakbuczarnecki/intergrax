# Notify tool bundle

**Bundle id:** `notify`  
**Tools:** `notify.send`

## Dependencies (`ToolWiringContext`)

| Field | Required | Purpose |
|-------|----------|---------|
| `notification_channel` | Yes | `NotificationChannel` adapter (Slack, Teams, log, webhook, …) |

Tier-3 example:

```python
from intergrax.tools.contracts.tool_profile import ToolProfile
from intergrax.tools.registry.bootstrap import register_default_tools
from intergrax.tools.registry.factory import build_registry_from_profile
from intergrax.tools.registry.wiring import ToolWiringContext

register_default_tools()
ctx = ToolWiringContext(notification_channel=notification_adapter)
registry = build_registry_from_profile(ToolProfile(enabled=["notify.send"]), ctx=ctx)
```

## Side effects

`notify.send` sets `side_effects=True` - outbound messages mutate external channels.

## Agent allow-list

```python
AgentContract(allowed_tools=["notify.send"], ...)
```
