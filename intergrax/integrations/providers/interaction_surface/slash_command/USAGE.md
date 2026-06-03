# Slash-command interaction surface

**Slug:** `slash_command`  
**Category:** `interaction_surface`  
**Factory:** `create_slash_command_interaction_surface()`

## Purpose

Vendor-neutral intake for slash-command shaped payloads (Slack slash commands, Microsoft Teams, CLI).

Wraps `SlashCommandInteractionAdapter` from the runtime interactions layer.

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `INTERGRAX_SLASH_COMMAND_DEFAULT_SOURCE` | `slash_command` | Logical source id stored in interaction metadata |

## Usage

```python
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(interaction_surface="slash_command")
adapter = profile.resolve("interaction_surface")

payload = {"command": "/research", "text": "summarize quarterly report", "user_id": "U1", "team_id": "T1"}
inbound = adapter.to_inbound(payload, tenant_id="t1", user_id="U1")
```

## Notes

- Parses `command` / `text` via `parse_slash_command_text()` — capability + message for Nexus routing.
- For Slack/Teams vendor SDK intake, prefer `"slack"` or `"teams"` (dual category registration).
