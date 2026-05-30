# `teams` integration — usage

**Category:** ``notification_channel + interaction_surface``  
**Catalog factory:** ``create_teams_notification_channel()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(notification_channel=IntegrationSlug.TEAMS)
backend = profile.resolve(IntegrationCategory.NOTIFICATION_CHANNEL)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.teams.bundle import create_teams_notification_channel

backend = create_teams_notification_channel(**config_overrides)
```


## Environment variables

`INTERGRAX_TEAMS_WEBHOOK_URL`; optional `INTERGRAX_TEAMS_SECURITY_TOKEN`

## Example

```python
from intergrax.integrations.providers.teams.bundle import create_teams_notification_channel

notifier = create_teams_notification_channel(webhook_url="https://outlook.office.com/webhook/...")
notifier.notify("Nexus run completed")

# Inbound: create_teams_interaction_surface(security_token="...")
# profile.resolve(IntegrationCategory.INTERACTION_SURFACE) when interaction_surface=TEAMS
```

## Notes

Same dual-category pattern as Slack — separate factory for ``INTERACTION_SURFACE``.
