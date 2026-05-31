# `discord` integration — usage

**Category:** ``notification_channel``  
**Catalog factory:** ``create_discord_notification_channel()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(notification_channel=IntegrationSlug.DISCORD)
backend = profile.resolve(IntegrationCategory.NOTIFICATION_CHANNEL)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.notification_channel.discord.bundle import create_discord_notification_channel

backend = create_discord_notification_channel(**config_overrides)
```


## Environment variables

`INTERGRAX_DISCORD_URL` (webhook URL)

## Example

```python
from intergrax.integrations.providers.notification_channel.discord.bundle import create_discord_notification_channel

notify = create_discord_notification_channel(base_url="https://discord.com/api/webhooks/...")
```

## Notes

Community ops notifications via webhook POST.
