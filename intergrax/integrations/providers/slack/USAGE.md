# `slack` integration — usage

**Category:** ``notification_channel + interaction_surface``  
**Catalog factory:** ``create_slack_notification_channel()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(notification_channel=IntegrationSlug.SLACK)
backend = profile.resolve(IntegrationCategory.NOTIFICATION_CHANNEL)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.slack.bundle import create_slack_notification_channel

backend = create_slack_notification_channel(**config_overrides)
```


## Environment variables

`INTERGRAX_SLACK_WEBHOOK_URL`; optional `INTERGRAX_SLACK_SIGNING_SECRET` (inbound)

## Example

```python
from intergrax.integrations.providers.slack.bundle import create_slack_notification_channel

notifier = create_slack_notification_channel(webhook_url="https://hooks.slack.com/...")
notifier.notify("Task t-1 finished")

# Inbound (interaction):
from intergrax.integrations.providers.slack.bundle import create_slack_interaction_surface
surface = create_slack_interaction_surface(signing_secret="...")
# profile.resolve(IntegrationCategory.INTERACTION_SURFACE) when interaction_surface=SLACK
```

## Notes

Catalog registers both categories. ``create_slack_catalog_factory`` selects by ``IntegrationCategory``.
