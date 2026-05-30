# `webhook` integration — usage

**Category:** ``notification_channel``  
**Catalog factory:** ``create_webhook_notification_channel()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(notification_channel=IntegrationSlug.WEBHOOK)
backend = profile.resolve(IntegrationCategory.NOTIFICATION_CHANNEL)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.webhook.bundle import create_webhook_notification_channel

backend = create_webhook_notification_channel(**config_overrides)
```


## Environment variables

`INTERGRAX_WEBHOOK_URL`

## Example

```python
from intergrax.integrations.providers.webhook.bundle import create_webhook_notification_channel

notifier = create_webhook_notification_channel(url="https://example.com/hooks/intergrax")
notifier.notify({"event": "task.completed", "task_id": "t-1"})
```

## Notes

Generic HTTP POST; JSON formatting via ``GenericJsonPayloadFormatter``.
