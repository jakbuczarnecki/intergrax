# `log` integration — usage

**Category:** ``notification_channel``  
**Catalog factory:** ``create_log_notification_channel()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(notification_channel="log")
backend = profile.resolve(IntegrationCategory.NOTIFICATION_CHANNEL)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.notification_channel.log.bundle import create_log_notification_channel

backend = create_log_notification_channel(**config_overrides)
```


## Environment variables

None — uses the application logger

## Example

```python
from intergrax.integrations.providers.notification_channel.log.bundle import create_log_notification_channel

notifier = create_log_notification_channel()
notifier.notify("HITL: approval required for task t-1")
```

## Notes

Default channel in ``IntegrationProfile.lab()`` — no network.
