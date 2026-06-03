# `opsgenie` integration — usage

**Category:** ``notification_channel``  
**Catalog factory:** ``create_opsgenie_notification_channel()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(notification_channel="opsgenie")
backend = profile.resolve(IntegrationCategory.NOTIFICATION_CHANNEL)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.notification_channel.opsgenie.bundle import create_opsgenie_notification_channel

backend = create_opsgenie_notification_channel(**config_overrides)
```


## Environment variables

`INTERGRAX_OPSGENIE_API_KEY`

## Example

```python
from intergrax.integrations.providers.notification_channel.opsgenie.bundle import create_opsgenie_notification_channel

og = create_opsgenie_notification_channel(api_key="...")
```

## Notes

Alertmanager-style HITL escalation.
