# `pagerduty` integration — usage

**Category:** ``notification_channel``  
**Catalog factory:** ``create_pagerduty_notification_channel()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(notification_channel="pagerduty")
backend = profile.resolve(IntegrationCategory.NOTIFICATION_CHANNEL)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.notification_channel.pagerduty.bundle import create_pagerduty_notification_channel

backend = create_pagerduty_notification_channel(**config_overrides)
```


## Environment variables

`INTERGRAX_PAGERDUTY_API_KEY` (routing key)

## Example

```python
from intergrax.integrations.providers.notification_channel.pagerduty.bundle import create_pagerduty_notification_channel

pd = create_pagerduty_notification_channel(api_key="routing-key")
```

## Notes

On-call escalation via Events API v2.
