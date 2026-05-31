# `twilio` integration — usage

**Category:** ``notification_channel``  
**Catalog factory:** ``create_twilio_notification_channel()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(notification_channel=IntegrationSlug.TWILIO)
backend = profile.resolve(IntegrationCategory.NOTIFICATION_CHANNEL)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.notification_channel.twilio.bundle import create_twilio_notification_channel

backend = create_twilio_notification_channel(**config_overrides)
```


## Environment variables

`INTERGRAX_TWILIO_ORG` (account SID), `INTERGRAX_TWILIO_USER`/`API_KEY`, `INTERGRAX_TWILIO_PASSWORD`/`TOKEN`, `INTERGRAX_TWILIO_SITE_URL` (from number)

## Example

```python
from intergrax.integrations.providers.notification_channel.twilio.bundle import create_twilio_notification_channel

sms = create_twilio_notification_channel(org="AC...", site_url="+1...")
```

## Notes

SMS HITL — set ``metadata['to']`` on ``NotificationMessage``.
