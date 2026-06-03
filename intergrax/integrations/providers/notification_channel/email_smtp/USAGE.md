# `email_smtp` integration — usage

**Category:** ``notification_channel``  
**Catalog factory:** ``create_email_smtp_notification_channel()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(notification_channel="email_smtp")
backend = profile.resolve(IntegrationCategory.NOTIFICATION_CHANNEL)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.notification_channel.email_smtp.bundle import create_email_smtp_notification_channel

backend = create_email_smtp_notification_channel(**config_overrides)
```


## Environment variables

`INTERGRAX_EMAIL_SMTP_HOST`, `INTERGRAX_EMAIL_SMTP_PORT` (default `587`); optional `USER`, `PASSWORD`, `FROM`

## Example

```python
from intergrax.integrations.providers.notification_channel.email_smtp.bundle import create_email_smtp_notification_channel

import asyncio
from intergrax.runtime.notifications.models import NotificationMessage

channel = create_email_smtp_notification_channel(
    smtp_host="smtp.example.com",
    smtp_port=587,
    user="bot@example.com",
    password="...",
    from_address="noreply@example.com",
)
asyncio.run(channel.notify(NotificationMessage(
    tenant_id="t1",
    channel="#alerts",
    task_id="task-1",
    subject="HITL approval required",
    body="Please review run r-42.",
    metadata={"to": "ops@example.com"},
)))
```

## Notes

stdlib ``smtplib`` in factory open path. Implements ``NotificationAdapter`` (async ``notify``).
