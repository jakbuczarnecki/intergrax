# Mailgun (mailgun)

Category: `notification_channel`

- **`MailgunNotificationChannelIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `MailgunNotificationChannelIntegration`.
- Contract factory: `create_mailgun_notification_channel_integration()`.
- Inbound Mailgun webhook parsing remains a private adapter and is not a provider identity.
