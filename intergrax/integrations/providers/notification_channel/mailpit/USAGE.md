# Mailpit (mailpit)

Category: `notification_channel`

## Single public entrypoint

- **`MailpitNotificationChannelIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `MailpitNotificationChannelIntegration`.
- Contract factory: `create_mailpit_notification_channel_integration()`.
