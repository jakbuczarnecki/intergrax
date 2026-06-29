# Sendgrid (sendgrid)

Category: `notification_channel`

## Single public entrypoint

- **`SendgridNotificationChannelIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `SendgridNotificationChannelIntegration`.
- Contract factory: `create_sendgrid_notification_channel_integration()`.
