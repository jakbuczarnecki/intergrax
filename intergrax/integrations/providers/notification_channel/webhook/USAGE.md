# Webhook (webhook)

Category: `notification_channel`

## Single public entrypoint

- **`WebhookNotificationChannelIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `WebhookNotificationChannelIntegration`.
- Contract factory: `create_webhook_notification_channel_integration()`.
