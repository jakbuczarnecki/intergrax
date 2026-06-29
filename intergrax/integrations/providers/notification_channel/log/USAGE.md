# Log (log)

Category: `notification_channel`

## Single public entrypoint

- **`LogNotificationChannelIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `LogNotificationChannelIntegration`.
- Contract factory: `create_log_notification_channel_integration()`.
