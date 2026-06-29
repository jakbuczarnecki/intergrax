# Opsgenie (opsgenie)

Category: `notification_channel`

## Single public entrypoint

- **`OpsgenieNotificationChannelIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `OpsgenieNotificationChannelIntegration`.
- Contract factory: `create_opsgenie_notification_channel_integration()`.
