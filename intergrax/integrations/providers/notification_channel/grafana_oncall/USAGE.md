# Grafana Oncall (grafana_oncall)

Category: `notification_channel`

## Single public entrypoint

- **`GrafanaOncallNotificationChannelIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `GrafanaOncallNotificationChannelIntegration`.
- Contract factory: `create_grafana_oncall_notification_channel_integration()`.
