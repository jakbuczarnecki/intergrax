# Pagerduty (pagerduty)

Category: `notification_channel`

## Single public entrypoint

- **`PagerdutyNotificationChannelIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `PagerdutyNotificationChannelIntegration`.
- Contract factory: `create_pagerduty_notification_channel_integration()`.
