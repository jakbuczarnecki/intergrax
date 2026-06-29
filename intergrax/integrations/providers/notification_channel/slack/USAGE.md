# Slack (slack)

Category: `notification_channel`

## Single public entrypoint

- **`SlackNotificationChannelIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `SlackNotificationChannelIntegration`.
- Contract factory: `create_slack_notification_channel_integration()`.
