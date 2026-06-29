# Teams (teams)

Category: `notification_channel`

## Single public entrypoint

- **`TeamsNotificationChannelIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `TeamsNotificationChannelIntegration`.
- Contract factory: `create_teams_notification_channel_integration()`.
