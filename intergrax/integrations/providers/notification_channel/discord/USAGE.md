# Discord (discord)

Category: `notification_channel`

## Single public entrypoint

- **`DiscordNotificationChannelIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `DiscordNotificationChannelIntegration`.
- Contract factory: `create_discord_notification_channel_integration()`.
