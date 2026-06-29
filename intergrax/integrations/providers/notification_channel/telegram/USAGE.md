# Telegram (telegram)

Category: `notification_channel`

## Single public entrypoint

- **`TelegramNotificationChannelIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `TelegramNotificationChannelIntegration`.
- Contract factory: `create_telegram_notification_channel_integration()`.
