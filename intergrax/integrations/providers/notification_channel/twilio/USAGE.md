# Twilio (twilio)

Category: `notification_channel`

## Single public entrypoint

- **`TwilioNotificationChannelIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `TwilioNotificationChannelIntegration`.
- Contract factory: `create_twilio_notification_channel_integration()`.
