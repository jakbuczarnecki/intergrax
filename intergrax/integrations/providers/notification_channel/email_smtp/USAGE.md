# Email Smtp (email_smtp)

Category: `notification_channel`

## Single public entrypoint

- **`EmailSmtpNotificationChannelIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `EmailSmtpNotificationChannelIntegration`.
- Contract factory: `create_email_smtp_notification_channel_integration()`.
