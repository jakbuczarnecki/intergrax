# Incident Io (incident_io)

Category: `notification_channel`

## Single public entrypoint

- **`IncidentIoNotificationChannelIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `IncidentIoNotificationChannelIntegration`.
- Contract factory: `create_incident_io_notification_channel_integration()`.
