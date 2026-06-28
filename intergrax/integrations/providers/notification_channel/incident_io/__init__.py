# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "INCIDENT_IO_NOTIFICATION_CHANNEL_PROVIDER_ID",
    "IncidentIoNotificationChannelIntegration",
    "IncidentIoNotificationChannelIntegrationConfig",
    "IncidentIoNotificationChannelClient",
    "create_incident_io_notification_channel",
    "create_incident_io_notification_channel_integration",
    "register_incident_io_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_incident_io_notification_channel",
        "create_incident_io_notification_channel_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "INCIDENT_IO_NOTIFICATION_CHANNEL_PROVIDER_ID",
        "IncidentIoNotificationChannelIntegration",
        "IncidentIoNotificationChannelIntegrationConfig",
        "IncidentIoNotificationChannelClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "INCIDENT_IO_NOTIFICATION_CHANNEL_PROVIDER_ID",
        "IncidentIoNotificationChannelIntegration",
        "IncidentIoNotificationChannelIntegrationConfig",
        "IncidentIoNotificationChannelClient",
    }
)

def __getattr__(name: str):
    if name == "register_incident_io_integration":
        from intergrax.integrations.providers.notification_channel.incident_io.register import register_incident_io_integration

        return register_incident_io_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.notification_channel.incident_io import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.notification_channel.incident_io import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.notification_channel.incident_io import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
