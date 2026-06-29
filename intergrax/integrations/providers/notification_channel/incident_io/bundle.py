# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p5.factories import create_incident_io_notification_channel as _legacy_create_incident_io_notification_channel

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.notification_channel.incident_io.integration import (
    INCIDENT_IO_NOTIFICATION_CHANNEL_PROVIDER_ID,
    IncidentIoNotificationChannelIntegration,
    IncidentIoNotificationChannelIntegrationConfig,
    IncidentIoNotificationChannelClient,
)

__all__ = [
    "create_incident_io_notification_channel",
    "create_incident_io_notification_channel_integration",
]


def create_incident_io_notification_channel_integration(
    *,
    client: IncidentIoNotificationChannelClient | None = None,
    enabled: bool = False,
) -> IncidentIoNotificationChannelIntegration:
    """
    Build a contract-based Incident Io notification channel integration.

    The legacy facade (create_incident_io_notification_channel) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Incident Io notification channel integration requires an injected client when enabled=True",
        )
    if client is not None:
        return IncidentIoNotificationChannelIntegration.from_client(client, enabled=enabled)
    return IncidentIoNotificationChannelIntegration.for_provider(
        provider_id=INCIDENT_IO_NOTIFICATION_CHANNEL_PROVIDER_ID,
        display_name="Incident Io",
        config=IncidentIoNotificationChannelIntegrationConfig(enabled=enabled),
    )


def create_incident_io_notification_channel(**kwargs: object) -> IncidentIoNotificationChannelIntegration:
    """Compatibility shim — constructs IncidentIoNotificationChannelIntegration from legacy runtime."""
    runtime = _legacy_create_incident_io_notification_channel(**kwargs)
    if isinstance(runtime, IncidentIoNotificationChannelIntegration):
        return runtime
    return IncidentIoNotificationChannelIntegration.from_client(runtime)
