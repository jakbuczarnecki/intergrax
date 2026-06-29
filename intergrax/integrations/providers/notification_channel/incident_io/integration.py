# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Incident Io notification channel integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence, Any

from pydantic import PrivateAttr

from intergrax.integrations._shared.health import probe_client_health
from intergrax.integrations.contracts.base import HealthStatus, IntegrationConfigurationError
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

INCIDENT_IO_NOTIFICATION_CHANNEL_PROVIDER_ID = "incident_io"


class IncidentIoNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Incident Io notification channel integration."""

    pass


IncidentIoNotificationChannelClient = NotificationChannel

class IncidentIoNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Single public Incident Io notification channel entrypoint.

    Legacy catalog factory (create_incident_io_notification_channel) owns catalog behavior; legacy factories use from_client().
    """

    config: IncidentIoNotificationChannelIntegrationConfig = IncidentIoNotificationChannelIntegrationConfig()
    _client: IncidentIoNotificationChannelClient | None = PrivateAttr(default=None)
    


    async def notify(self, message: Any) -> None:
        await self._require_client().notify(message)

    def health(self) -> HealthStatus:
        return probe_client_health(
            self._require_client(),
            slug=INCIDENT_IO_NOTIFICATION_CHANNEL_PROVIDER_ID,
            default_detail="incident_io ready probe",
        )



    def _require_client(self) -> NotificationChannel:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


    @classmethod
    def from_client(
        cls,
        client: IncidentIoNotificationChannelClient,
        *,
        enabled: bool = False,
    ) -> IncidentIoNotificationChannelIntegration:
        integration = cls.for_provider(
            provider_id=INCIDENT_IO_NOTIFICATION_CHANNEL_PROVIDER_ID,
            display_name="Incident Io",
            config=IncidentIoNotificationChannelIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> IncidentIoNotificationChannelClient | None:
        return self._client

NotificationChannel.register(IncidentIoNotificationChannelIntegration)
