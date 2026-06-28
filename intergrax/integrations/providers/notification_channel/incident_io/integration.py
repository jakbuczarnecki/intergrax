# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Incident Io notification channel integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

INCIDENT_IO_NOTIFICATION_CHANNEL_PROVIDER_ID = "incident_io"


class IncidentIoNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Incident Io notification channel integration."""

    pass


@runtime_checkable
class IncidentIoNotificationChannelClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class IncidentIoNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Incident Io notification channel integration.

    The legacy facade (create_incident_io_notification_channel) remains separate and backward-compatible.
    """

    config: IncidentIoNotificationChannelIntegrationConfig = IncidentIoNotificationChannelIntegrationConfig()
    _client: IncidentIoNotificationChannelClient | None = PrivateAttr(default=None)

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
