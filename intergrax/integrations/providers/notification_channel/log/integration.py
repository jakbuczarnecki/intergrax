# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Log notification channel integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence, Any

from pydantic import PrivateAttr

from intergrax.integrations._shared.health import probe_client_health
from intergrax.integrations.contracts.base import HealthStatus, IntegrationConfigurationError
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

LOG_NOTIFICATION_CHANNEL_PROVIDER_ID = "log"


class LogNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Log notification channel integration."""

    pass


LogNotificationChannelClient = NotificationChannel

class LogNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Single public Log notification channel entrypoint.

    Legacy catalog factory (create_log_integration) owns catalog behavior; legacy factories use from_client().
    """

    config: LogNotificationChannelIntegrationConfig = LogNotificationChannelIntegrationConfig()
    _client: LogNotificationChannelClient | None = PrivateAttr(default=None)
    


    async def notify(self, message: Any) -> None:
        await self._require_client().notify(message)

    def health(self) -> HealthStatus:
        return probe_client_health(
            self._require_client(),
            slug=LOG_NOTIFICATION_CHANNEL_PROVIDER_ID,
            default_detail="log ready probe",
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
        client: LogNotificationChannelClient,
        *,
        enabled: bool = False,
    ) -> LogNotificationChannelIntegration:
        integration = cls.for_provider(
            provider_id=LOG_NOTIFICATION_CHANNEL_PROVIDER_ID,
            display_name="Log",
            config=LogNotificationChannelIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> LogNotificationChannelClient | None:
        return self._client

NotificationChannel.register(LogNotificationChannelIntegration)
