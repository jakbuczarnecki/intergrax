# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Grafana Oncall notification channel integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence, Any

from pydantic import PrivateAttr

from intergrax.integrations._shared.health import probe_client_health
from intergrax.integrations.contracts.base import HealthStatus, IntegrationConfigurationError
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

GRAFANA_ONCALL_NOTIFICATION_CHANNEL_PROVIDER_ID = "grafana_oncall"


class GrafanaOncallNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Grafana Oncall notification channel integration."""

    pass


GrafanaOncallNotificationChannelClient = NotificationChannel

class GrafanaOncallNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Single public Grafana Oncall notification channel entrypoint.

    Legacy catalog factory (create_grafana_oncall_notification_channel) owns catalog behavior; legacy factories use from_client().
    """

    config: GrafanaOncallNotificationChannelIntegrationConfig = GrafanaOncallNotificationChannelIntegrationConfig()
    _client: GrafanaOncallNotificationChannelClient | None = PrivateAttr(default=None)
    


    async def notify(self, message: Any) -> None:
        await self._require_client().notify(message)

    def health(self) -> HealthStatus:
        return probe_client_health(
            self._require_client(),
            slug=GRAFANA_ONCALL_NOTIFICATION_CHANNEL_PROVIDER_ID,
            default_detail="grafana_oncall ready probe",
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
        client: GrafanaOncallNotificationChannelClient,
        *,
        enabled: bool = False,
    ) -> GrafanaOncallNotificationChannelIntegration:
        integration = cls.for_provider(
            provider_id=GRAFANA_ONCALL_NOTIFICATION_CHANNEL_PROVIDER_ID,
            display_name="Grafana Oncall",
            config=GrafanaOncallNotificationChannelIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> GrafanaOncallNotificationChannelClient | None:
        return self._client

NotificationChannel.register(GrafanaOncallNotificationChannelIntegration)
