# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pagerduty notification channel integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence, Any

from pydantic import PrivateAttr

from intergrax.integrations._shared.health import probe_client_health
from intergrax.integrations.contracts.base import HealthStatus, IntegrationConfigurationError
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

PAGERDUTY_NOTIFICATION_CHANNEL_PROVIDER_ID = "pagerduty"


class PagerdutyNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Pagerduty notification channel integration."""

    pass


PagerdutyNotificationChannelClient = NotificationChannel

class PagerdutyNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Single public Pagerduty notification channel entrypoint.

    Legacy catalog factory (create_pagerduty_integration) owns catalog behavior; legacy factories use from_client().
    """

    config: PagerdutyNotificationChannelIntegrationConfig = PagerdutyNotificationChannelIntegrationConfig()
    _client: PagerdutyNotificationChannelClient | None = PrivateAttr(default=None)
    


    async def notify(self, message: Any) -> None:
        await self._require_client().notify(message)

    def health(self) -> HealthStatus:
        return probe_client_health(
            self._require_client(),
            slug=PAGERDUTY_NOTIFICATION_CHANNEL_PROVIDER_ID,
            default_detail="pagerduty ready probe",
        )

    def trigger_incident(self, *, summary: str, **kwargs: Any) -> str:
        return self._require_client().trigger_incident(summary=summary, **kwargs)


    def _require_client(self) -> NotificationChannel:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


    @classmethod
    def from_client(
        cls,
        client: PagerdutyNotificationChannelClient,
        *,
        enabled: bool = False,
    ) -> PagerdutyNotificationChannelIntegration:
        integration = cls.for_provider(
            provider_id=PAGERDUTY_NOTIFICATION_CHANNEL_PROVIDER_ID,
            display_name="Pagerduty",
            config=PagerdutyNotificationChannelIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> PagerdutyNotificationChannelClient | None:
        return self._client

NotificationChannel.register(PagerdutyNotificationChannelIntegration)
