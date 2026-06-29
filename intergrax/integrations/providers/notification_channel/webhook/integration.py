# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Webhook notification channel integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence, Any

from pydantic import PrivateAttr

from intergrax.integrations._shared.health import probe_client_health
from intergrax.integrations.contracts.base import HealthStatus, IntegrationConfigurationError
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

WEBHOOK_NOTIFICATION_CHANNEL_PROVIDER_ID = "webhook"


class WebhookNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Webhook notification channel integration."""

    pass


WebhookNotificationChannelClient = NotificationChannel

class WebhookNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Single public Webhook notification channel entrypoint.

    Legacy catalog factory (create_webhook_integration) owns catalog behavior; legacy factories use from_client().
    """

    config: WebhookNotificationChannelIntegrationConfig = WebhookNotificationChannelIntegrationConfig()
    _client: WebhookNotificationChannelClient | None = PrivateAttr(default=None)
    


    async def notify(self, message: Any) -> None:
        await self._require_client().notify(message)

    def health(self) -> HealthStatus:
        return probe_client_health(
            self._require_client(),
            slug=WEBHOOK_NOTIFICATION_CHANNEL_PROVIDER_ID,
            default_detail="webhook ready probe",
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
        client: WebhookNotificationChannelClient,
        *,
        enabled: bool = False,
    ) -> WebhookNotificationChannelIntegration:
        integration = cls.for_provider(
            provider_id=WEBHOOK_NOTIFICATION_CHANNEL_PROVIDER_ID,
            display_name="Webhook",
            config=WebhookNotificationChannelIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> WebhookNotificationChannelClient | None:
        return self._client

NotificationChannel.register(WebhookNotificationChannelIntegration)
