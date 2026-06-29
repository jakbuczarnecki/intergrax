# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sendgrid notification channel integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence, Any

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

SENDGRID_NOTIFICATION_CHANNEL_PROVIDER_ID = "sendgrid"


class SendgridNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Sendgrid notification channel integration."""

    pass


SendgridNotificationChannelClient = NotificationChannel

class SendgridNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Single public Sendgrid notification channel entrypoint.

    Legacy catalog factory (create_sendgrid_notification_channel) owns catalog behavior; legacy factories use from_client().
    """

    config: SendgridNotificationChannelIntegrationConfig = SendgridNotificationChannelIntegrationConfig()
    _client: SendgridNotificationChannelClient | None = PrivateAttr(default=None)
    


    async def notify(self, message: Any) -> None:
        await self._require_client().notify(message)

    def health(self) -> Any:
        return self._require_client().health()



    def _require_client(self) -> NotificationChannel:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


    @classmethod
    def from_client(
        cls,
        client: SendgridNotificationChannelClient,
        *,
        enabled: bool = False,
    ) -> SendgridNotificationChannelIntegration:
        integration = cls.for_provider(
            provider_id=SENDGRID_NOTIFICATION_CHANNEL_PROVIDER_ID,
            display_name="Sendgrid",
            config=SendgridNotificationChannelIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> SendgridNotificationChannelClient | None:
        return self._client

NotificationChannel.register(SendgridNotificationChannelIntegration)
