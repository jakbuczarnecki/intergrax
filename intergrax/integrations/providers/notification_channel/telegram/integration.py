# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Telegram notification channel integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence, Any

from pydantic import PrivateAttr

from intergrax.integrations._shared.health import probe_client_health
from intergrax.integrations.contracts.base import HealthStatus, IntegrationConfigurationError
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

TELEGRAM_NOTIFICATION_CHANNEL_PROVIDER_ID = "telegram"


class TelegramNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Telegram notification channel integration."""

    pass


TelegramNotificationChannelClient = NotificationChannel

class TelegramNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Single public Telegram notification channel entrypoint.

    Legacy catalog factory (create_telegram_catalog_factory) owns catalog behavior; legacy factories use from_client().
    """

    config: TelegramNotificationChannelIntegrationConfig = TelegramNotificationChannelIntegrationConfig()
    _client: TelegramNotificationChannelClient | None = PrivateAttr(default=None)
    


    async def notify(self, message: Any) -> None:
        await self._require_client().notify(message)

    def health(self) -> HealthStatus:
        return probe_client_health(
            self._require_client(),
            slug=TELEGRAM_NOTIFICATION_CHANNEL_PROVIDER_ID,
            default_detail="telegram ready probe",
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
        client: TelegramNotificationChannelClient,
        *,
        enabled: bool = False,
    ) -> TelegramNotificationChannelIntegration:
        integration = cls.for_provider(
            provider_id=TELEGRAM_NOTIFICATION_CHANNEL_PROVIDER_ID,
            display_name="Telegram",
            config=TelegramNotificationChannelIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> TelegramNotificationChannelClient | None:
        return self._client

NotificationChannel.register(TelegramNotificationChannelIntegration)
