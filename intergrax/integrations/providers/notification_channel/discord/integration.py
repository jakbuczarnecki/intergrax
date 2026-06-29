# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Discord notification channel integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence, Any

from pydantic import PrivateAttr

from intergrax.integrations._shared.health import probe_client_health
from intergrax.integrations.contracts.base import HealthStatus, IntegrationConfigurationError
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

DISCORD_NOTIFICATION_CHANNEL_PROVIDER_ID = "discord"


class DiscordNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Discord notification channel integration."""

    pass


DiscordNotificationChannelClient = NotificationChannel

class DiscordNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Single public Discord notification channel entrypoint.

    Legacy catalog factory (create_discord_notification_channel) owns catalog behavior; legacy factories use from_client().
    """

    config: DiscordNotificationChannelIntegrationConfig = DiscordNotificationChannelIntegrationConfig()
    _client: DiscordNotificationChannelClient | None = PrivateAttr(default=None)
    


    async def notify(self, message: Any) -> None:
        await self._require_client().notify(message)

    def health(self) -> HealthStatus:
        return probe_client_health(
            self._require_client(),
            slug=DISCORD_NOTIFICATION_CHANNEL_PROVIDER_ID,
            default_detail="discord ready probe",
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
        client: DiscordNotificationChannelClient,
        *,
        enabled: bool = False,
    ) -> DiscordNotificationChannelIntegration:
        integration = cls.for_provider(
            provider_id=DISCORD_NOTIFICATION_CHANNEL_PROVIDER_ID,
            display_name="Discord",
            config=DiscordNotificationChannelIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> DiscordNotificationChannelClient | None:
        return self._client

NotificationChannel.register(DiscordNotificationChannelIntegration)
