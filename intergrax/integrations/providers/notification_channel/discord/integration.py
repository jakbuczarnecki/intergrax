# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Discord notification channel integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

DISCORD_NOTIFICATION_CHANNEL_PROVIDER_ID = "discord"


class DiscordNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Discord notification channel integration."""

    pass


@runtime_checkable
class DiscordNotificationChannelClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class DiscordNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Discord notification channel integration.

    The legacy facade (create_discord_notification_channel) remains separate and backward-compatible.
    """

    config: DiscordNotificationChannelIntegrationConfig = DiscordNotificationChannelIntegrationConfig()
    _client: DiscordNotificationChannelClient | None = PrivateAttr(default=None)

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
