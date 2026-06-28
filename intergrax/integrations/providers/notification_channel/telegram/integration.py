# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Telegram notification channel integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

TELEGRAM_NOTIFICATION_CHANNEL_PROVIDER_ID = "telegram"


class TelegramNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Telegram notification channel integration."""

    pass


@runtime_checkable
class TelegramNotificationChannelClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class TelegramNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Telegram notification channel integration.

    The legacy facade (create_telegram_catalog_factory) remains separate and backward-compatible.
    """

    config: TelegramNotificationChannelIntegrationConfig = TelegramNotificationChannelIntegrationConfig()
    _client: TelegramNotificationChannelClient | None = PrivateAttr(default=None)

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
