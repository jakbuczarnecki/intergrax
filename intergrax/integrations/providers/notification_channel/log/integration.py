# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Log notification channel integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

LOG_NOTIFICATION_CHANNEL_PROVIDER_ID = "log"


class LogNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Log notification channel integration."""

    pass


@runtime_checkable
class LogNotificationChannelClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class LogNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Log notification channel integration.

    The legacy facade (create_log_integration) remains separate and backward-compatible.
    """

    config: LogNotificationChannelIntegrationConfig = LogNotificationChannelIntegrationConfig()
    _client: LogNotificationChannelClient | None = PrivateAttr(default=None)

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
