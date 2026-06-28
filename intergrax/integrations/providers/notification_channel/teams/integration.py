# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Teams notification channel integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

TEAMS_NOTIFICATION_CHANNEL_PROVIDER_ID = "teams"


class TeamsNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Teams notification channel integration."""

    pass


@runtime_checkable
class TeamsNotificationChannelClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class TeamsNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Teams notification channel integration.

    The legacy facade (create_teams_integration) remains separate and backward-compatible.
    """

    config: TeamsNotificationChannelIntegrationConfig = TeamsNotificationChannelIntegrationConfig()
    _client: TeamsNotificationChannelClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: TeamsNotificationChannelClient,
        *,
        enabled: bool = False,
    ) -> TeamsNotificationChannelIntegration:
        integration = cls.for_provider(
            provider_id=TEAMS_NOTIFICATION_CHANNEL_PROVIDER_ID,
            display_name="Teams",
            config=TeamsNotificationChannelIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> TeamsNotificationChannelClient | None:
        return self._client
