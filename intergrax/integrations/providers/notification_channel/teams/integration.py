# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Teams notification channel integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence, Any

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

TEAMS_NOTIFICATION_CHANNEL_PROVIDER_ID = "teams"


class TeamsNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Teams notification channel integration."""

    pass


TeamsNotificationChannelClient = NotificationChannel

class TeamsNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Single public Teams notification channel entrypoint.

    Legacy catalog factory (create_teams_integration) owns catalog behavior; legacy factories use from_client().
    """

    config: TeamsNotificationChannelIntegrationConfig = TeamsNotificationChannelIntegrationConfig()
    _client: TeamsNotificationChannelClient | None = PrivateAttr(default=None)
    


    async def notify(self, message: Any) -> None:
        await self._require_client().notify(message)

    @property
    def webhook_url(self) -> str | None:
        client = self._client
        if client is None:
            return None
        return client.webhook_url

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

NotificationChannel.register(TeamsNotificationChannelIntegration)
