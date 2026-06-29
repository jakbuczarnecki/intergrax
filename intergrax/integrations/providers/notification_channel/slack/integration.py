# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slack notification channel integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence, Any

from pydantic import PrivateAttr

from intergrax.integrations._shared.health import probe_client_health
from intergrax.integrations.contracts.base import HealthStatus, IntegrationConfigurationError
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

SLACK_NOTIFICATION_CHANNEL_PROVIDER_ID = "slack"


class SlackNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Slack notification channel integration."""

    pass


SlackNotificationChannelClient = NotificationChannel

class SlackNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Single public Slack notification channel entrypoint.

    Legacy catalog factory (create_slack_integration) owns catalog behavior; legacy factories use from_client().
    """

    config: SlackNotificationChannelIntegrationConfig = SlackNotificationChannelIntegrationConfig()
    _client: SlackNotificationChannelClient | None = PrivateAttr(default=None)
    


    async def notify(self, message: Any) -> None:
        await self._require_client().notify(message)

    def health(self) -> HealthStatus:
        return probe_client_health(
            self._require_client(),
            slug=SLACK_NOTIFICATION_CHANNEL_PROVIDER_ID,
            default_detail="slack ready probe",
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
        client: SlackNotificationChannelClient,
        *,
        enabled: bool = False,
    ) -> SlackNotificationChannelIntegration:
        integration = cls.for_provider(
            provider_id=SLACK_NOTIFICATION_CHANNEL_PROVIDER_ID,
            display_name="Slack",
            config=SlackNotificationChannelIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> SlackNotificationChannelClient | None:
        return self._client

NotificationChannel.register(SlackNotificationChannelIntegration)
