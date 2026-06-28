# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slack notification channel integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

SLACK_NOTIFICATION_CHANNEL_PROVIDER_ID = "slack"


class SlackNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Slack notification channel integration."""

    pass


@runtime_checkable
class SlackNotificationChannelClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class SlackNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Slack notification channel integration.

    The legacy facade (create_slack_integration) remains separate and backward-compatible.
    """

    config: SlackNotificationChannelIntegrationConfig = SlackNotificationChannelIntegrationConfig()
    _client: SlackNotificationChannelClient | None = PrivateAttr(default=None)

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
