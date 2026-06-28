# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sendgrid notification channel integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

SENDGRID_NOTIFICATION_CHANNEL_PROVIDER_ID = "sendgrid"


class SendgridNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Sendgrid notification channel integration."""

    pass


@runtime_checkable
class SendgridNotificationChannelClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class SendgridNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Sendgrid notification channel integration.

    The legacy facade (create_sendgrid_notification_channel) remains separate and backward-compatible.
    """

    config: SendgridNotificationChannelIntegrationConfig = SendgridNotificationChannelIntegrationConfig()
    _client: SendgridNotificationChannelClient | None = PrivateAttr(default=None)

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
