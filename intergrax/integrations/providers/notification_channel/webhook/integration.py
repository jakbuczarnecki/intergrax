# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Webhook notification channel integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

WEBHOOK_NOTIFICATION_CHANNEL_PROVIDER_ID = "webhook"


class WebhookNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Webhook notification channel integration."""

    pass


@runtime_checkable
class WebhookNotificationChannelClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class WebhookNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Webhook notification channel integration.

    The legacy facade (create_webhook_integration) remains separate and backward-compatible.
    """

    config: WebhookNotificationChannelIntegrationConfig = WebhookNotificationChannelIntegrationConfig()
    _client: WebhookNotificationChannelClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: WebhookNotificationChannelClient,
        *,
        enabled: bool = False,
    ) -> WebhookNotificationChannelIntegration:
        integration = cls.for_provider(
            provider_id=WEBHOOK_NOTIFICATION_CHANNEL_PROVIDER_ID,
            display_name="Webhook",
            config=WebhookNotificationChannelIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> WebhookNotificationChannelClient | None:
        return self._client
