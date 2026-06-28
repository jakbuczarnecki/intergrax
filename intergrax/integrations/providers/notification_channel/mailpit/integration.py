# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Mailpit notification channel integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

MAILPIT_NOTIFICATION_CHANNEL_PROVIDER_ID = "mailpit"


class MailpitNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Mailpit notification channel integration."""

    pass


@runtime_checkable
class MailpitNotificationChannelClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class MailpitNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Mailpit notification channel integration.

    The legacy facade (create_mailpit_notification_channel) remains separate and backward-compatible.
    """

    config: MailpitNotificationChannelIntegrationConfig = MailpitNotificationChannelIntegrationConfig()
    _client: MailpitNotificationChannelClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: MailpitNotificationChannelClient,
        *,
        enabled: bool = False,
    ) -> MailpitNotificationChannelIntegration:
        integration = cls.for_provider(
            provider_id=MAILPIT_NOTIFICATION_CHANNEL_PROVIDER_ID,
            display_name="Mailpit",
            config=MailpitNotificationChannelIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> MailpitNotificationChannelClient | None:
        return self._client
