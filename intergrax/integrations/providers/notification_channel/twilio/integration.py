# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Twilio notification channel integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

TWILIO_NOTIFICATION_CHANNEL_PROVIDER_ID = "twilio"


class TwilioNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Twilio notification channel integration."""

    pass


@runtime_checkable
class TwilioNotificationChannelClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class TwilioNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Twilio notification channel integration.

    The legacy facade (create_twilio_notification_channel) remains separate and backward-compatible.
    """

    config: TwilioNotificationChannelIntegrationConfig = TwilioNotificationChannelIntegrationConfig()
    _client: TwilioNotificationChannelClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: TwilioNotificationChannelClient,
        *,
        enabled: bool = False,
    ) -> TwilioNotificationChannelIntegration:
        integration = cls.for_provider(
            provider_id=TWILIO_NOTIFICATION_CHANNEL_PROVIDER_ID,
            display_name="Twilio",
            config=TwilioNotificationChannelIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> TwilioNotificationChannelClient | None:
        return self._client
