# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Twilio notification channel integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence, Any

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

TWILIO_NOTIFICATION_CHANNEL_PROVIDER_ID = "twilio"


class TwilioNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Twilio notification channel integration."""

    pass


TwilioNotificationChannelClient = NotificationChannel

class TwilioNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Single public Twilio notification channel entrypoint.

    Legacy catalog factory (create_twilio_notification_channel) owns catalog behavior; legacy factories use from_client().
    """

    config: TwilioNotificationChannelIntegrationConfig = TwilioNotificationChannelIntegrationConfig()
    _client: TwilioNotificationChannelClient | None = PrivateAttr(default=None)
    


    async def notify(self, message: Any) -> None:
        await self._require_client().notify(message)

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

NotificationChannel.register(TwilioNotificationChannelIntegration)
