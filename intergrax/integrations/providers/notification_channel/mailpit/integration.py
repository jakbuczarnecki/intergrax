# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Mailpit notification channel integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence, Any

from pydantic import PrivateAttr

from intergrax.integrations._shared.health import probe_client_health
from intergrax.integrations.contracts.base import HealthStatus, IntegrationConfigurationError
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

MAILPIT_NOTIFICATION_CHANNEL_PROVIDER_ID = "mailpit"


class MailpitNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Mailpit notification channel integration."""

    pass


MailpitNotificationChannelClient = NotificationChannel

class MailpitNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Single public Mailpit notification channel entrypoint.

    Legacy catalog factory (create_mailpit_notification_channel) owns catalog behavior; legacy factories use from_client().
    """

    config: MailpitNotificationChannelIntegrationConfig = MailpitNotificationChannelIntegrationConfig()
    _client: MailpitNotificationChannelClient | None = PrivateAttr(default=None)
    


    async def notify(self, message: Any) -> None:
        await self._require_client().notify(message)

    def health(self) -> HealthStatus:
        return probe_client_health(
            self._require_client(),
            slug=MAILPIT_NOTIFICATION_CHANNEL_PROVIDER_ID,
            default_detail="mailpit ready probe",
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

NotificationChannel.register(MailpitNotificationChannelIntegration)
