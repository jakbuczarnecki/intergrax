# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Mailgun notification channel integration."""

from __future__ import annotations

from typing import Any

from pydantic import PrivateAttr

from intergrax.integrations._shared.health import probe_client_health
from intergrax.integrations.contracts.base import HealthStatus, IntegrationConfigurationError
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig
from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract

MAILGUN_NOTIFICATION_CHANNEL_PROVIDER_ID = "mailgun"


class MailgunNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Mailgun notification channel integration."""

    pass


MailgunNotificationChannelClient = NotificationChannel


class MailgunNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Single public Mailgun notification channel entrypoint.

    Outbound email delivery via ``notify``. Inbound webhook parsing remains a private
    adapter (``MailgunInteractionAdapter``) and is not a provider-category identity.
    """

    config: MailgunNotificationChannelIntegrationConfig = MailgunNotificationChannelIntegrationConfig()
    _client: MailgunNotificationChannelClient | None = PrivateAttr(default=None)

    async def notify(self, message: Any) -> None:
        await self._require_client().notify(message)

    def health(self) -> HealthStatus:
        return probe_client_health(
            self._require_client(),
            slug=MAILGUN_NOTIFICATION_CHANNEL_PROVIDER_ID,
            default_detail="mailgun ready probe",
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
        client: MailgunNotificationChannelClient,
        *,
        enabled: bool = False,
    ) -> MailgunNotificationChannelIntegration:
        integration = cls.for_provider(
            provider_id=MAILGUN_NOTIFICATION_CHANNEL_PROVIDER_ID,
            display_name="Mailgun",
            config=MailgunNotificationChannelIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> MailgunNotificationChannelClient | None:
        return self._client


NotificationChannel.register(MailgunNotificationChannelIntegration)
