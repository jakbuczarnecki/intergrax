# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Email Smtp notification channel integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence, Any

from pydantic import PrivateAttr

from intergrax.integrations._shared.health import probe_client_health
from intergrax.integrations.contracts.base import HealthStatus, IntegrationConfigurationError
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

EMAIL_SMTP_NOTIFICATION_CHANNEL_PROVIDER_ID = "email_smtp"


class EmailSmtpNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Email Smtp notification channel integration."""

    pass


EmailSmtpNotificationChannelClient = NotificationChannel

class EmailSmtpNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Single public Email Smtp notification channel entrypoint.

    Legacy catalog factory (create_email_smtp_notification_channel) owns catalog behavior; legacy factories use from_client().
    """

    config: EmailSmtpNotificationChannelIntegrationConfig = EmailSmtpNotificationChannelIntegrationConfig()
    _client: EmailSmtpNotificationChannelClient | None = PrivateAttr(default=None)
    


    async def notify(self, message: Any) -> None:
        await self._require_client().notify(message)

    def health(self) -> HealthStatus:
        return probe_client_health(
            self._require_client(),
            slug=EMAIL_SMTP_NOTIFICATION_CHANNEL_PROVIDER_ID,
            default_detail="email_smtp ready probe",
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
        client: EmailSmtpNotificationChannelClient,
        *,
        enabled: bool = False,
    ) -> EmailSmtpNotificationChannelIntegration:
        integration = cls.for_provider(
            provider_id=EMAIL_SMTP_NOTIFICATION_CHANNEL_PROVIDER_ID,
            display_name="Email Smtp",
            config=EmailSmtpNotificationChannelIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> EmailSmtpNotificationChannelClient | None:
        return self._client

NotificationChannel.register(EmailSmtpNotificationChannelIntegration)
