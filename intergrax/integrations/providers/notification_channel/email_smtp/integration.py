# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Email Smtp notification channel integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

EMAIL_SMTP_NOTIFICATION_CHANNEL_PROVIDER_ID = "email_smtp"


class EmailSmtpNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Email Smtp notification channel integration."""

    pass


@runtime_checkable
class EmailSmtpNotificationChannelClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class EmailSmtpNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Email Smtp notification channel integration.

    The legacy facade (create_email_smtp_notification_channel) remains separate and backward-compatible.
    """

    config: EmailSmtpNotificationChannelIntegrationConfig = EmailSmtpNotificationChannelIntegrationConfig()
    _client: EmailSmtpNotificationChannelClient | None = PrivateAttr(default=None)

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
