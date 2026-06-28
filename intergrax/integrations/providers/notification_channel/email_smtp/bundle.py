# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p2.factories import create_email_smtp_notification_channel

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.notification_channel.email_smtp.integration import (
    EMAIL_SMTP_NOTIFICATION_CHANNEL_PROVIDER_ID,
    EmailSmtpNotificationChannelIntegration,
    EmailSmtpNotificationChannelIntegrationConfig,
    EmailSmtpNotificationChannelClient,
)

__all__ = [
    "create_email_smtp_notification_channel",
    "create_email_smtp_notification_channel_integration",
]


def create_email_smtp_notification_channel_integration(
    *,
    client: EmailSmtpNotificationChannelClient | None = None,
    enabled: bool = False,
) -> EmailSmtpNotificationChannelIntegration:
    """
    Build a contract-based Email Smtp notification channel integration.

    The legacy facade (create_email_smtp_notification_channel) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Email Smtp notification channel integration requires an injected client when enabled=True",
        )
    if client is not None:
        return EmailSmtpNotificationChannelIntegration.from_client(client, enabled=enabled)
    return EmailSmtpNotificationChannelIntegration.for_provider(
        provider_id=EMAIL_SMTP_NOTIFICATION_CHANNEL_PROVIDER_ID,
        display_name="Email Smtp",
        config=EmailSmtpNotificationChannelIntegrationConfig(enabled=enabled),
    )
