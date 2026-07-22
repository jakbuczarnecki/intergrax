# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p5.factories import (
    create_mailgun_notification_channel as _legacy_create_mailgun_notification_channel,
)
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.notification_channel.mailgun.integration import (
    MAILGUN_NOTIFICATION_CHANNEL_PROVIDER_ID,
    MailgunNotificationChannelClient,
    MailgunNotificationChannelIntegration,
    MailgunNotificationChannelIntegrationConfig,
)

__all__ = [
    "create_mailgun_notification_channel",
    "create_mailgun_notification_channel_integration",
]


def create_mailgun_notification_channel_integration(
    *,
    client: MailgunNotificationChannelClient | None = None,
    enabled: bool = False,
) -> MailgunNotificationChannelIntegration:
    """
    Build a contract-based Mailgun notification channel integration.

    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Mailgun notification channel integration requires an injected client when enabled=True",
        )
    if client is not None:
        return MailgunNotificationChannelIntegration.from_client(client, enabled=enabled)
    return MailgunNotificationChannelIntegration.for_provider(
        provider_id=MAILGUN_NOTIFICATION_CHANNEL_PROVIDER_ID,
        display_name="Mailgun",
        config=MailgunNotificationChannelIntegrationConfig(enabled=enabled),
    )


def create_mailgun_notification_channel(**kwargs: object) -> MailgunNotificationChannelIntegration:
    """Compatibility shim — constructs MailgunNotificationChannelIntegration from legacy runtime."""
    runtime = _legacy_create_mailgun_notification_channel(**kwargs)
    if isinstance(runtime, MailgunNotificationChannelIntegration):
        return runtime
    return MailgunNotificationChannelIntegration.from_client(runtime)
