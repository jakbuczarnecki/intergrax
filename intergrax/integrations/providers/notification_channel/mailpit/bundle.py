# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p6.factories import create_mailpit_notification_channel as _legacy_create_mailpit_notification_channel

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.notification_channel.mailpit.integration import (
    MAILPIT_NOTIFICATION_CHANNEL_PROVIDER_ID,
    MailpitNotificationChannelIntegration,
    MailpitNotificationChannelIntegrationConfig,
    MailpitNotificationChannelClient,
)

__all__ = [
    "create_mailpit_notification_channel",
    "create_mailpit_notification_channel_integration",
]


def create_mailpit_notification_channel_integration(
    *,
    client: MailpitNotificationChannelClient | None = None,
    enabled: bool = False,
) -> MailpitNotificationChannelIntegration:
    """
    Build a contract-based Mailpit notification channel integration.

    The legacy facade (create_mailpit_notification_channel) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Mailpit notification channel integration requires an injected client when enabled=True",
        )
    if client is not None:
        return MailpitNotificationChannelIntegration.from_client(client, enabled=enabled)
    return MailpitNotificationChannelIntegration.for_provider(
        provider_id=MAILPIT_NOTIFICATION_CHANNEL_PROVIDER_ID,
        display_name="Mailpit",
        config=MailpitNotificationChannelIntegrationConfig(enabled=enabled),
    )


def create_mailpit_notification_channel(**kwargs: object) -> MailpitNotificationChannelIntegration:
    """Compatibility shim — constructs MailpitNotificationChannelIntegration from legacy runtime."""
    runtime = _legacy_create_mailpit_notification_channel(**kwargs)
    if isinstance(runtime, MailpitNotificationChannelIntegration):
        return runtime
    return MailpitNotificationChannelIntegration.from_client(runtime)
