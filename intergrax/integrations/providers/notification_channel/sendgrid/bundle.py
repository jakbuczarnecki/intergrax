# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p5.factories import create_sendgrid_notification_channel as _legacy_create_sendgrid_notification_channel

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.notification_channel.sendgrid.integration import (
    SENDGRID_NOTIFICATION_CHANNEL_PROVIDER_ID,
    SendgridNotificationChannelIntegration,
    SendgridNotificationChannelIntegrationConfig,
    SendgridNotificationChannelClient,
)

__all__ = [
    "create_sendgrid_notification_channel",
    "create_sendgrid_notification_channel_integration",
]


def create_sendgrid_notification_channel_integration(
    *,
    client: SendgridNotificationChannelClient | None = None,
    enabled: bool = False,
) -> SendgridNotificationChannelIntegration:
    """
    Build a contract-based Sendgrid notification channel integration.

    The legacy facade (create_sendgrid_notification_channel) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Sendgrid notification channel integration requires an injected client when enabled=True",
        )
    if client is not None:
        return SendgridNotificationChannelIntegration.from_client(client, enabled=enabled)
    return SendgridNotificationChannelIntegration.for_provider(
        provider_id=SENDGRID_NOTIFICATION_CHANNEL_PROVIDER_ID,
        display_name="Sendgrid",
        config=SendgridNotificationChannelIntegrationConfig(enabled=enabled),
    )


def create_sendgrid_notification_channel(**kwargs: object) -> SendgridNotificationChannelIntegration:
    """Compatibility shim — constructs SendgridNotificationChannelIntegration from legacy runtime."""
    runtime = _legacy_create_sendgrid_notification_channel(**kwargs)
    if isinstance(runtime, SendgridNotificationChannelIntegration):
        return runtime
    return SendgridNotificationChannelIntegration.from_runtime(runtime)
