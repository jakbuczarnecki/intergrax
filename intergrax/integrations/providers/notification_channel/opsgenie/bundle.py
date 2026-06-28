# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p4.factories import create_opsgenie_notification_channel

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.notification_channel.opsgenie.integration import (
    OPSGENIE_NOTIFICATION_CHANNEL_PROVIDER_ID,
    OpsgenieNotificationChannelIntegration,
    OpsgenieNotificationChannelIntegrationConfig,
    OpsgenieNotificationChannelClient,
)

__all__ = [
    "create_opsgenie_notification_channel",
    "create_opsgenie_notification_channel_integration",
]


def create_opsgenie_notification_channel_integration(
    *,
    client: OpsgenieNotificationChannelClient | None = None,
    enabled: bool = False,
) -> OpsgenieNotificationChannelIntegration:
    """
    Build a contract-based Opsgenie notification channel integration.

    The legacy facade (create_opsgenie_notification_channel) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Opsgenie notification channel integration requires an injected client when enabled=True",
        )
    if client is not None:
        return OpsgenieNotificationChannelIntegration.from_client(client, enabled=enabled)
    return OpsgenieNotificationChannelIntegration.for_provider(
        provider_id=OPSGENIE_NOTIFICATION_CHANNEL_PROVIDER_ID,
        display_name="Opsgenie",
        config=OpsgenieNotificationChannelIntegrationConfig(enabled=enabled),
    )
