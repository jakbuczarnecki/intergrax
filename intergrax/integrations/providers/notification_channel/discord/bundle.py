# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p3.factories import create_discord_notification_channel as _legacy_create_discord_notification_channel

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.notification_channel.discord.integration import (
    DISCORD_NOTIFICATION_CHANNEL_PROVIDER_ID,
    DiscordNotificationChannelIntegration,
    DiscordNotificationChannelIntegrationConfig,
    DiscordNotificationChannelClient,
)

__all__ = [
    "create_discord_notification_channel",
    "create_discord_notification_channel_integration",
]


def create_discord_notification_channel_integration(
    *,
    client: DiscordNotificationChannelClient | None = None,
    enabled: bool = False,
) -> DiscordNotificationChannelIntegration:
    """
    Build a contract-based Discord notification channel integration.

    The legacy facade (create_discord_notification_channel) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Discord notification channel integration requires an injected client when enabled=True",
        )
    if client is not None:
        return DiscordNotificationChannelIntegration.from_client(client, enabled=enabled)
    return DiscordNotificationChannelIntegration.for_provider(
        provider_id=DISCORD_NOTIFICATION_CHANNEL_PROVIDER_ID,
        display_name="Discord",
        config=DiscordNotificationChannelIntegrationConfig(enabled=enabled),
    )


def create_discord_notification_channel(**kwargs: object) -> DiscordNotificationChannelIntegration:
    """Compatibility shim — constructs DiscordNotificationChannelIntegration from legacy runtime."""
    runtime = _legacy_create_discord_notification_channel(**kwargs)
    if isinstance(runtime, DiscordNotificationChannelIntegration):
        return runtime
    return DiscordNotificationChannelIntegration.from_runtime(runtime)
