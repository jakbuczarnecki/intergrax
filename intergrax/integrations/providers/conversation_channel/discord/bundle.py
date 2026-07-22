# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Discord conversation channel bundle — contract factory (no vendor runtime)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.conversation_channel import ConversationChannelBackend
from intergrax.integrations.providers.conversation_channel.discord.integration import (
    DISCORD_CONVERSATION_CHANNEL_PROVIDER_ID,
    DiscordConversationChannelIntegrationConfig,
    DiscordConversationChannelIntegration,
)

__all__ = ["create_discord_conversation_channel_integration"]


def create_discord_conversation_channel_integration(
    *,
    backend: ConversationChannelBackend | None = None,
    enabled: bool = False,
) -> DiscordConversationChannelIntegration:
    """
    Build a contract-based Discord conversation channel integration.

    Backend must be injected explicitly when enabled=True; disabled by default.
    Runtime vendor binding is not supported yet.
    """
    if enabled and backend is None:
        raise IntegrationConfigurationError(
            "Discord conversation channel integration requires an injected backend when enabled=True",
        )
    if backend is not None:
        return DiscordConversationChannelIntegration.from_backend(backend, enabled=enabled)
    return DiscordConversationChannelIntegration.for_provider(
        provider_id=DISCORD_CONVERSATION_CHANNEL_PROVIDER_ID,
        display_name="Discord",
        config=DiscordConversationChannelIntegrationConfig(enabled=enabled),
    )
