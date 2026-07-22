# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Rocket.Chat conversation channel bundle — contract factory (no vendor runtime)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.conversation_channel import ConversationChannelBackend
from intergrax.integrations.providers.conversation_channel.rocket_chat.integration import (
    ROCKET_CHAT_CONVERSATION_CHANNEL_PROVIDER_ID,
    RocketChatConversationChannelIntegrationConfig,
    RocketChatConversationChannelIntegration,
)

__all__ = ["create_rocket_chat_conversation_channel_integration"]


def create_rocket_chat_conversation_channel_integration(
    *,
    backend: ConversationChannelBackend | None = None,
    enabled: bool = False,
) -> RocketChatConversationChannelIntegration:
    """
    Build a contract-based Rocket.Chat conversation channel integration.

    Backend must be injected explicitly when enabled=True; disabled by default.
    Runtime vendor binding is not supported yet.
    """
    if enabled and backend is None:
        raise IntegrationConfigurationError(
            "Rocket.Chat conversation channel integration requires an injected backend when enabled=True",
        )
    if backend is not None:
        return RocketChatConversationChannelIntegration.from_backend(backend, enabled=enabled)
    return RocketChatConversationChannelIntegration.for_provider(
        provider_id=ROCKET_CHAT_CONVERSATION_CHANNEL_PROVIDER_ID,
        display_name="Rocket.Chat",
        config=RocketChatConversationChannelIntegrationConfig(enabled=enabled),
    )
