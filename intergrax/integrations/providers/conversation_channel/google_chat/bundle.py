# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Chat conversation channel bundle — contract factory (no vendor runtime)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.conversation_channel import ConversationChannelBackend
from intergrax.integrations.providers.conversation_channel.google_chat.integration import (
    GOOGLE_CHAT_CONVERSATION_CHANNEL_PROVIDER_ID,
    GoogleChatConversationChannelIntegrationConfig,
    GoogleChatConversationChannelIntegration,
)

__all__ = ["create_google_chat_conversation_channel_integration"]


def create_google_chat_conversation_channel_integration(
    *,
    backend: ConversationChannelBackend | None = None,
    enabled: bool = False,
) -> GoogleChatConversationChannelIntegration:
    """
    Build a contract-based Google Chat conversation channel integration.

    Backend must be injected explicitly when enabled=True; disabled by default.
    Runtime vendor binding is not supported yet.
    """
    if enabled and backend is None:
        raise IntegrationConfigurationError(
            "Google Chat conversation channel integration requires an injected backend when enabled=True",
        )
    if backend is not None:
        return GoogleChatConversationChannelIntegration.from_backend(backend, enabled=enabled)
    return GoogleChatConversationChannelIntegration.for_provider(
        provider_id=GOOGLE_CHAT_CONVERSATION_CHANNEL_PROVIDER_ID,
        display_name="Google Chat",
        config=GoogleChatConversationChannelIntegrationConfig(enabled=enabled),
    )
