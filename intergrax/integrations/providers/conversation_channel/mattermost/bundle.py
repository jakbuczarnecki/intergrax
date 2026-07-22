# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Mattermost conversation channel bundle — contract factory (no vendor runtime)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.conversation_channel import ConversationChannelBackend
from intergrax.integrations.providers.conversation_channel.mattermost.integration import (
    MATTERMOST_CONVERSATION_CHANNEL_PROVIDER_ID,
    MattermostConversationChannelIntegrationConfig,
    MattermostConversationChannelIntegration,
)

__all__ = ["create_mattermost_conversation_channel_integration"]


def create_mattermost_conversation_channel_integration(
    *,
    backend: ConversationChannelBackend | None = None,
    enabled: bool = False,
) -> MattermostConversationChannelIntegration:
    """
    Build a contract-based Mattermost conversation channel integration.

    Backend must be injected explicitly when enabled=True; disabled by default.
    Runtime vendor binding is not supported yet.
    """
    if enabled and backend is None:
        raise IntegrationConfigurationError(
            "Mattermost conversation channel integration requires an injected backend when enabled=True",
        )
    if backend is not None:
        return MattermostConversationChannelIntegration.from_backend(backend, enabled=enabled)
    return MattermostConversationChannelIntegration.for_provider(
        provider_id=MATTERMOST_CONVERSATION_CHANNEL_PROVIDER_ID,
        display_name="Mattermost",
        config=MattermostConversationChannelIntegrationConfig(enabled=enabled),
    )
