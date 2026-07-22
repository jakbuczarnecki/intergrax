# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Teams conversation channel bundle — contract factory (no vendor runtime)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.conversation_channel import ConversationChannelBackend
from intergrax.integrations.providers.conversation_channel.teams.integration import (
    TEAMS_CONVERSATION_CHANNEL_PROVIDER_ID,
    TeamsConversationChannelIntegrationConfig,
    TeamsConversationChannelIntegration,
)

__all__ = ["create_teams_conversation_channel_integration"]


def create_teams_conversation_channel_integration(
    *,
    backend: ConversationChannelBackend | None = None,
    enabled: bool = False,
) -> TeamsConversationChannelIntegration:
    """
    Build a contract-based Teams conversation channel integration.

    Backend must be injected explicitly when enabled=True; disabled by default.
    Runtime vendor binding is not supported yet.
    """
    if enabled and backend is None:
        raise IntegrationConfigurationError(
            "Teams conversation channel integration requires an injected backend when enabled=True",
        )
    if backend is not None:
        return TeamsConversationChannelIntegration.from_backend(backend, enabled=enabled)
    return TeamsConversationChannelIntegration.for_provider(
        provider_id=TEAMS_CONVERSATION_CHANNEL_PROVIDER_ID,
        display_name="Teams",
        config=TeamsConversationChannelIntegrationConfig(enabled=enabled),
    )
