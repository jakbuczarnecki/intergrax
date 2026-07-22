# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slack conversation channel bundle — contract factory (no vendor runtime)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.conversation_channel import ConversationChannelBackend
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
    SlackConversationChannelIntegrationConfig,
    SlackConversationChannelIntegration,
)

__all__ = ["create_slack_conversation_channel_integration"]


def create_slack_conversation_channel_integration(
    *,
    backend: ConversationChannelBackend | None = None,
    enabled: bool = False,
) -> SlackConversationChannelIntegration:
    """
    Build a contract-based Slack conversation channel integration.

    Backend must be injected explicitly when enabled=True; disabled by default.
    Runtime vendor binding is not supported yet.
    """
    if enabled and backend is None:
        raise IntegrationConfigurationError(
            "Slack conversation channel integration requires an injected backend when enabled=True",
        )
    if backend is not None:
        return SlackConversationChannelIntegration.from_backend(backend, enabled=enabled)
    return SlackConversationChannelIntegration.for_provider(
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        display_name="Slack",
        config=SlackConversationChannelIntegrationConfig(enabled=enabled),
    )
