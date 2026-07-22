# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slack conversation channel bundle — disabled-safe + production runtime factory."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.conversation_channel import ConversationChannelBackend
from intergrax.integrations.providers.conversation_channel.slack.config import (
    SlackConversationChannelIntegrationConfig,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
    SlackConversationChannelIntegration,
)

__all__ = ["create_slack_conversation_channel_integration"]


def create_slack_conversation_channel_integration(
    *,
    backend: ConversationChannelBackend | None = None,
    enabled: bool = False,
    config: SlackConversationChannelIntegrationConfig | None = None,
) -> SlackConversationChannelIntegration:
    """
    Build a Slack conversation channel integration.

    - ``enabled=False`` (default): no SDK init, no tokens, no network I/O.
    - ``backend=...``: test/injection construction path.
    - ``enabled=True`` without backend: production runtime from ``config`` or env.
    """
    if backend is not None:
        resolved = config or SlackConversationChannelIntegrationConfig(enabled=enabled)
        return SlackConversationChannelIntegration.from_backend(
            backend,
            enabled=enabled,
            config=resolved,
        )
    if enabled:
        resolved = config or SlackConversationChannelIntegrationConfig.from_env(enabled=True)
        if not resolved.enabled:
            raise IntegrationConfigurationError(
                "Slack conversation channel enabled=True requires enabled configuration",
            )
        try:
            return SlackConversationChannelIntegration.from_config(resolved)
        except IntegrationConfigurationError:
            raise
        except Exception as exc:  # noqa: BLE001 — normalize construction failures
            raise IntegrationConfigurationError(
                "Slack conversation channel runtime construction failed "
                "(tokens redacted; check dependency and configuration)",
            ) from exc
    resolved = config or SlackConversationChannelIntegrationConfig(enabled=False)
    return SlackConversationChannelIntegration.for_provider(
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        display_name="Slack",
        config=SlackConversationChannelIntegrationConfig(
            enabled=False,
            app_token=resolved.app_token,
            bot_token=resolved.bot_token,
            api_timeout_seconds=resolved.api_timeout_seconds,
        ),
    )
