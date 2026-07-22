# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Telegram conversation channel bundle — contract factory (no vendor runtime)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.conversation_channel import ConversationChannelBackend
from intergrax.integrations.providers.conversation_channel.telegram.integration import (
    TELEGRAM_CONVERSATION_CHANNEL_PROVIDER_ID,
    TelegramConversationChannelIntegrationConfig,
    TelegramConversationChannelIntegration,
)

__all__ = ["create_telegram_conversation_channel_integration"]


def create_telegram_conversation_channel_integration(
    *,
    backend: ConversationChannelBackend | None = None,
    enabled: bool = False,
) -> TelegramConversationChannelIntegration:
    """
    Build a contract-based Telegram conversation channel integration.

    Backend must be injected explicitly when enabled=True; disabled by default.
    Runtime vendor binding is not supported yet.
    """
    if enabled and backend is None:
        raise IntegrationConfigurationError(
            "Telegram conversation channel integration requires an injected backend when enabled=True",
        )
    if backend is not None:
        return TelegramConversationChannelIntegration.from_backend(backend, enabled=enabled)
    return TelegramConversationChannelIntegration.for_provider(
        provider_id=TELEGRAM_CONVERSATION_CHANNEL_PROVIDER_ID,
        display_name="Telegram",
        config=TelegramConversationChannelIntegrationConfig(enabled=enabled),
    )
