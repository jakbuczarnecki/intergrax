# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "TELEGRAM_CONVERSATION_CHANNEL_PROVIDER_ID",
    "TelegramConversationChannelIntegration",
    "TelegramConversationChannelIntegrationConfig",
    "create_telegram_conversation_channel_integration",
    "register_telegram_integration",
]

_BUNDLE_EXPORTS = frozenset({"create_telegram_conversation_channel_integration"})
_INTEGRATION_EXPORTS = frozenset(
    {
        "TELEGRAM_CONVERSATION_CHANNEL_PROVIDER_ID",
        "TelegramConversationChannelIntegration",
        "TelegramConversationChannelIntegrationConfig",
    }
)


def __getattr__(name: str):
    if name == "register_telegram_integration":
        from intergrax.integrations.providers.conversation_channel.telegram.register import (
            register_telegram_integration,
        )

        return register_telegram_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.conversation_channel.telegram import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.conversation_channel.telegram import (
            integration as _integration,
        )

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
