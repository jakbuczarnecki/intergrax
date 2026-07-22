# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "GOOGLE_CHAT_CONVERSATION_CHANNEL_PROVIDER_ID",
    "GoogleChatConversationChannelIntegration",
    "GoogleChatConversationChannelIntegrationConfig",
    "create_google_chat_conversation_channel_integration",
    "register_google_chat_integration",
]

_BUNDLE_EXPORTS = frozenset({"create_google_chat_conversation_channel_integration"})
_INTEGRATION_EXPORTS = frozenset(
    {
        "GOOGLE_CHAT_CONVERSATION_CHANNEL_PROVIDER_ID",
        "GoogleChatConversationChannelIntegration",
        "GoogleChatConversationChannelIntegrationConfig",
    }
)


def __getattr__(name: str):
    if name == "register_google_chat_integration":
        from intergrax.integrations.providers.conversation_channel.google_chat.register import (
            register_google_chat_integration,
        )

        return register_google_chat_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.conversation_channel.google_chat import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.conversation_channel.google_chat import (
            integration as _integration,
        )

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
