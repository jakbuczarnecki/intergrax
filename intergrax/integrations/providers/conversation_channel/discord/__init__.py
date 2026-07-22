# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "DISCORD_CONVERSATION_CHANNEL_PROVIDER_ID",
    "DiscordConversationChannelIntegration",
    "DiscordConversationChannelIntegrationConfig",
    "create_discord_conversation_channel_integration",
    "register_discord_integration",
]

_BUNDLE_EXPORTS = frozenset({"create_discord_conversation_channel_integration"})
_INTEGRATION_EXPORTS = frozenset(
    {
        "DISCORD_CONVERSATION_CHANNEL_PROVIDER_ID",
        "DiscordConversationChannelIntegration",
        "DiscordConversationChannelIntegrationConfig",
    }
)


def __getattr__(name: str):
    if name == "register_discord_integration":
        from intergrax.integrations.providers.conversation_channel.discord.register import (
            register_discord_integration,
        )

        return register_discord_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.conversation_channel.discord import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.conversation_channel.discord import (
            integration as _integration,
        )

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
