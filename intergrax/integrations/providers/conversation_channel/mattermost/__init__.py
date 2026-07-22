# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "MATTERMOST_CONVERSATION_CHANNEL_PROVIDER_ID",
    "MattermostConversationChannelIntegration",
    "MattermostConversationChannelIntegrationConfig",
    "create_mattermost_conversation_channel_integration",
    "register_mattermost_integration",
]

_BUNDLE_EXPORTS = frozenset({"create_mattermost_conversation_channel_integration"})
_INTEGRATION_EXPORTS = frozenset(
    {
        "MATTERMOST_CONVERSATION_CHANNEL_PROVIDER_ID",
        "MattermostConversationChannelIntegration",
        "MattermostConversationChannelIntegrationConfig",
    }
)


def __getattr__(name: str):
    if name == "register_mattermost_integration":
        from intergrax.integrations.providers.conversation_channel.mattermost.register import (
            register_mattermost_integration,
        )

        return register_mattermost_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.conversation_channel.mattermost import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.conversation_channel.mattermost import (
            integration as _integration,
        )

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
