# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "TEAMS_CONVERSATION_CHANNEL_PROVIDER_ID",
    "TeamsConversationChannelIntegration",
    "TeamsConversationChannelIntegrationConfig",
    "create_teams_conversation_channel_integration",
    "register_teams_integration",
]

_BUNDLE_EXPORTS = frozenset({"create_teams_conversation_channel_integration"})
_INTEGRATION_EXPORTS = frozenset(
    {
        "TEAMS_CONVERSATION_CHANNEL_PROVIDER_ID",
        "TeamsConversationChannelIntegration",
        "TeamsConversationChannelIntegrationConfig",
    }
)


def __getattr__(name: str):
    if name == "register_teams_integration":
        from intergrax.integrations.providers.conversation_channel.teams.register import (
            register_teams_integration,
        )

        return register_teams_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.conversation_channel.teams import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.conversation_channel.teams import (
            integration as _integration,
        )

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
