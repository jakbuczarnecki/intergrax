# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "SLACK_CONVERSATION_CHANNEL_PROVIDER_ID",
    "SlackConversationChannelIntegration",
    "SlackConversationChannelIntegrationConfig",
    "create_slack_conversation_channel_integration",
    "register_slack_integration",
]

_BUNDLE_EXPORTS = frozenset({"create_slack_conversation_channel_integration"})
_INTEGRATION_EXPORTS = frozenset(
    {
        "SLACK_CONVERSATION_CHANNEL_PROVIDER_ID",
        "SlackConversationChannelIntegration",
        "SlackConversationChannelIntegrationConfig",
    }
)


def __getattr__(name: str):
    if name == "register_slack_integration":
        from intergrax.integrations.providers.conversation_channel.slack.register import (
            register_slack_integration,
        )

        return register_slack_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.conversation_channel.slack import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.conversation_channel.slack import (
            integration as _integration,
        )

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
