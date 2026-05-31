# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Slack integration — single public entry for Slack notifications and interactions.

Implementation lives under ``runtime/notifications`` and ``runtime/interactions``;
compose only through this package.
"""

from intergrax.integrations.providers.notification_channel.slack.config import (
    DEFAULT_WEBHOOK_URL,
    ENV_SLACK_SIGNING_SECRET,
    ENV_SLACK_WEBHOOK_URL,
    SlackIntegrationConfig,
)

__all__ = [
    "DEFAULT_WEBHOOK_URL",
    "ENV_SLACK_SIGNING_SECRET",
    "ENV_SLACK_WEBHOOK_URL",
    "SlackIntegrationBundle",
    "SlackIntegrationConfig",
    "SlackInteractionAdapter",
    "create_slack_catalog_factory",
    "create_slack_integration",
    "create_slack_interaction_surface",
    "create_slack_notification_channel",
    "create_slack_signature_verifier",
    "register_slack_integration",
    "resolve_slack_config",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "SlackIntegrationBundle",
        "create_slack_catalog_factory",
        "create_slack_integration",
        "create_slack_interaction_surface",
        "create_slack_notification_channel",
        "create_slack_signature_verifier",
        "resolve_slack_config",
    }
)


def __getattr__(name: str):
    if name == "register_slack_integration":
        from intergrax.integrations.providers.notification_channel.slack.register import register_slack_integration

        return register_slack_integration
    if name == "SlackInteractionAdapter":
        from intergrax.integrations.providers.notification_channel.slack.adapter import SlackInteractionAdapter

        return SlackInteractionAdapter
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.notification_channel.slack import bundle as _bundle

        return getattr(_bundle, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
