# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Teams integration — single public entry for Microsoft Teams notifications and interactions.

Implementation lives under ``runtime/notifications`` and ``runtime/interactions``;
compose only through this package.
"""

from intergrax.integrations.providers.notification_channel.teams.config import (
    DEFAULT_WEBHOOK_URL,
    ENV_TEAMS_SECURITY_TOKEN,
    ENV_TEAMS_WEBHOOK_URL,
    TeamsIntegrationConfig,
)

__all__ = [
    "DEFAULT_WEBHOOK_URL",
    "ENV_TEAMS_SECURITY_TOKEN",
    "ENV_TEAMS_WEBHOOK_URL",
    "TeamsIntegrationBundle",
    "TeamsIntegrationConfig",
    "TeamsInteractionAdapter",
    "create_teams_catalog_factory",
    "create_teams_integration",
    "create_teams_interaction_surface",
    "create_teams_notification_channel",
    "create_teams_signature_verifier",
    "register_teams_integration",
    "resolve_teams_config",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "TeamsIntegrationBundle",
        "create_teams_catalog_factory",
        "create_teams_integration",
        "create_teams_interaction_surface",
        "create_teams_notification_channel",
        "create_teams_signature_verifier",
        "resolve_teams_config",
    }
)


def __getattr__(name: str):
    if name == "register_teams_integration":
        from intergrax.integrations.providers.notification_channel.teams.register import register_teams_integration

        return register_teams_integration
    if name == "TeamsInteractionAdapter":
        from intergrax.integrations.providers.notification_channel.teams.adapter import TeamsInteractionAdapter

        return TeamsInteractionAdapter
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.notification_channel.teams import bundle as _bundle

        return getattr(_bundle, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
