# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "SLASH_COMMAND_INTERACTION_SURFACE_PROVIDER_ID",
    "SlashCommandInteractionSurfaceIntegration",
    "SlashCommandInteractionSurfaceIntegrationConfig",
    "SlashCommandInteractionSurfaceClient",
    "create_slash_command_integration",
    "create_slash_command_interaction_surface",
    "create_slash_command_interaction_surface_integration",
    "register_slash_command_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_slash_command_integration",
        "create_slash_command_interaction_surface",
        "create_slash_command_interaction_surface_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "SLASH_COMMAND_INTERACTION_SURFACE_PROVIDER_ID",
        "SlashCommandInteractionSurfaceIntegration",
        "SlashCommandInteractionSurfaceIntegrationConfig",
        "SlashCommandInteractionSurfaceClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "SLASH_COMMAND_INTERACTION_SURFACE_PROVIDER_ID",
        "SlashCommandInteractionSurfaceIntegration",
        "SlashCommandInteractionSurfaceIntegrationConfig",
        "SlashCommandInteractionSurfaceClient",
    }
)

def __getattr__(name: str):
    if name == "register_slash_command_integration":
        from intergrax.integrations.providers.interaction_surface.slash_command.register import register_slash_command_integration

        return register_slash_command_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.interaction_surface.slash_command import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.interaction_surface.slash_command import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.interaction_surface.slash_command import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
