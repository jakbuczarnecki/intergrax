# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "MAILGUN_INTERACTION_SURFACE_PROVIDER_ID",
    "MailgunInteractionSurfaceIntegration",
    "MailgunInteractionSurfaceIntegrationConfig",
    "MailgunInteractionSurfaceClient",
    "create_mailgun_interaction_surface",
    "create_mailgun_interaction_surface_integration",
    "register_mailgun_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_mailgun_interaction_surface",
        "create_mailgun_interaction_surface_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "MAILGUN_INTERACTION_SURFACE_PROVIDER_ID",
        "MailgunInteractionSurfaceIntegration",
        "MailgunInteractionSurfaceIntegrationConfig",
        "MailgunInteractionSurfaceClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "MAILGUN_INTERACTION_SURFACE_PROVIDER_ID",
        "MailgunInteractionSurfaceIntegration",
        "MailgunInteractionSurfaceIntegrationConfig",
        "MailgunInteractionSurfaceClient",
    }
)

def __getattr__(name: str):
    if name == "register_mailgun_integration":
        from intergrax.integrations.providers.interaction_surface.mailgun.register import register_mailgun_integration

        return register_mailgun_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.interaction_surface.mailgun import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.interaction_surface.mailgun import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.interaction_surface.mailgun import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
