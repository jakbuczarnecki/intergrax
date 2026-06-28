# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "WORKOS_IDENTITY_PROVIDER_PROVIDER_ID",
    "WorkosIdentityProviderIntegration",
    "WorkosIdentityProviderIntegrationConfig",
    "WorkosIdentityProviderClient",
    "create_workos_identity_provider",
    "create_workos_identity_provider_integration",
    "register_workos_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_workos_identity_provider",
        "create_workos_identity_provider_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "WORKOS_IDENTITY_PROVIDER_PROVIDER_ID",
        "WorkosIdentityProviderIntegration",
        "WorkosIdentityProviderIntegrationConfig",
        "WorkosIdentityProviderClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "WORKOS_IDENTITY_PROVIDER_PROVIDER_ID",
        "WorkosIdentityProviderIntegration",
        "WorkosIdentityProviderIntegrationConfig",
        "WorkosIdentityProviderClient",
    }
)

def __getattr__(name: str):
    if name == "register_workos_integration":
        from intergrax.integrations.providers.identity_provider.workos.register import register_workos_integration

        return register_workos_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.identity_provider.workos import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.identity_provider.workos import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.identity_provider.workos import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
