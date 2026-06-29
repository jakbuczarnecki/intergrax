# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "KEYCLOAK_IDENTITY_PROVIDER_PROVIDER_ID",
    "KeycloakIdentityProviderIntegration",
    "KeycloakIdentityProviderIntegrationConfig",
    "KeycloakIdentityProviderClient",
    "create_keycloak_identity_provider",
    "create_keycloak_identity_provider_integration",
    "register_keycloak_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_keycloak_identity_provider",
        "create_keycloak_identity_provider_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "KEYCLOAK_IDENTITY_PROVIDER_PROVIDER_ID",
        "KeycloakIdentityProviderIntegration",
        "KeycloakIdentityProviderIntegrationConfig",
        "KeycloakIdentityProviderClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "KEYCLOAK_IDENTITY_PROVIDER_PROVIDER_ID",
        "KeycloakIdentityProviderIntegration",
        "KeycloakIdentityProviderIntegrationConfig",
        "KeycloakIdentityProviderClient",
    }
)

def __getattr__(name: str):
    if name == "register_keycloak_integration":
        from intergrax.integrations.providers.identity_provider.keycloak.register import register_keycloak_integration

        return register_keycloak_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.identity_provider.keycloak import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.identity_provider.keycloak import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.identity_provider.keycloak import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
