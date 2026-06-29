# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "OKTA_IDENTITY_PROVIDER_PROVIDER_ID",
    "OktaIdentityProviderIntegration",
    "OktaIdentityProviderIntegrationConfig",
    "OktaIdentityProviderClient",
    "create_okta_identity_provider",
    "create_okta_identity_provider_integration",
    "register_okta_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_okta_identity_provider",
        "create_okta_identity_provider_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "OKTA_IDENTITY_PROVIDER_PROVIDER_ID",
        "OktaIdentityProviderIntegration",
        "OktaIdentityProviderIntegrationConfig",
        "OktaIdentityProviderClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "OKTA_IDENTITY_PROVIDER_PROVIDER_ID",
        "OktaIdentityProviderIntegration",
        "OktaIdentityProviderIntegrationConfig",
        "OktaIdentityProviderClient",
    }
)

def __getattr__(name: str):
    if name == "register_okta_integration":
        from intergrax.integrations.providers.identity_provider.okta.register import register_okta_integration

        return register_okta_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.identity_provider.okta import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.identity_provider.okta import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.identity_provider.okta import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
