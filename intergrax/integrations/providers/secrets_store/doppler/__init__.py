# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "DOPPLER_SECRETS_STORE_PROVIDER_ID",
    "DopplerSecretsStoreIntegration",
    "DopplerSecretsStoreIntegrationConfig",
    "DopplerSecretsStoreClient",
    "create_doppler_secrets_store",
    "create_doppler_secrets_store_integration",
    "register_doppler_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_doppler_secrets_store",
        "create_doppler_secrets_store_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "DOPPLER_SECRETS_STORE_PROVIDER_ID",
        "DopplerSecretsStoreIntegration",
        "DopplerSecretsStoreIntegrationConfig",
        "DopplerSecretsStoreClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "DOPPLER_SECRETS_STORE_PROVIDER_ID",
        "DopplerSecretsStoreIntegration",
        "DopplerSecretsStoreIntegrationConfig",
        "DopplerSecretsStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_doppler_integration":
        from intergrax.integrations.providers.secrets_store.doppler.register import register_doppler_integration

        return register_doppler_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.secrets_store.doppler import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.secrets_store.doppler import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.secrets_store.doppler import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
