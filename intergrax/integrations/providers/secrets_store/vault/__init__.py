# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "VAULT_SECRETS_STORE_PROVIDER_ID",
    "VaultSecretsStoreIntegration",
    "VaultSecretsStoreIntegrationConfig",
    "VaultSecretsStoreClient",
    "create_vault_secrets_store",
    "create_vault_secrets_store_integration",
    "register_vault_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_vault_secrets_store",
        "create_vault_secrets_store_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "VAULT_SECRETS_STORE_PROVIDER_ID",
        "VaultSecretsStoreIntegration",
        "VaultSecretsStoreIntegrationConfig",
        "VaultSecretsStoreClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "VAULT_SECRETS_STORE_PROVIDER_ID",
        "VaultSecretsStoreIntegration",
        "VaultSecretsStoreIntegrationConfig",
        "VaultSecretsStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_vault_integration":
        from intergrax.integrations.providers.secrets_store.vault.register import register_vault_integration

        return register_vault_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.secrets_store.vault import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.secrets_store.vault import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.secrets_store.vault import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
