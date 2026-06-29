# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "AZURE_KEY_VAULT_SECRETS_STORE_PROVIDER_ID",
    "AzureKeyVaultSecretsStoreIntegration",
    "AzureKeyVaultSecretsStoreIntegrationConfig",
    "AzureKeyVaultSecretsStoreClient",
    "create_azure_key_vault_secrets_store",
    "create_azure_key_vault_secrets_store_integration",
    "register_azure_key_vault_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_azure_key_vault_secrets_store",
        "create_azure_key_vault_secrets_store_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "AZURE_KEY_VAULT_SECRETS_STORE_PROVIDER_ID",
        "AzureKeyVaultSecretsStoreIntegration",
        "AzureKeyVaultSecretsStoreIntegrationConfig",
        "AzureKeyVaultSecretsStoreClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "AZURE_KEY_VAULT_SECRETS_STORE_PROVIDER_ID",
        "AzureKeyVaultSecretsStoreIntegration",
        "AzureKeyVaultSecretsStoreIntegrationConfig",
        "AzureKeyVaultSecretsStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_azure_key_vault_integration":
        from intergrax.integrations.providers.secrets_store.azure_key_vault.register import register_azure_key_vault_integration

        return register_azure_key_vault_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.secrets_store.azure_key_vault import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.secrets_store.azure_key_vault import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.secrets_store.azure_key_vault import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
