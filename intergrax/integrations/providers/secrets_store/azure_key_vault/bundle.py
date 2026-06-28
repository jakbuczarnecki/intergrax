# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p5.factories import create_azure_key_vault_secrets_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.secrets_store.azure_key_vault.integration import (
    AZURE_KEY_VAULT_SECRETS_STORE_PROVIDER_ID,
    AzureKeyVaultSecretsStoreIntegration,
    AzureKeyVaultSecretsStoreIntegrationConfig,
    AzureKeyVaultSecretsStoreClient,
)

__all__ = [
    "create_azure_key_vault_secrets_store",
    "create_azure_key_vault_secrets_store_integration",
]


def create_azure_key_vault_secrets_store_integration(
    *,
    client: AzureKeyVaultSecretsStoreClient | None = None,
    enabled: bool = False,
) -> AzureKeyVaultSecretsStoreIntegration:
    """
    Build a contract-based Azure Key Vault secrets store integration.

    The legacy facade (create_azure_key_vault_secrets_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Azure Key Vault secrets store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return AzureKeyVaultSecretsStoreIntegration.from_client(client, enabled=enabled)
    return AzureKeyVaultSecretsStoreIntegration.for_provider(
        provider_id=AZURE_KEY_VAULT_SECRETS_STORE_PROVIDER_ID,
        display_name="Azure Key Vault",
        config=AzureKeyVaultSecretsStoreIntegrationConfig(enabled=enabled),
    )
