# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p3.factories import create_vault_secrets_store as _legacy_create_vault_secrets_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.secrets_store.vault.integration import (
    VAULT_SECRETS_STORE_PROVIDER_ID,
    VaultSecretsStoreIntegration,
    VaultSecretsStoreIntegrationConfig,
    VaultSecretsStoreClient,
)

__all__ = [
    "create_vault_secrets_store",
    "create_vault_secrets_store_integration",
]


def create_vault_secrets_store_integration(
    *,
    client: VaultSecretsStoreClient | None = None,
    enabled: bool = False,
) -> VaultSecretsStoreIntegration:
    """
    Build a contract-based Vault secrets store integration.

    The legacy facade (create_vault_secrets_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Vault secrets store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return VaultSecretsStoreIntegration.from_client(client, enabled=enabled)
    return VaultSecretsStoreIntegration.for_provider(
        provider_id=VAULT_SECRETS_STORE_PROVIDER_ID,
        display_name="Vault",
        config=VaultSecretsStoreIntegrationConfig(enabled=enabled),
    )


def create_vault_secrets_store(**kwargs: object) -> VaultSecretsStoreIntegration:
    """Compatibility shim — constructs VaultSecretsStoreIntegration from legacy runtime."""
    runtime = _legacy_create_vault_secrets_store(**kwargs)
    if isinstance(runtime, VaultSecretsStoreIntegration):
        return runtime
    return VaultSecretsStoreIntegration.from_client(runtime)
