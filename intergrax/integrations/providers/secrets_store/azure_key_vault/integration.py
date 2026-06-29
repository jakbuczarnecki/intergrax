# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Azure Key Vault secrets store integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.secrets_store import SecretsStore
from intergrax.runtime.integrations.categories.security import SecretsStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

AZURE_KEY_VAULT_SECRETS_STORE_PROVIDER_ID = "azure_key_vault"


class AzureKeyVaultSecretsStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Azure Key Vault secrets store integration."""

    pass


AzureKeyVaultSecretsStoreClient = SecretsStore

class AzureKeyVaultSecretsStoreIntegration(SecretsStoreIntegrationContract):
    """
    Single public Azure Key Vault secrets store entrypoint.

    Legacy catalog factory (create_azure_key_vault_secrets_store) owns catalog behavior; legacy factories use from_client().
    """

    config: AzureKeyVaultSecretsStoreIntegrationConfig = AzureKeyVaultSecretsStoreIntegrationConfig()
    _client: AzureKeyVaultSecretsStoreClient | None = PrivateAttr(default=None)
    


    def get_secret(self, key: str) -> str | None:
        return self._require_client().get_secret(key)

    def set_secret(self, key: str, value: str) -> None:
        self._require_client().set_secret(key, value)

    def delete_secret(self, key: str) -> None:
        self._require_client().delete_secret(key)


    def put_secret(self, path, value):
        return self._require_client().put_secret(path, value)

    def _require_client(self) -> SecretsStore:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


    @classmethod
    def from_client(
        cls,
        client: AzureKeyVaultSecretsStoreClient,
        *,
        enabled: bool = False,
    ) -> AzureKeyVaultSecretsStoreIntegration:
        integration = cls.for_provider(
            provider_id=AZURE_KEY_VAULT_SECRETS_STORE_PROVIDER_ID,
            display_name="Azure Key Vault",
            config=AzureKeyVaultSecretsStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> AzureKeyVaultSecretsStoreClient | None:
        return self._client

SecretsStore.register(AzureKeyVaultSecretsStoreIntegration)
