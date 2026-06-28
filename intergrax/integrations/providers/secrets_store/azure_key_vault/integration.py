# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Azure Key Vault secrets store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.security import SecretsStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

AZURE_KEY_VAULT_SECRETS_STORE_PROVIDER_ID = "azure_key_vault"


class AzureKeyVaultSecretsStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Azure Key Vault secrets store integration."""

    pass


@runtime_checkable
class AzureKeyVaultSecretsStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class AzureKeyVaultSecretsStoreIntegration(SecretsStoreIntegrationContract):
    """
    Azure Key Vault secrets store integration.

    The legacy facade (create_azure_key_vault_secrets_store) remains separate and backward-compatible.
    """

    config: AzureKeyVaultSecretsStoreIntegrationConfig = AzureKeyVaultSecretsStoreIntegrationConfig()
    _client: AzureKeyVaultSecretsStoreClient | None = PrivateAttr(default=None)

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
