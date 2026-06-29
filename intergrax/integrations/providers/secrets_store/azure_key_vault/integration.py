# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Azure Key Vault secrets store integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.secrets_store import SecretsStore
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
    Single public Azure Key Vault secrets store entrypoint.

    Legacy catalog factory (create_azure_key_vault_secrets_store) delegates to this class.
    """

    config: AzureKeyVaultSecretsStoreIntegrationConfig = AzureKeyVaultSecretsStoreIntegrationConfig()
    _client: AzureKeyVaultSecretsStoreClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        *,
        enabled: bool = True,
    ) -> AzureKeyVaultSecretsStoreIntegration:
        integration = cls.for_provider(
            provider_id=AZURE_KEY_VAULT_SECRETS_STORE_PROVIDER_ID,
            display_name="Azure Key Vault",
            config=AzureKeyVaultSecretsStoreIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration


    def get_secret(self, key: str) -> str | None:
        return self._require_runtime().get_secret(key)

    def set_secret(self, key: str, value: str) -> None:
        self._require_runtime().set_secret(key, value)

    def delete_secret(self, key: str) -> None:
        self._require_runtime().delete_secret(key)


    def _require_runtime(self) -> Any:
        private = object.__getattribute__(self, "__pydantic_private__")
        runtime = private.get("_runtime")
        if runtime is None:
            runtime = private.get("_backend")
        if runtime is None:
            runtime = private.get("_inner")
        if runtime is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a runtime delegate for catalog operations",
            )
        return runtime


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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

SecretsStore.register(AzureKeyVaultSecretsStoreIntegration)
