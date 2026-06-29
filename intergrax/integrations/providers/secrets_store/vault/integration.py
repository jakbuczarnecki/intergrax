# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vault secrets store integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import HealthStatus, IntegrationConfigurationError
from intergrax.integrations.contracts.secrets_store import SecretsStore
from intergrax.runtime.integrations.categories.security import SecretsStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

VAULT_SECRETS_STORE_PROVIDER_ID = "vault"


class VaultSecretsStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Vault secrets store integration."""

    pass


@runtime_checkable
class VaultSecretsStoreClient(SecretsStore, Protocol):
    """Vault client with health probe."""

    def health(self) -> HealthStatus | bool: ...


class VaultSecretsStoreIntegration(SecretsStoreIntegrationContract):
    """
    Single public Vault secrets store entrypoint.

    Legacy catalog factory (create_vault_secrets_store) owns catalog behavior; legacy factories use from_client().
    """

    config: VaultSecretsStoreIntegrationConfig = VaultSecretsStoreIntegrationConfig()
    _client: VaultSecretsStoreClient | None = PrivateAttr(default=None)
    


    def get_secret(self, key: str) -> str | None:
        return self._require_client().get_secret(key)

    def set_secret(self, key: str, value: str) -> None:
        self._require_client().put_secret(key, value)

    def delete_secret(self, key: str) -> None:
        self._require_client().delete_secret(key)

    def health(self):
        return self._require_client().health()

    def put_secret(self, path, value):
        return self._require_client().put_secret(path, value)

    def _require_client(self) -> VaultSecretsStoreClient:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


    @classmethod
    def from_client(
        cls,
        client: VaultSecretsStoreClient,
        *,
        enabled: bool = False,
    ) -> VaultSecretsStoreIntegration:
        integration = cls.for_provider(
            provider_id=VAULT_SECRETS_STORE_PROVIDER_ID,
            display_name="Vault",
            config=VaultSecretsStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> VaultSecretsStoreClient | None:
        return self._client

SecretsStore.register(VaultSecretsStoreIntegration)
