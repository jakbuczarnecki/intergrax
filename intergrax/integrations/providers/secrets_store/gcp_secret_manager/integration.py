# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Gcp Secret Manager secrets store integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.secrets_store import SecretsStore
from intergrax.runtime.integrations.categories.security import SecretsStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

GCP_SECRET_MANAGER_SECRETS_STORE_PROVIDER_ID = "gcp_secret_manager"


class GcpSecretManagerSecretsStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Gcp Secret Manager secrets store integration."""

    pass


GcpSecretManagerSecretsStoreClient = SecretsStore

class GcpSecretManagerSecretsStoreIntegration(SecretsStoreIntegrationContract):
    """
    Single public Gcp Secret Manager secrets store entrypoint.

    Legacy catalog factory (create_gcp_secret_manager_secrets_store) owns catalog behavior; legacy factories use from_client().
    """

    config: GcpSecretManagerSecretsStoreIntegrationConfig = GcpSecretManagerSecretsStoreIntegrationConfig()
    _client: GcpSecretManagerSecretsStoreClient | None = PrivateAttr(default=None)
    


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
        client: GcpSecretManagerSecretsStoreClient,
        *,
        enabled: bool = False,
    ) -> GcpSecretManagerSecretsStoreIntegration:
        integration = cls.for_provider(
            provider_id=GCP_SECRET_MANAGER_SECRETS_STORE_PROVIDER_ID,
            display_name="Gcp Secret Manager",
            config=GcpSecretManagerSecretsStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> GcpSecretManagerSecretsStoreClient | None:
        return self._client

SecretsStore.register(GcpSecretManagerSecretsStoreIntegration)
