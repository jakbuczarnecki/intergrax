# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Infisical secrets store integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.secrets_store import SecretsStore
from intergrax.runtime.integrations.categories.security import SecretsStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

INFISICAL_SECRETS_STORE_PROVIDER_ID = "infisical"


class InfisicalSecretsStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Infisical secrets store integration."""

    pass


InfisicalSecretsStoreClient = SecretsStore

class InfisicalSecretsStoreIntegration(SecretsStoreIntegrationContract):
    """
    Single public Infisical secrets store entrypoint.

    Legacy catalog factory (create_infisical_secrets_store) owns catalog behavior; legacy factories use from_client().
    """

    config: InfisicalSecretsStoreIntegrationConfig = InfisicalSecretsStoreIntegrationConfig()
    _client: InfisicalSecretsStoreClient | None = PrivateAttr(default=None)
    


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
        client: InfisicalSecretsStoreClient,
        *,
        enabled: bool = False,
    ) -> InfisicalSecretsStoreIntegration:
        integration = cls.for_provider(
            provider_id=INFISICAL_SECRETS_STORE_PROVIDER_ID,
            display_name="Infisical",
            config=InfisicalSecretsStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> InfisicalSecretsStoreClient | None:
        return self._client

SecretsStore.register(InfisicalSecretsStoreIntegration)
