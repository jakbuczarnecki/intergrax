# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Doppler secrets store integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.secrets_store import SecretsStore
from intergrax.runtime.integrations.categories.security import SecretsStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

DOPPLER_SECRETS_STORE_PROVIDER_ID = "doppler"


class DopplerSecretsStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Doppler secrets store integration."""

    pass


DopplerSecretsStoreClient = SecretsStore

class DopplerSecretsStoreIntegration(SecretsStoreIntegrationContract):
    """
    Single public Doppler secrets store entrypoint.

    Legacy catalog factory (create_doppler_secrets_store) owns catalog behavior; legacy factories use from_client().
    """

    config: DopplerSecretsStoreIntegrationConfig = DopplerSecretsStoreIntegrationConfig()
    _client: DopplerSecretsStoreClient | None = PrivateAttr(default=None)
    


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
        client: DopplerSecretsStoreClient,
        *,
        enabled: bool = False,
    ) -> DopplerSecretsStoreIntegration:
        integration = cls.for_provider(
            provider_id=DOPPLER_SECRETS_STORE_PROVIDER_ID,
            display_name="Doppler",
            config=DopplerSecretsStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> DopplerSecretsStoreClient | None:
        return self._client

SecretsStore.register(DopplerSecretsStoreIntegration)
