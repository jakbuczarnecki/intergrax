# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Gcp Secret Manager secrets store integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.secrets_store import SecretsStore
from intergrax.runtime.integrations.categories.security import SecretsStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

GCP_SECRET_MANAGER_SECRETS_STORE_PROVIDER_ID = "gcp_secret_manager"


class GcpSecretManagerSecretsStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Gcp Secret Manager secrets store integration."""

    pass


@runtime_checkable
class GcpSecretManagerSecretsStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class GcpSecretManagerSecretsStoreIntegration(SecretsStoreIntegrationContract):
    """
    Single public Gcp Secret Manager secrets store entrypoint.

    Legacy catalog factory (create_gcp_secret_manager_secrets_store) delegates to this class.
    """

    config: GcpSecretManagerSecretsStoreIntegrationConfig = GcpSecretManagerSecretsStoreIntegrationConfig()
    _client: GcpSecretManagerSecretsStoreClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        *,
        enabled: bool = True,
    ) -> GcpSecretManagerSecretsStoreIntegration:
        integration = cls.for_provider(
            provider_id=GCP_SECRET_MANAGER_SECRETS_STORE_PROVIDER_ID,
            display_name="Gcp Secret Manager",
            config=GcpSecretManagerSecretsStoreIntegrationConfig(enabled=enabled),
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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

SecretsStore.register(GcpSecretManagerSecretsStoreIntegration)
