# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Doppler secrets store integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.secrets_store import SecretsStore
from intergrax.runtime.integrations.categories.security import SecretsStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

DOPPLER_SECRETS_STORE_PROVIDER_ID = "doppler"


class DopplerSecretsStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Doppler secrets store integration."""

    pass


@runtime_checkable
class DopplerSecretsStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class DopplerSecretsStoreIntegration(SecretsStoreIntegrationContract):
    """
    Single public Doppler secrets store entrypoint.

    Legacy catalog factory (create_doppler_secrets_store) delegates to this class.
    """

    config: DopplerSecretsStoreIntegrationConfig = DopplerSecretsStoreIntegrationConfig()
    _client: DopplerSecretsStoreClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        *,
        enabled: bool = True,
    ) -> DopplerSecretsStoreIntegration:
        integration = cls.for_provider(
            provider_id=DOPPLER_SECRETS_STORE_PROVIDER_ID,
            display_name="Doppler",
            config=DopplerSecretsStoreIntegrationConfig(enabled=enabled),
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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

SecretsStore.register(DopplerSecretsStoreIntegration)
