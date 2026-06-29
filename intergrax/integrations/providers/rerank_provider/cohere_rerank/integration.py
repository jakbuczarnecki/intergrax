# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cohere Rerank rerank provider integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.runtime.integrations.categories.search import RerankProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

COHERE_RERANK_RERANK_PROVIDER_PROVIDER_ID = "cohere_rerank"


class CohereRerankRerankProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Cohere Rerank rerank provider integration."""

    pass


@runtime_checkable
class CohereRerankRerankProviderClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class CohereRerankRerankProviderIntegration(RerankProviderIntegrationContract):
    """
    Single public Cohere Rerank rerank provider entrypoint.

    Legacy catalog factory (create_cohere_rerank_provider) delegates to this class.
    """

    config: CohereRerankRerankProviderIntegrationConfig = CohereRerankRerankProviderIntegrationConfig()
    _client: CohereRerankRerankProviderClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> CohereRerankRerankProviderIntegration:
        integration = cls.for_provider(
            provider_id=COHERE_RERANK_RERANK_PROVIDER_PROVIDER_ID,
            display_name="Cohere Rerank",
            config=CohereRerankRerankProviderIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Cohere Rerank integration requires a runtime delegate")
        return self._runtime



    @classmethod
    def from_client(
        cls,
        client: CohereRerankRerankProviderClient,
        *,
        enabled: bool = False,
    ) -> CohereRerankRerankProviderIntegration:
        integration = cls.for_provider(
            provider_id=COHERE_RERANK_RERANK_PROVIDER_PROVIDER_ID,
            display_name="Cohere Rerank",
            config=CohereRerankRerankProviderIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> CohereRerankRerankProviderClient | None:
        return self._client
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

