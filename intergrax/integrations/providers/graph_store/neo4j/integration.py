# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neo4J graph store integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, Mapping, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.graph_store import GraphStore
from intergrax.integrations.contracts.graph_store import GraphQueryResult, GraphStore
from intergrax.runtime.integrations.categories.data import GraphStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

NEO4J_GRAPH_STORE_PROVIDER_ID = "neo4j"


class Neo4jGraphStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Neo4J graph store integration."""

    pass


@runtime_checkable
class Neo4jGraphStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class Neo4jGraphStoreIntegration(GraphStoreIntegrationContract):
    """
    Single public Neo4J graph store entrypoint.

    Legacy catalog factory (create_neo4j_graph_store) delegates to this class.
    """

    config: Neo4jGraphStoreIntegrationConfig = Neo4jGraphStoreIntegrationConfig()
    _client: Neo4jGraphStoreClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        *,
        enabled: bool = True,
    ) -> Neo4jGraphStoreIntegration:
        integration = cls.for_provider(
            provider_id=NEO4J_GRAPH_STORE_PROVIDER_ID,
            display_name="Neo4J",
            config=Neo4jGraphStoreIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration


    def query(self, query: str, *, params: Mapping[str, Any] | None = None) -> GraphQueryResult:
        return self._require_runtime().query(query, params=params)

    def close(self) -> None:
        self._require_runtime().close()


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
        client: Neo4jGraphStoreClient,
        *,
        enabled: bool = False,
    ) -> Neo4jGraphStoreIntegration:
        integration = cls.for_provider(
            provider_id=NEO4J_GRAPH_STORE_PROVIDER_ID,
            display_name="Neo4J",
            config=Neo4jGraphStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> Neo4jGraphStoreClient | None:
        return self._client
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

GraphStore.register(Neo4jGraphStoreIntegration)
