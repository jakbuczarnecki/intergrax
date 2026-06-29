# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Memgraph graph store integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence, Mapping, Any

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.graph_store import GraphStore
from intergrax.integrations.contracts.graph_store import GraphQueryResult, GraphStore
from intergrax.runtime.integrations.categories.data import GraphStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

MEMGRAPH_GRAPH_STORE_PROVIDER_ID = "memgraph"


class MemgraphGraphStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Memgraph graph store integration."""

    pass


MemgraphGraphStoreClient = GraphStore

class MemgraphGraphStoreIntegration(GraphStoreIntegrationContract):
    """
    Single public Memgraph graph store entrypoint.

    Legacy catalog factory (create_memgraph_graph_store) owns catalog behavior; legacy factories use from_client().
    """

    config: MemgraphGraphStoreIntegrationConfig = MemgraphGraphStoreIntegrationConfig()
    _client: MemgraphGraphStoreClient | None = PrivateAttr(default=None)
    


    def query(self, query: str, *, params: Mapping[str, Any] | None = None) -> GraphQueryResult:
        return self._require_client().query(query, params=params)

    def close(self) -> None:
        self._require_client().close()


    def get_node(self, node_id):
        return self._require_client().get_node(node_id)

    def run_query(self, statement, parameters: Optional[Mapping[str, Any]] = None):
        return self._require_client().run_query(statement, parameters=parameters)

    def _require_client(self) -> GraphStore:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


    @classmethod
    def from_client(
        cls,
        client: MemgraphGraphStoreClient,
        *,
        enabled: bool = False,
    ) -> MemgraphGraphStoreIntegration:
        integration = cls.for_provider(
            provider_id=MEMGRAPH_GRAPH_STORE_PROVIDER_ID,
            display_name="Memgraph",
            config=MemgraphGraphStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> MemgraphGraphStoreClient | None:
        return self._client

GraphStore.register(MemgraphGraphStoreIntegration)
