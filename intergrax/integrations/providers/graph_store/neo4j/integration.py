# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neo4J graph store integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence, Mapping, Any

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


Neo4jGraphStoreClient = GraphStore

class Neo4jGraphStoreIntegration(GraphStoreIntegrationContract):
    """
    Single public Neo4J graph store entrypoint.

    Legacy catalog factory (create_neo4j_graph_store) owns catalog behavior; legacy factories use from_client().
    """

    config: Neo4jGraphStoreIntegrationConfig = Neo4jGraphStoreIntegrationConfig()
    _client: Neo4jGraphStoreClient | None = PrivateAttr(default=None)
    


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

GraphStore.register(Neo4jGraphStoreIntegration)
