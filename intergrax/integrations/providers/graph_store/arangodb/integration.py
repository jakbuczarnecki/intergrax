# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ArangoDB graph store integration (H-INT-GRAPH-3)."""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.graph_store import GraphQueryResult, GraphStore
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig
from intergrax.runtime.integrations.categories.data import GraphStoreIntegrationContract

ARANGODB_GRAPH_STORE_PROVIDER_ID = "arangodb"


class ArangoDbGraphStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for ArangoDB graph store integration."""

    pass


ArangoDbGraphStoreClient = GraphStore


class ArangoDbGraphStoreIntegration(GraphStoreIntegrationContract):
    """Thin typed boundary over the existing ArangoDB GraphStore runtime."""

    config: ArangoDbGraphStoreIntegrationConfig = ArangoDbGraphStoreIntegrationConfig()
    _client: ArangoDbGraphStoreClient | None = PrivateAttr(default=None)

    def query(self, query: str, *, params: Mapping[str, Any] | None = None) -> GraphQueryResult:
        return self._require_client().query(query, params=params)

    def close(self) -> None:
        self._require_client().close()

    def get_node(self, node_id):
        return self._require_client().get_node(node_id)

    def run_query(self, statement, parameters: Mapping[str, Any] | None = None):
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
        client: ArangoDbGraphStoreClient,
        *,
        enabled: bool = False,
    ) -> ArangoDbGraphStoreIntegration:
        integration = cls.for_provider(
            provider_id=ARANGODB_GRAPH_STORE_PROVIDER_ID,
            display_name="ArangoDB",
            config=ArangoDbGraphStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> ArangoDbGraphStoreClient | None:
        return self._client


GraphStore.register(ArangoDbGraphStoreIntegration)
