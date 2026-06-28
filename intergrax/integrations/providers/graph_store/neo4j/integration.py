# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neo4J graph store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

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
    Neo4J graph store integration.

    The legacy facade (create_neo4j_graph_store) remains separate and backward-compatible.
    """

    config: Neo4jGraphStoreIntegrationConfig = Neo4jGraphStoreIntegrationConfig()
    _client: Neo4jGraphStoreClient | None = PrivateAttr(default=None)

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
