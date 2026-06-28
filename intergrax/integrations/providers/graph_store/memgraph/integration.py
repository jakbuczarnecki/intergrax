# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Memgraph graph store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.data import GraphStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

MEMGRAPH_GRAPH_STORE_PROVIDER_ID = "memgraph"


class MemgraphGraphStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Memgraph graph store integration."""

    pass


@runtime_checkable
class MemgraphGraphStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class MemgraphGraphStoreIntegration(GraphStoreIntegrationContract):
    """
    Memgraph graph store integration.

    The legacy facade (create_memgraph_graph_store) remains separate and backward-compatible.
    """

    config: MemgraphGraphStoreIntegrationConfig = MemgraphGraphStoreIntegrationConfig()
    _client: MemgraphGraphStoreClient | None = PrivateAttr(default=None)

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
