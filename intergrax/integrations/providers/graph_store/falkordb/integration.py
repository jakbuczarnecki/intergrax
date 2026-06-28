# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Falkordb graph store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.data import GraphStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

FALKORDB_GRAPH_STORE_PROVIDER_ID = "falkordb"


class FalkordbGraphStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Falkordb graph store integration."""

    pass


@runtime_checkable
class FalkordbGraphStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class FalkordbGraphStoreIntegration(GraphStoreIntegrationContract):
    """
    Falkordb graph store integration.

    The legacy facade (create_falkordb_graph_store) remains separate and backward-compatible.
    """

    config: FalkordbGraphStoreIntegrationConfig = FalkordbGraphStoreIntegrationConfig()
    _client: FalkordbGraphStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: FalkordbGraphStoreClient,
        *,
        enabled: bool = False,
    ) -> FalkordbGraphStoreIntegration:
        integration = cls.for_provider(
            provider_id=FALKORDB_GRAPH_STORE_PROVIDER_ID,
            display_name="Falkordb",
            config=FalkordbGraphStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> FalkordbGraphStoreClient | None:
        return self._client
