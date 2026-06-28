# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Duckdb relational store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

DUCKDB_RELATIONAL_STORE_PROVIDER_ID = "duckdb"


class DuckdbRelationalStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Duckdb relational store integration."""

    pass


@runtime_checkable
class DuckdbRelationalStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class DuckdbRelationalStoreIntegration(RelationalStoreIntegrationContract):
    """
    Duckdb relational store integration.

    The legacy facade (create_duckdb_relational_store) remains separate and backward-compatible.
    """

    config: DuckdbRelationalStoreIntegrationConfig = DuckdbRelationalStoreIntegrationConfig()
    _client: DuckdbRelationalStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: DuckdbRelationalStoreClient,
        *,
        enabled: bool = False,
    ) -> DuckdbRelationalStoreIntegration:
        integration = cls.for_provider(
            provider_id=DUCKDB_RELATIONAL_STORE_PROVIDER_ID,
            display_name="Duckdb",
            config=DuckdbRelationalStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> DuckdbRelationalStoreClient | None:
        return self._client
