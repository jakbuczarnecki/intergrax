# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Mssql relational store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

MSSQL_RELATIONAL_STORE_PROVIDER_ID = "mssql"


class MssqlRelationalStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Mssql relational store integration."""

    pass


@runtime_checkable
class MssqlRelationalStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class MssqlRelationalStoreIntegration(RelationalStoreIntegrationContract):
    """
    Mssql relational store integration.

    The legacy facade (create_mssql_relational_store) remains separate and backward-compatible.
    """

    config: MssqlRelationalStoreIntegrationConfig = MssqlRelationalStoreIntegrationConfig()
    _client: MssqlRelationalStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: MssqlRelationalStoreClient,
        *,
        enabled: bool = False,
    ) -> MssqlRelationalStoreIntegration:
        integration = cls.for_provider(
            provider_id=MSSQL_RELATIONAL_STORE_PROVIDER_ID,
            display_name="Mssql",
            config=MssqlRelationalStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> MssqlRelationalStoreClient | None:
        return self._client
