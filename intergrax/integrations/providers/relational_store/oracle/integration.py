# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Oracle relational store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

ORACLE_RELATIONAL_STORE_PROVIDER_ID = "oracle"


class OracleRelationalStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Oracle relational store integration."""

    pass


@runtime_checkable
class OracleRelationalStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class OracleRelationalStoreIntegration(RelationalStoreIntegrationContract):
    """
    Oracle relational store integration.

    The legacy facade (create_oracle_relational_store) remains separate and backward-compatible.
    """

    config: OracleRelationalStoreIntegrationConfig = OracleRelationalStoreIntegrationConfig()
    _client: OracleRelationalStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: OracleRelationalStoreClient,
        *,
        enabled: bool = False,
    ) -> OracleRelationalStoreIntegration:
        integration = cls.for_provider(
            provider_id=ORACLE_RELATIONAL_STORE_PROVIDER_ID,
            display_name="Oracle",
            config=OracleRelationalStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> OracleRelationalStoreClient | None:
        return self._client
