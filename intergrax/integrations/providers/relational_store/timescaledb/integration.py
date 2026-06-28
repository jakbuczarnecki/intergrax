# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Timescaledb relational store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

TIMESCALEDB_RELATIONAL_STORE_PROVIDER_ID = "timescaledb"


class TimescaledbRelationalStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Timescaledb relational store integration."""

    pass


@runtime_checkable
class TimescaledbRelationalStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class TimescaledbRelationalStoreIntegration(RelationalStoreIntegrationContract):
    """
    Timescaledb relational store integration.

    The legacy facade (create_timescaledb_relational_store) remains separate and backward-compatible.
    """

    config: TimescaledbRelationalStoreIntegrationConfig = TimescaledbRelationalStoreIntegrationConfig()
    _client: TimescaledbRelationalStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: TimescaledbRelationalStoreClient,
        *,
        enabled: bool = False,
    ) -> TimescaledbRelationalStoreIntegration:
        integration = cls.for_provider(
            provider_id=TIMESCALEDB_RELATIONAL_STORE_PROVIDER_ID,
            display_name="Timescaledb",
            config=TimescaledbRelationalStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> TimescaledbRelationalStoreClient | None:
        return self._client
