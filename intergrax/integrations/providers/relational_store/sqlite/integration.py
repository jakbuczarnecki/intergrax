# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sqlite relational store integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

SQLITE_RELATIONAL_STORE_PROVIDER_ID = "sqlite"


class SqliteRelationalStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Sqlite relational store integration."""

    pass


@runtime_checkable
class SqliteRelationalStoreClient(RelationalStore, Protocol):
    """SQLite relational store client with filesystem path."""

    @property
    def db_path(self) -> str: ...


class SqliteRelationalStoreIntegration(RelationalStoreIntegrationContract):
    """
    Single public Sqlite relational store entrypoint.

    Legacy catalog factory (create_sqlite_integration) owns catalog behavior; legacy factories use from_client().
    """

    config: SqliteRelationalStoreIntegrationConfig = SqliteRelationalStoreIntegrationConfig()
    _client: SqliteRelationalStoreClient | None = PrivateAttr(default=None)
    


    def connect(self) -> None:
        self._require_client().connect()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        self._require_client().execute(sql, params)

    def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Mapping[str, Any]]:
        return self._require_client().fetch_all(sql, params)

    def close(self) -> None:
        self._require_client().close()

    @property
    def db_path(self):
        return self._require_client().db_path

    def _require_client(self) -> SqliteRelationalStoreClient:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


    @classmethod
    def from_client(
        cls,
        client: SqliteRelationalStoreClient,
        *,
        enabled: bool = False,
    ) -> SqliteRelationalStoreIntegration:
        integration = cls.for_provider(
            provider_id=SQLITE_RELATIONAL_STORE_PROVIDER_ID,
            display_name="Sqlite",
            config=SqliteRelationalStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> SqliteRelationalStoreClient | None:
        return self._client

RelationalStore.register(SqliteRelationalStoreIntegration)
