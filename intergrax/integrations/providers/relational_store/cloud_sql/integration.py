# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cloud Sql relational store integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, Mapping, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

CLOUD_SQL_RELATIONAL_STORE_PROVIDER_ID = "cloud_sql"


class CloudSqlRelationalStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Cloud Sql relational store integration."""

    pass


@runtime_checkable
class CloudSqlRelationalStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class CloudSqlRelationalStoreIntegration(RelationalStoreIntegrationContract):
    """
    Single public Cloud Sql relational store entrypoint.

    Legacy catalog factory (create_cloud_sql_relational_store) delegates to this class.
    """

    config: CloudSqlRelationalStoreIntegrationConfig = CloudSqlRelationalStoreIntegrationConfig()
    _client: CloudSqlRelationalStoreClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        *,
        enabled: bool = True,
    ) -> CloudSqlRelationalStoreIntegration:
        integration = cls.for_provider(
            provider_id=CLOUD_SQL_RELATIONAL_STORE_PROVIDER_ID,
            display_name="Cloud Sql",
            config=CloudSqlRelationalStoreIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration


    def connect(self) -> None:
        self._require_runtime().connect()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        self._require_runtime().execute(sql, params)

    def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Mapping[str, Any]]:
        return self._require_runtime().fetch_all(sql, params)

    def close(self) -> None:
        self._require_runtime().close()


    def _require_runtime(self) -> Any:
        private = object.__getattribute__(self, "__pydantic_private__")
        runtime = private.get("_runtime")
        if runtime is None:
            runtime = private.get("_backend")
        if runtime is None:
            runtime = private.get("_inner")
        if runtime is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a runtime delegate for catalog operations",
            )
        return runtime


    @classmethod
    def from_client(
        cls,
        client: CloudSqlRelationalStoreClient,
        *,
        enabled: bool = False,
    ) -> CloudSqlRelationalStoreIntegration:
        integration = cls.for_provider(
            provider_id=CLOUD_SQL_RELATIONAL_STORE_PROVIDER_ID,
            display_name="Cloud Sql",
            config=CloudSqlRelationalStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> CloudSqlRelationalStoreClient | None:
        return self._client

RelationalStore.register(CloudSqlRelationalStoreIntegration)
