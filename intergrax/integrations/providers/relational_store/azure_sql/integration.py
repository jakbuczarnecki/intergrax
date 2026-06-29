# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Azure Sql relational store integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, Mapping, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

AZURE_SQL_RELATIONAL_STORE_PROVIDER_ID = "azure_sql"


class AzureSqlRelationalStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Azure Sql relational store integration."""

    pass


@runtime_checkable
class AzureSqlRelationalStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class AzureSqlRelationalStoreIntegration(RelationalStoreIntegrationContract):
    """
    Single public Azure Sql relational store entrypoint.

    Legacy catalog factory (create_azure_sql_relational_store) delegates to this class.
    """

    config: AzureSqlRelationalStoreIntegrationConfig = AzureSqlRelationalStoreIntegrationConfig()
    _client: AzureSqlRelationalStoreClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        *,
        enabled: bool = True,
    ) -> AzureSqlRelationalStoreIntegration:
        integration = cls.for_provider(
            provider_id=AZURE_SQL_RELATIONAL_STORE_PROVIDER_ID,
            display_name="Azure Sql",
            config=AzureSqlRelationalStoreIntegrationConfig(enabled=enabled),
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
        client: AzureSqlRelationalStoreClient,
        *,
        enabled: bool = False,
    ) -> AzureSqlRelationalStoreIntegration:
        integration = cls.for_provider(
            provider_id=AZURE_SQL_RELATIONAL_STORE_PROVIDER_ID,
            display_name="Azure Sql",
            config=AzureSqlRelationalStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> AzureSqlRelationalStoreClient | None:
        return self._client

RelationalStore.register(AzureSqlRelationalStoreIntegration)
