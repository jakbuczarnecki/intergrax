# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Snowflake relational store integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, Mapping, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

SNOWFLAKE_RELATIONAL_STORE_PROVIDER_ID = "snowflake"


class SnowflakeRelationalStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Snowflake relational store integration."""

    pass


@runtime_checkable
class SnowflakeRelationalStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class SnowflakeRelationalStoreIntegration(RelationalStoreIntegrationContract):
    """
    Single public Snowflake relational store entrypoint.

    Legacy catalog factory (create_snowflake_relational_store) delegates to this class.
    """

    config: SnowflakeRelationalStoreIntegrationConfig = SnowflakeRelationalStoreIntegrationConfig()
    _client: SnowflakeRelationalStoreClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        *,
        enabled: bool = True,
    ) -> SnowflakeRelationalStoreIntegration:
        integration = cls.for_provider(
            provider_id=SNOWFLAKE_RELATIONAL_STORE_PROVIDER_ID,
            display_name="Snowflake",
            config=SnowflakeRelationalStoreIntegrationConfig(enabled=enabled),
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
        client: SnowflakeRelationalStoreClient,
        *,
        enabled: bool = False,
    ) -> SnowflakeRelationalStoreIntegration:
        integration = cls.for_provider(
            provider_id=SNOWFLAKE_RELATIONAL_STORE_PROVIDER_ID,
            display_name="Snowflake",
            config=SnowflakeRelationalStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> SnowflakeRelationalStoreClient | None:
        return self._client

RelationalStore.register(SnowflakeRelationalStoreIntegration)
