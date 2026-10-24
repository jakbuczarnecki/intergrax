# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""PostgreSQL relational store adapter — ``RelationalStore`` facade (no driver I/O here)."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.relational_store_bounded_query import (
    RelationalStoreBoundedQueryPort,
)
from intergrax.integrations.providers.relational_store.postgresql.config import PostgreSQLIntegrationConfig
from intergrax.integrations.relational_bounded_row_accumulation import (
    accumulate_bounded_sql_rows,
)


class _PostgreSQLRelationalStore(RelationalStoreBoundedQueryPort):
    """
    Minimal ``RelationalStore`` over an existing PostgreSQL connection.

    Connections are opened only in ``opens.open_postgresql_relational_store()``.
    Tier-3 code MUST use ``create_postgresql_relational_store()`` or ``profile.resolve()``.
    """

    def __init__(
        self,
        config: PostgreSQLIntegrationConfig,
        connection: Any,
    ) -> None:
        self._config = config
        self._connection = connection

    @property
    def config(self) -> PostgreSQLIntegrationConfig:
        return self._config

    def connect(self) -> None:
        """Connection is established by ``opens`` before this adapter is constructed."""
        if self._connection is None:
            raise IntegrationConfigurationError(
                "PostgreSQL store is closed; create a new store via create_postgresql_relational_store()"
            )

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        conn = self._require_connection()
        conn.execute(sql, params)
        conn.commit()

    def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Mapping[str, Any]]:
        conn = self._require_connection()
        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def fetch_query_bounded(
        self,
        sql: str,
        params: Sequence[Any] = (),
        *,
        max_rows: int,
        max_result_bytes: int,
    ) -> tuple[tuple[Mapping[str, Any], ...], bool, str]:
        conn = self._require_connection()
        cursor = conn.execute(sql, params)
        try:

            def _row_iter():
                for row in cursor:
                    yield dict(row)

            accumulated, truncated, reason = accumulate_bounded_sql_rows(
                _row_iter(),
                max_rows=max_rows,
                max_result_bytes=max_result_bytes,
            )
            return tuple(accumulated), truncated, reason
        finally:
            cursor.close()

    def cancel_backend(self) -> None:
        """Request PostgreSQL backend cancellation for the store connection (physical cancel)."""
        conn = self._require_connection()
        conn.cancel()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def release_connection_for_collaborative_work_materialization(
        self,
    ) -> tuple[PostgreSQLIntegrationConfig, Callable[[], Any]]:
        """Transfer sole connection ownership to Collaborative Work materialization."""
        connection = self._require_connection()
        self._connection = None
        config = self._config

        def _reuse_connection() -> Any:
            return connection

        return config, _reuse_connection

    def _require_connection(self) -> Any:
        if self._connection is None:
            raise IntegrationConfigurationError(
                "PostgreSQL store is closed; create a new store via create_postgresql_relational_store()"
            )
        return self._connection
