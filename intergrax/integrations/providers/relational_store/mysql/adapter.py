# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""MySQL relational store adapter — ``RelationalStore`` facade (no driver I/O here)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.mysql.config import MySQLIntegrationConfig


class _MySQLRelationalStore:
    """
    Minimal ``RelationalStore`` over an existing MySQL connection.

    Connections are opened only in ``opens.open_mysql_relational_store()``.
    Tier-3 code MUST use ``create_mysql_relational_store()`` or ``profile.resolve()``.
    """

    def __init__(
        self,
        config: MySQLIntegrationConfig,
        connection: Any,
    ) -> None:
        self._config = config
        self._connection = connection

    @property
    def config(self) -> MySQLIntegrationConfig:
        return self._config

    def connect(self) -> None:
        if self._connection is None:
            raise IntegrationConfigurationError(
                "MySQL store is closed; create a new store via create_mysql_relational_store()"
            )

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        conn = self._require_connection()
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
        conn.commit()

    def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Mapping[str, Any]]:
        conn = self._require_connection()
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _require_connection(self) -> Any:
        if self._connection is None:
            raise IntegrationConfigurationError(
                "MySQL store is closed; create a new store via create_mysql_relational_store()"
            )
        return self._connection
