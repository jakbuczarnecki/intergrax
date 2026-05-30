# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Databricks relational store adapter — ``RelationalStore`` facade (no driver I/O here)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.databricks.config import DatabricksIntegrationConfig


class DatabricksRelationalStore:
    """
    Minimal ``RelationalStore`` over an existing Databricks SQL connection.

    Connections are opened only in ``opens.open_databricks_relational_store()``.
    Tier-3 code MUST use ``create_databricks_relational_store()`` or ``profile.resolve()``.
    """

    def __init__(
        self,
        config: DatabricksIntegrationConfig,
        connection: Any,
    ) -> None:
        self._config = config
        self._connection = connection

    @property
    def config(self) -> DatabricksIntegrationConfig:
        return self._config

    def connect(self) -> None:
        if self._connection is None:
            raise IntegrationConfigurationError(
                "Databricks store is closed; create a new store via create_databricks_relational_store()"
            )

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        conn = self._require_connection()
        with conn.cursor() as cursor:
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)

    def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Mapping[str, Any]]:
        conn = self._require_connection()
        with conn.cursor() as cursor:
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [col[0] for col in (cursor.description or ())]
        return [_row_to_mapping(columns, row) for row in rows]

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _require_connection(self) -> Any:
        if self._connection is None:
            raise IntegrationConfigurationError(
                "Databricks store is closed; create a new store via create_databricks_relational_store()"
            )
        return self._connection


def _row_to_mapping(columns: list[str], row: Any) -> Mapping[str, Any]:
    if hasattr(row, "asDict"):
        return dict(row.asDict())
    if columns:
        return dict(zip(columns, row, strict=False))
    return dict(enumerate(row))
