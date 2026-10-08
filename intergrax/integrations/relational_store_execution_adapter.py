# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed adapter over legacy RelationalStore transport (TRACE-X-P5-R2-P3)."""

from __future__ import annotations

from intergrax.integrations.contracts.configured_relational_store_execution import (
    ConfiguredRelationalStoreExecutionPort,
    RelationalExecuteRequest,
    RelationalExecuteResult,
    RelationalQueryRequest,
    RelationalQueryResult,
)
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.integrations.contracts.sql_scalar import (
    sql_scalar_row_from_mapping,
    sql_scalar_tuple_from_sequence,
)


class RelationalStoreExecutionAdapter(ConfiguredRelationalStoreExecutionPort):
    """Contain legacy ``Any`` typing inside the adapter boundary."""

    __slots__ = ("_store",)

    def __init__(self, store: RelationalStore) -> None:
        self._store = store

    def query(self, request: RelationalQueryRequest) -> RelationalQueryResult:
        raw_rows = self._store.fetch_all(
            request.sql.strip(),
            sql_scalar_tuple_from_sequence(request.params),
        )
        rows = tuple(sql_scalar_row_from_mapping(row) for row in raw_rows)
        return RelationalQueryResult(rows=rows, row_count=len(rows))

    def execute(self, request: RelationalExecuteRequest) -> RelationalExecuteResult:
        self._store.execute(
            request.sql.strip(),
            sql_scalar_tuple_from_sequence(request.params),
        )
        return RelationalExecuteResult(executed=True)


__all__ = ["RelationalStoreExecutionAdapter"]
