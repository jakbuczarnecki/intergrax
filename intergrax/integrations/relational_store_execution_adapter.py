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
from intergrax.integrations.contracts.relational_store_bounded_query import (
    RelationalStoreBoundedQueryPort,
)
from intergrax.integrations.contracts.sql_scalar import (
    sql_scalar_row_from_mapping,
    sql_scalar_tuple_from_sequence,
)
from intergrax.integrations.relational_bounded_row_accumulation import (
    RelationalResultByteLimitExceeded,
    accumulate_bounded_sql_rows,
)


class RelationalStoreExecutionAdapter(ConfiguredRelationalStoreExecutionPort):
    """Contain legacy ``Any`` typing inside the adapter boundary."""

    __slots__ = ("_store",)

    def __init__(self, store: RelationalStore) -> None:
        self._store = store

    def query(self, request: RelationalQueryRequest) -> RelationalQueryResult:
        sql = request.sql.strip()
        params = sql_scalar_tuple_from_sequence(request.params)
        bounds_requested = request.max_rows is not None and request.max_result_bytes is not None
        if bounds_requested:
            max_rows = request.max_rows or 0
            max_result_bytes = request.max_result_bytes or 0
            if isinstance(self._store, RelationalStoreBoundedQueryPort):
                raw_rows, truncated, limit_reason = self._store.fetch_query_bounded(
                    sql,
                    params,
                    max_rows=max_rows,
                    max_result_bytes=max_result_bytes,
                )
                rows = tuple(sql_scalar_row_from_mapping(row) for row in raw_rows)
                return RelationalQueryResult(
                    rows=rows,
                    row_count=len(rows),
                    truncated=truncated,
                    limit_reason=limit_reason,
                    physical_bounds_enforced=True,
                )
            raw_rows = self._store.fetch_all(sql, params)

            def _iter_rows():
                for row in raw_rows:
                    yield row

            try:
                bounded, truncated, limit_reason = accumulate_bounded_sql_rows(
                    _iter_rows(),
                    max_rows=max_rows,
                    max_result_bytes=max_result_bytes,
                )
            except RelationalResultByteLimitExceeded as exc:
                raise RuntimeError(exc.code) from exc
            rows = tuple(sql_scalar_row_from_mapping(row) for row in bounded)
            return RelationalQueryResult(
                rows=rows,
                row_count=len(rows),
                truncated=truncated,
                limit_reason=limit_reason,
                physical_bounds_enforced=False,
            )

        raw_rows = self._store.fetch_all(sql, params)
        rows = tuple(sql_scalar_row_from_mapping(row) for row in raw_rows)
        return RelationalQueryResult(rows=rows, row_count=len(rows))

    def execute(self, request: RelationalExecuteRequest) -> RelationalExecuteResult:
        self._store.execute(
            request.sql.strip(),
            sql_scalar_tuple_from_sequence(request.params),
        )
        return RelationalExecuteResult(executed=True)


__all__ = ["RelationalStoreExecutionAdapter"]
