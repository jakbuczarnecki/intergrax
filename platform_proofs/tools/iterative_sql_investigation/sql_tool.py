# © Artur Czarnecki. All rights reserved.
# Intergrax platform proof — TOOLS-ITERATIVE-SQL-INVESTIGATION (PP-3B).

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.knowledge.contracts.validation import JsonPrimitive
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.tool_executor import ToolHandler

from platform_proofs.tools.iterative_sql_investigation.contracts import (
    FETCH_LIMIT,
    MAX_VISIBLE_ROWS,
    SqlQueryInput,
    SqlQueryOutput,
    SqlRow,
)

_FORBIDDEN_KEYWORDS = (
    "INSERT",
    "UPDATE",
    "DELETE",
    "MERGE",
    "CREATE",
    "ALTER",
    "DROP",
    "TRUNCATE",
    "COPY",
    "CALL",
    "GRANT",
    "REVOKE",
    "BEGIN",
    "COMMIT",
    "ROLLBACK",
    "SAVEPOINT",
    "VACUUM",
    "ANALYZE",
    "REINDEX",
    "LISTEN",
    "NOTIFY",
    "PREPARE",
    "EXECUTE",
    "DO",
)
_FORBIDDEN_PATTERN = re.compile(
    r"\b(" + "|".join(_FORBIDDEN_KEYWORDS) + r")\b",
    re.IGNORECASE,
)
_ALLOWED_START = re.compile(r"^(SELECT|WITH)\b", re.IGNORECASE | re.DOTALL)


class SqlValidationError(ValueError):
    """Proof-local read-only SQL validation failure."""


def normalize_sql(sql: str) -> str:
    cleaned = sql.strip()
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].rstrip()
    return cleaned


def validate_read_only_sql(sql: str) -> str:
    cleaned = normalize_sql(sql)
    if not cleaned:
        raise SqlValidationError("SQL must not be empty.")
    if ";" in cleaned:
        raise SqlValidationError("Multiple statements are not allowed.")
    if not _ALLOWED_START.match(cleaned):
        raise SqlValidationError("Only SELECT or WITH queries are allowed.")
    if _FORBIDDEN_PATTERN.search(cleaned):
        raise SqlValidationError("Mutating or administrative SQL keywords are not allowed.")
    return cleaned


def wrap_bounded_query(sql: str) -> str:
    cleaned = validate_read_only_sql(sql)
    return f"SELECT * FROM ({cleaned}) AS _proof_bounded LIMIT {FETCH_LIMIT}"


def _to_json_scalar(value: Any) -> JsonPrimitive:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            return str(value)
        return value
    return str(value)


def _rows_to_output(rows: Sequence[Mapping[str, Any]]) -> SqlQueryOutput:
    if not rows:
        return SqlQueryOutput(columns=(), rows=(), row_count=0, truncated=False)
    columns = tuple(rows[0].keys())
    truncated = len(rows) > MAX_VISIBLE_ROWS
    visible = rows[:MAX_VISIBLE_ROWS]
    output_rows = tuple(
        SqlRow(values=tuple(_to_json_scalar(row.get(column)) for column in columns))
        for row in visible
    )
    return SqlQueryOutput(
        columns=columns,
        rows=output_rows,
        row_count=len(output_rows),
        truncated=truncated,
    )


def execute_bounded_query(store: RelationalStore, sql: str) -> SqlQueryOutput:
    bounded_sql = wrap_bounded_query(sql)
    rows = store.fetch_all(bounded_sql)
    return _rows_to_output(rows)


class ProofSqlQueryHandler(ToolHandler[SqlQueryInput, SqlQueryOutput]):
    def __init__(self, store: RelationalStore) -> None:
        self._store = store

    def execute(self, request: ToolExecutionRequest[SqlQueryInput]) -> SqlQueryOutput:
        return execute_bounded_query(self._store, request.input.sql)
