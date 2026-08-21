# © Artur Czarnecki. All rights reserved.
# Intergrax platform proof — TOOLS-ITERATIVE-SQL-INVESTIGATION (PP-3B).

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.knowledge.contracts.validation import JsonPrimitive

JsonScalar = JsonPrimitive

PLATFORM_PROOF_SQL_QUERY_TOOL_ID = "platform_proof.sql.query"

MAX_VISIBLE_ROWS = 200
FETCH_LIMIT = MAX_VISIBLE_ROWS + 1


class SqlQueryInput(BaseModel):
    sql: str = Field(..., min_length=1, description="Single read-only SELECT or WITH query.")


class SqlRow(BaseModel):
    values: tuple[JsonScalar, ...]


class SqlQueryOutput(BaseModel):
    columns: tuple[str, ...]
    rows: tuple[SqlRow, ...]
    row_count: int
    truncated: bool
