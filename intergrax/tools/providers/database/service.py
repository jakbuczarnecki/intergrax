# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import re

from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.tools.providers.database.contracts import (
    DatabaseDescribeSchemaInput,
    DatabaseDescribeSchemaOutput,
    DatabaseColumnOutput,
    DatabaseExecuteInput,
    DatabaseExecuteOutput,
    DatabaseQueryInput,
    DatabaseQueryOutput,
    DatabaseTableOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

DATABASE_QUERY_TOOL_ID = "database.query"
DATABASE_EXECUTE_TOOL_ID = "database.execute"
DATABASE_DESCRIBE_SCHEMA_TOOL_ID = "database.describe_schema"

_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validated_table_name(name: str) -> str:
    cleaned = name.strip()
    if not _TABLE_NAME_RE.match(cleaned):
        raise RuntimeError("invalid_table_name")
    return cleaned


def _require_relational_store(ctx: ToolWiringContext) -> RelationalStore:
    store = ctx.relational_store
    if store is None:
        raise RuntimeError("relational_store_not_configured")
    return store


def database_query(ctx: ToolWiringContext, params: DatabaseQueryInput) -> DatabaseQueryOutput:
    rows = _require_relational_store(ctx).fetch_all(params.sql.strip(), tuple(params.params))
    normalized = [dict(row) for row in rows]
    return DatabaseQueryOutput(rows=normalized, row_count=len(normalized))


def database_execute(ctx: ToolWiringContext, params: DatabaseExecuteInput) -> DatabaseExecuteOutput:
    _require_relational_store(ctx).execute(params.sql.strip(), tuple(params.params))
    return DatabaseExecuteOutput(executed=True)


def database_describe_schema(
    ctx: ToolWiringContext,
    params: DatabaseDescribeSchemaInput,
) -> DatabaseDescribeSchemaOutput:
    store = _require_relational_store(ctx)
    table_filter = params.table.strip()
    try:
        if table_filter:
            safe_name = _validated_table_name(table_filter)
            column_rows = store.fetch_all(f'PRAGMA table_info("{safe_name}")')
            columns = [
                DatabaseColumnOutput(
                    name=str(row.get("name") or ""),
                    type=str(row.get("type") or ""),
                    not_null=bool(row.get("notnull")),
                    primary_key=bool(row.get("pk")),
                )
                for row in column_rows
            ]
            tables = [DatabaseTableOutput(name=safe_name, columns=columns)]
        else:
            table_rows = store.fetch_all(
                """
                SELECT name, type FROM sqlite_master
                WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                LIMIT ?
                """,
                (params.limit,),
            )
            tables = []
            for row in table_rows:
                name = _validated_table_name(str(row.get("name") or ""))
                table_type = str(row.get("type") or "table")
                column_rows = store.fetch_all(f'PRAGMA table_info("{name}")')
                columns = [
                    DatabaseColumnOutput(
                        name=str(col.get("name") or ""),
                        type=str(col.get("type") or ""),
                        not_null=bool(col.get("notnull")),
                        primary_key=bool(col.get("pk")),
                    )
                    for col in column_rows
                ]
                tables.append(DatabaseTableOutput(name=name, type=table_type, columns=columns))
    except RuntimeError as exc:
        return DatabaseDescribeSchemaOutput(used=False, reason=str(exc))
    except Exception as exc:
        return DatabaseDescribeSchemaOutput(
            used=False,
            reason=f"describe_schema_error:{exc.__class__.__name__}",
        )

    return DatabaseDescribeSchemaOutput(used=True, tables=tables, reason="ok")
