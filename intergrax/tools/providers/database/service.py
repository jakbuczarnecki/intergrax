# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.tools.providers.database.contracts import (
    DatabaseExecuteInput,
    DatabaseExecuteOutput,
    DatabaseQueryInput,
    DatabaseQueryOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

DATABASE_QUERY_TOOL_ID = "database.query"
DATABASE_EXECUTE_TOOL_ID = "database.execute"


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
