# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.database.contracts import (
    DatabaseExecuteInput,
    DatabaseExecuteOutput,
    DatabaseQueryInput,
    DatabaseQueryOutput,
)
from intergrax.tools.providers.database.handlers import DatabaseExecuteHandler, DatabaseQueryHandler
from intergrax.tools.providers.database.service import DATABASE_EXECUTE_TOOL_ID, DATABASE_QUERY_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

DATABASE_BUNDLE_ID = "database"
DATABASE_TOOL_IDS: tuple[str, ...] = (DATABASE_QUERY_TOOL_ID, DATABASE_EXECUTE_TOOL_ID)


def register_database_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=DATABASE_QUERY_TOOL_ID,
            name=DATABASE_QUERY_TOOL_ID,
            description="Run a parameterized read-only SQL query against the configured relational store.",
            description_short="SQL SELECT query.",
            input_schema=DatabaseQueryInput,
            output_schema=DatabaseQueryOutput,
            error_mapping={},
            side_effects=False,
            category="database",
            risk_level=ToolRiskLevel.LOW,
            tags=("database", "sql", "relational"),
        ),
        DatabaseQueryHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=DATABASE_EXECUTE_TOOL_ID,
            name=DATABASE_EXECUTE_TOOL_ID,
            description="Execute a parameterized mutating SQL statement (INSERT/UPDATE/DELETE/DDL).",
            description_short="SQL execute.",
            input_schema=DatabaseExecuteInput,
            output_schema=DatabaseExecuteOutput,
            error_mapping={},
            side_effects=True,
            category="database",
            risk_level=ToolRiskLevel.HIGH,
            tags=("database", "sql", "relational"),
        ),
        DatabaseExecuteHandler(ctx),
    )
