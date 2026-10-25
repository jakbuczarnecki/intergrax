# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from intergrax.integrations.contracts.sql_scalar import SqlScalar
from intergrax.tools.providers.provider_resource_enforcement import (
    DEFAULT_DATABASE_MAX_RESULT_BYTES,
    DEFAULT_DATABASE_MAX_ROWS,
)


class DatabaseQueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sql: str = Field(..., min_length=1, description="Parameterized SELECT query.")
    params: list[SqlScalar] = Field(
        default_factory=list,
        description="Positional bind parameters.",
    )
    max_rows: int = Field(
        default=DEFAULT_DATABASE_MAX_ROWS,
        ge=1,
        le=10_000,
        description="Maximum rows returned (truncation when exceeded).",
    )
    max_result_bytes: int = Field(
        default=DEFAULT_DATABASE_MAX_RESULT_BYTES,
        ge=1,
        le=16_777_216,
        description="Maximum UTF-8 JSON size of the rows payload.",
    )


class DatabaseQueryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[dict[str, SqlScalar]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    limit_reason: str = Field(
        default="",
        description="row_limit | byte_limit when truncated; empty otherwise.",
    )


class DatabaseExecuteInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sql: str = Field(..., min_length=1, description="Parameterized INSERT/UPDATE/DELETE/DDL statement.")
    params: list[SqlScalar] = Field(
        default_factory=list,
        description="Positional bind parameters.",
    )


class DatabaseExecuteOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    executed: bool = True


class DatabaseDescribeSchemaInput(BaseModel):
    table: str = Field(default="", description="Optional table name filter.")
    limit: int = Field(default=100, ge=1, le=500)


class DatabaseColumnOutput(BaseModel):
    name: str
    type: str = ""
    not_null: bool = False
    primary_key: bool = False


class DatabaseTableOutput(BaseModel):
    name: str
    type: str = "table"
    columns: list[DatabaseColumnOutput] = Field(default_factory=list)


class DatabaseDescribeSchemaOutput(BaseModel):
    used: bool = False
    tables: list[DatabaseTableOutput] = Field(default_factory=list)
    reason: str = ""
