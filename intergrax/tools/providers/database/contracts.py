# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from intergrax.integrations.contracts.sql_scalar import SqlScalar


class DatabaseQueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sql: str = Field(..., min_length=1, description="Parameterized SELECT query.")
    params: list[SqlScalar] = Field(
        default_factory=list,
        description="Positional bind parameters.",
    )


class DatabaseQueryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[dict[str, SqlScalar]] = Field(default_factory=list)
    row_count: int = 0


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
