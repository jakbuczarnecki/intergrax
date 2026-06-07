# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DatabaseQueryInput(BaseModel):
    sql: str = Field(..., min_length=1, description="Parameterized SELECT query.")
    params: list[Any] = Field(default_factory=list, description="Positional bind parameters.")


class DatabaseQueryOutput(BaseModel):
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0


class DatabaseExecuteInput(BaseModel):
    sql: str = Field(..., min_length=1, description="Parameterized INSERT/UPDATE/DELETE/DDL statement.")
    params: list[Any] = Field(default_factory=list, description="Positional bind parameters.")


class DatabaseExecuteOutput(BaseModel):
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
