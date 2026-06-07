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
