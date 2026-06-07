# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GraphRunQueryInput(BaseModel):
    statement: str = Field(..., min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)


class GraphRunQueryOutput(BaseModel):
    records: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""
    record_count: int = 0


class GraphGetNodeInput(BaseModel):
    node_id: str = Field(..., min_length=1)


class GraphNodeOutput(BaseModel):
    id: str
    labels: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)
    found: bool = True
