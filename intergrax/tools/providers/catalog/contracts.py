# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field


class CatalogListToolsInput(BaseModel):
    category: str = Field(default="", description="Optional category filter (empty = all).")
    tag: str = Field(default="", description="Optional tag filter (empty = all).")


class CatalogToolSummary(BaseModel):
    tool_id: str
    name: str
    description_short: str = ""
    category: str = ""
    risk_level: str = ""
    side_effects: bool = False


class CatalogListToolsOutput(BaseModel):
    tools: list[CatalogToolSummary] = Field(default_factory=list)
    total: int = 0


class CatalogDescribeToolInput(BaseModel):
    tool_id: str = Field(..., min_length=1)


class CatalogDescribeToolOutput(BaseModel):
    found: bool = False
    tool_id: str = ""
    name: str = ""
    description: str = ""
    description_short: str = ""
    category: str = ""
    risk_level: str = ""
    side_effects: bool = False
    input_schema: dict = Field(default_factory=dict)
    output_schema: dict = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
