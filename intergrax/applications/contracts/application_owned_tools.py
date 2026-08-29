# © Artur Czarnecki. All rights reserved.

"""Canonical declaration contract for application-owned tools (PLATFORM-5B)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ApplicationOwnedToolDeclaration(BaseModel):
    """Declares one tool identity owned by the hosting application."""

    model_config = ConfigDict(extra="forbid")

    tool_id: str = Field(
        ...,
        min_length=1,
        description="Canonical tool id — must match ToolProfile, ToolRegistry, and traces.",
    )

    @field_validator("tool_id")
    @classmethod
    def _normalize_tool_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("tool_id must be non-empty")
        return normalized
