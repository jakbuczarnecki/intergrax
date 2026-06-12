# © Artur Czarnecki. All rights reserved.

"""Optional agent contract context assembly hints (CE-5.1)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AgentContextHints(BaseModel):
    """Declarative CE hints consumed by ACP assembly bridge (CE-4.2 / CE-5.1)."""

    model_config = ConfigDict(extra="forbid")

    required_sources: list[str] = Field(default_factory=list)
    excluded_sources: list[str] = Field(default_factory=list)
    step_kind: str | None = None
