# © Artur Czarnecki. All rights reserved.

"""Declarative intent → capability routes for rules classifier (ORCH-CONFIG.1)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class IntentRoute(BaseModel):
    """Match user message keywords to a Nexus capability before classification."""

    model_config = ConfigDict(extra="forbid")

    capability: str = Field(min_length=1)
    keywords: list[str] = Field(default_factory=list)
    case_insensitive: bool = True
