# © Artur Czarnecki. All rights reserved.

"""Minimal step context for Wave 0 authoring helpers (ACP-STEP-1 expands)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentStepContext(BaseModel):
    """Author-facing step context snapshot (architecture §32.2 target)."""

    model_config = ConfigDict(extra="forbid")

    step_index: int = Field(default=0, ge=0)
    run_id: str = ""
    state_snapshot: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
