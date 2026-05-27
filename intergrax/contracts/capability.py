# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class CapabilityMatchResult(BaseModel):
    """Outcome of capability matching for agent selection (§16)."""

    matched: bool
    agent_id: Optional[str] = None
    matched_capabilities: List[str] = Field(default_factory=list)
    score: float = 0.0
    rationale: str = ""
