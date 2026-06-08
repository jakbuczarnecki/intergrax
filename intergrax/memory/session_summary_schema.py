# © Artur Czarnecki. All rights reserved.

"""Structured session summary schema (Phase MEM-DEPTH-3.5)."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class SessionSummarySchema(BaseModel):
    """Structured recap produced by consolidation — not a plain text blob."""

    model_config = ConfigDict(extra="forbid")

    title: str = ""
    narrative: str = ""
    facts: List[str] = Field(default_factory=list)
    open_tasks: List[str] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list)
    session_id: Optional[str] = None

    def to_storage_text(self) -> str:
        parts = [self.narrative.strip()] if self.narrative.strip() else []
        if self.facts:
            parts.append("Facts: " + "; ".join(self.facts))
        if self.decisions:
            parts.append("Decisions: " + "; ".join(self.decisions))
        if self.open_tasks:
            parts.append("Open tasks: " + "; ".join(self.open_tasks))
        return "\n".join(parts).strip()
