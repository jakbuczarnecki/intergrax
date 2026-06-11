# © Artur Czarnecki. All rights reserved.

"""Shared typed models for cognitive patterns (architecture §24.3)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CognitiveEvaluation(StrEnum):
    CONTINUE = "continue"
    COMPLETE = "complete"
    FAIL = "fail"
    REPLAN = "replan"
    HUMAN = "human"


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class ReasoningResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thought: str = ""
    planned_tool_ids: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)


class AgentEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: CognitiveEvaluation
    reason: str = ""
    confidence: float | None = None
