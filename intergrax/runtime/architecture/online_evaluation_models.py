# © Artur Czarnecki. All rights reserved.

"""Typed models for harness online/shadow evaluation (W-OPS.11)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


class OnlineEvaluationMode(str, Enum):
    ONLINE = "online"
    SHADOW = "shadow"


class OnlineEvaluationObservation(BaseModel):
    observation_id: str
    run_id: str
    agent_id: str
    mode: OnlineEvaluationMode
    scenario_id: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class OnlineEvaluationBatch(BaseModel):
    release_id: str
    observations: list[OnlineEvaluationObservation] = Field(default_factory=list)


class OnlineEvaluationRegistryStore(BaseModel):
    schema_version: str = "1.0.0"
    observations: list[OnlineEvaluationObservation] = Field(default_factory=list)
