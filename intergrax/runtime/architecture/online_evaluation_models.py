# © Artur Czarnecki. All rights reserved.

"""Typed models for harness online/shadow evaluation (W-OPS.11)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class OnlineEvaluationMode(str, Enum):  # noqa: UP042
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
    candidate_profile_version_id: str | None = None
    tenant_id: str | None = None
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("tenant_id")
    @classmethod
    def _tenant_id_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("tenant_id must be a non-blank string")
        return value


class OnlineEvaluationBatch(BaseModel):
    release_id: str
    observations: list[OnlineEvaluationObservation] = Field(default_factory=list)


class OnlineEvaluationRegistryStore(BaseModel):
    schema_version: str = "1.0.0"
    observations: list[OnlineEvaluationObservation] = Field(default_factory=list)
