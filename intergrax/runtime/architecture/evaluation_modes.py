# © Artur Czarnecki. All rights reserved.

"""Unified evaluation mode contracts (Phase V-EVAL.1)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class EvaluationMode(str, Enum):
    OFFLINE = "offline"
    ONLINE = "online"
    SHADOW = "shadow"
    HUMAN = "human"


class EvaluationModeRequest(BaseModel):
    run_id: str
    target_id: str
    mode: EvaluationMode
    dataset_ref: str = ""
    traffic_slice_ref: str = ""
    reviewer_ref: str = ""
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_mode_requirements(self) -> "EvaluationModeRequest":
        if self.mode == EvaluationMode.OFFLINE and not self.dataset_ref:
            raise ValueError("Offline evaluation requires dataset_ref")
        if self.mode in {EvaluationMode.ONLINE, EvaluationMode.SHADOW} and not self.traffic_slice_ref:
            raise ValueError("Online and shadow evaluation require traffic_slice_ref")
        if self.mode == EvaluationMode.HUMAN and not self.reviewer_ref:
            raise ValueError("Human evaluation requires reviewer_ref")
        return self


class EvaluationModeResult(BaseModel):
    run_id: str
    target_id: str
    mode: EvaluationMode
    success: bool
    score: float
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    evidence_refs: list[str] = Field(default_factory=list)


class UnifiedEvaluationReport(BaseModel):
    schema_version: str = "1.0.0"
    requests: list[EvaluationModeRequest] = Field(default_factory=list)
    results: list[EvaluationModeResult] = Field(default_factory=list)
