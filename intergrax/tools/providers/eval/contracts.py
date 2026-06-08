# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field


class EvalRecordObservationInput(BaseModel):
    observation_id: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    agent_id: str = Field(..., min_length=1)
    mode: str = Field(default="shadow", pattern="^(online|shadow)$")
    scenario_id: str = Field(..., min_length=1)
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    candidate_profile_version_id: str | None = None


class EvalRecordObservationOutput(BaseModel):
    recorded: bool = True
    observation_id: str


class EvalListObservationsInput(BaseModel):
    limit: int = Field(default=100, ge=1, le=1000)


class EvalObservationOutput(BaseModel):
    observation_id: str
    run_id: str
    agent_id: str
    mode: str
    scenario_id: str
    passed: bool
    score: float
    candidate_profile_version_id: str | None = None
    recorded_at: str = ""


class EvalListObservationsOutput(BaseModel):
    observations: list[EvalObservationOutput] = Field(default_factory=list)
    total: int = 0
    pass_rate: float = 0.0
    average_score: float = 0.0


class EvalSummarizeReleaseInput(BaseModel):
    release_id: str = Field(..., min_length=1)


class EvalSummarizeReleaseOutput(BaseModel):
    release_id: str
    observation_count: int = 0
    pass_rate: float = 0.0
    average_score: float = 0.0
    passed_count: int = 0
    failed_count: int = 0


class EvalCompareReleasesInput(BaseModel):
    baseline_release_id: str = Field(..., min_length=1)
    candidate_release_id: str = Field(..., min_length=1)


class EvalCompareReleasesOutput(BaseModel):
    baseline_release_id: str
    candidate_release_id: str
    baseline_observation_count: int = 0
    candidate_observation_count: int = 0
    baseline_pass_rate: float = 0.0
    candidate_pass_rate: float = 0.0
    pass_rate_delta: float = 0.0
    baseline_average_score: float = 0.0
    candidate_average_score: float = 0.0
    score_delta: float = 0.0
    candidate_better: bool = False


class EvalExportObservationsInput(BaseModel):
    limit: int = Field(default=1000, ge=1, le=5000)


class EvalExportObservationsOutput(BaseModel):
    exported: bool = True
    observation_count: int = 0
    export_json: str = ""


class EvalJudgeInput(BaseModel):
    output_text: str = Field(..., min_length=1)
    rubric_id: str = Field(..., min_length=1)
    criteria: list[str] = Field(default_factory=list)
    reference_context: str | None = None
    min_score: float = Field(default=0.75, ge=0.0, le=1.0)
    run_id: str | None = None
    agent_id: str | None = None
    record_observation: bool = False
    observation_id: str | None = None
    scenario_id: str | None = None
    mode: str = Field(default="online", pattern="^(online|shadow)$")
    candidate_profile_version_id: str | None = None


class EvalJudgeOutput(BaseModel):
    rubric_id: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    observation_recorded: bool = False


class EvalTrajectoryInput(BaseModel):
    run_id: str = Field(..., min_length=1)
    tenant_id: str = Field(default="default", min_length=1)
    min_score: float = Field(default=0.75, ge=0.0, le=1.0)
    agent_id: str = Field(default="unknown", min_length=1)
    record_observation: bool = False
    observation_id: str | None = None
    scenario_id: str | None = None
    mode: str = Field(default="online", pattern="^(online|shadow)$")
    candidate_profile_version_id: str | None = None


class EvalTrajectoryOutput(BaseModel):
    run_id: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    tool_call_count: int = 0
    duplicate_tool_calls: int = 0
    error_count: int = 0
    denied_count: int = 0
    observation_recorded: bool = False
