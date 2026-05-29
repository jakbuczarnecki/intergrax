# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""JSON response models for debug API (Phase D.2)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from intergrax.runtime.nexus.tracing.persistence_models import PersistedRun, RunSummary


class RunSummaryItem(BaseModel):
    run_id: str
    tenant_id: str
    user_id: str
    session_id: str
    started_at_utc: str
    duration_ms: int
    event_count: int

    @classmethod
    def from_summary(cls, summary: RunSummary) -> RunSummaryItem:
        return cls(
            run_id=summary.run_id,
            tenant_id=summary.tenant_id,
            user_id=summary.user_id,
            session_id=summary.session_id,
            started_at_utc=summary.started_at_utc,
            duration_ms=summary.duration_ms,
            event_count=summary.event_count,
        )


class RunListResponse(BaseModel):
    tenant_id: str
    count: int
    runs: List[RunSummaryItem]


class RunStatsResponse(BaseModel):
    duration_ms: int
    cost: Optional[float] = None
    total_tokens: Optional[int] = None
    llm_usage: Dict[str, Any] = Field(default_factory=dict)


class RunErrorResponse(BaseModel):
    error_type: str
    message: str


class RunDetailResponse(BaseModel):
    run_id: str
    tenant_id: str
    user_id: str
    session_id: str
    started_at_utc: str
    stats: RunStatsResponse
    event_count: int
    error: Optional[RunErrorResponse] = None

    @classmethod
    def from_persisted(cls, persisted: PersistedRun) -> RunDetailResponse:
        meta = persisted.metadata
        error = None
        if meta.error is not None:
            error = RunErrorResponse(
                error_type=meta.error.error_type.value,
                message=meta.error.message,
            )
        return cls(
            run_id=meta.run_id,
            tenant_id=meta.tenant_id,
            user_id=meta.user_id,
            session_id=meta.session_id,
            started_at_utc=meta.started_at_utc,
            stats=RunStatsResponse(
                duration_ms=meta.stats.duration_ms,
                cost=_coerce_optional_float((meta.stats.llm_usage or {}).get("cost")),
                total_tokens=_coerce_optional_int((meta.stats.llm_usage or {}).get("total_tokens")),
                llm_usage=dict(meta.stats.llm_usage or {}),
            ),
            event_count=len(persisted.events),
            error=error,
        )


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class TraceResponse(BaseModel):
    run_id: str
    tenant_id: str
    trace_events: List[Dict[str, Any]]
    runtime_events: Optional[List[Dict[str, Any]]] = None


class ExperimentListResponse(BaseModel):
    count: int
    experiments: List[Dict[str, Any]]


class ExperimentDeletedResponse(BaseModel):
    experiment_id: str
    deleted: bool = True


class RuntimeEventItem(BaseModel):
    event_id: str
    event_type: str
    task_id: str
    run_id: str
    tenant_id: Optional[str] = None
    phase: str
    severity: str
    timestamp: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class RuntimeEventListResponse(BaseModel):
    task_id: str
    tenant_id: str
    count: int
    events: List[RuntimeEventItem]


class CheckpointItem(BaseModel):
    checkpoint_id: str
    task_id: str
    tenant_id: str
    resume_token: str
    task_state: str
    progress_message: str = ""
    notify_channel: Optional[str] = None
    created_at_utc: str = ""
    has_runtime_checkpoint: bool = False


class CheckpointListResponse(BaseModel):
    task_id: str
    tenant_id: str
    count: int
    checkpoints: List[CheckpointItem]


class PartialResultItem(BaseModel):
    checkpoint_id: str
    progress_message: str
    task_state: str
    created_at_utc: str
    uaep_step_index: Optional[int] = None
    uaep_step_id: Optional[str] = None
    last_step_summary: Optional[str] = None
    partial_payload: Dict[str, Any] = Field(default_factory=dict)


class TaskProgressResponse(BaseModel):
    task_id: str
    tenant_id: str
    task_state: str
    progress_message: str = ""
    resume_token: Optional[str] = None
    checkpoint_id: Optional[str] = None
    notify_channel: Optional[str] = None
    human_request_expires_at: Optional[str] = None
    is_paused: bool = False
    checkpoint_count: int = 0
    progress_event_count: int = 0
    partial_results: List[PartialResultItem] = Field(default_factory=list)
    latest_partial_result: Optional[PartialResultItem] = None


class SubmitHumanResponseRequest(BaseModel):
    response: str = Field(description="Human verdict text: approve, reject, escalate, …")
    resume_token: Optional[str] = Field(
        default=None,
        description="Optional resume token; defaults to latest checkpoint",
    )
    user_id: Optional[str] = Field(default=None, description="Operator id for audit")


class HumanResponseResult(BaseModel):
    task_id: str
    run_id: Optional[str] = None
    state: str
    answer: str = ""
    resume_token: Optional[str] = None
    checkpoint_id: Optional[str] = None


class InteractionIntakeResponse(BaseModel):
    task_id: str
    tenant_id: str
    user_id: str
    capability: Optional[str] = None
    message: str = ""
    interaction_channel: str = ""
    executed: bool = False
    state: Optional[str] = None
    answer: Optional[str] = None
    run_id: Optional[str] = None
    resume_token: Optional[str] = None
    checkpoint_id: Optional[str] = None
