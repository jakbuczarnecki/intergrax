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
                llm_usage=dict(meta.stats.llm_usage or {}),
            ),
            event_count=len(persisted.events),
            error=error,
        )


class TraceResponse(BaseModel):
    run_id: str
    tenant_id: str
    trace_events: List[Dict[str, Any]]
    runtime_events: Optional[List[Dict[str, Any]]] = None
