# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neutral agent/orchestration runtime request and answer models (EE public surface)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant
from intergrax.contracts.delegation_authority import (
    EffectiveDelegationAuthority,
    ParentExecutionAuthority,
)
from intergrax.contracts.execution_identity import (
    RunId,
    TaskId,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.task_envelope import TaskEnvelope
from intergrax.contracts.tracing import TraceEvent
from intergrax.llm.messages import AttachmentRef
from intergrax.llm_adapters.contracts.llm_usage_report import LLMUsageReport
from intergrax.runtime.long_running.runtime_checkpoint import RuntimeCheckpoint
from intergrax.runtime.task.task_contract import HumanApprovalResolution, TaskPauseRecord


@dataclass
class Citation:
    source_id: str
    source_type: str
    source_label: str | None = None
    url: str | None = None
    score: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteInfo:
    used_rag: bool = False
    used_websearch: bool = False
    used_tools: bool = False
    used_user_profile: bool = False
    used_user_longterm_memory: bool = False
    strategy: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallInfo:
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result_summary: str | None = None
    success: bool = True
    error_message: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeStats:
    total_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    rag_tokens: int | None = None
    websearch_tokens: int | None = None
    tool_tokens: int | None = None
    duration_ms: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class HistoryCompressionStrategy(Enum):
    OFF = "off"
    TRUNCATE_OLDEST = "truncate_oldest"
    SUMMARIZE_OLDEST = "summarize_oldest"
    HYBRID = "hybrid"


class StopReason(Enum):
    COMPLETED = "completed"
    NEEDS_USER_INPUT = "needs_user_input"
    ABORTED = "aborted"
    ERROR = "error"


@dataclass
class RuntimeRequest:
    """High-level request structure for agent/orchestration execution."""

    agent_id: str
    user_id: str
    session_id: str
    message: str
    task_id: TaskId
    run_id: RunId
    attachments: list[AttachmentRef] = field(default_factory=list)
    workspace_id: str | None = None
    tenant_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    hitl_resolution: HumanApprovalResolution | None = None
    canonical_identity: RequestIdentity | None = None
    execution_authority: ParentExecutionAuthority | None = None
    effective_delegation_authority: EffectiveDelegationAuthority | None = None
    hitl_pause_record: TaskPauseRecord | None = None
    declarative_hitl_grant: DeclarativeHitlApprovalGrant | None = None
    runtime_checkpoint: RuntimeCheckpoint | None = None
    instructions: str | None = None
    history_compression_strategy: HistoryCompressionStrategy = (
        HistoryCompressionStrategy.TRUNCATE_OLDEST
    )
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        self.task_id = validate_task_id(self.task_id)
        self.run_id = validate_run_id(self.run_id)

    def to_envelope(self) -> TaskEnvelope:
        if self.tenant_id is None or not str(self.tenant_id).strip():
            raise ValueError("tenant_id is required for RuntimeRequest envelope")
        meta_tenant = self.metadata.get("tenant_id")
        if meta_tenant is not None and str(meta_tenant).strip():
            if str(meta_tenant).strip() != str(self.tenant_id).strip():
                raise ValueError(
                    "metadata tenant_id cannot override canonical RuntimeRequest.tenant_id"
                )
        return TaskEnvelope(
            tenant_id=str(self.tenant_id).strip(),
            user_id=self.user_id,
            message=self.message,
            session_id=self.session_id,
            agent_id=self.agent_id,
            workspace_id=self.workspace_id,
            metadata=dict(self.metadata),
            canonical_identity=self.canonical_identity,
        )

    @classmethod
    def from_envelope(
        cls,
        envelope: TaskEnvelope,
        *,
        task_id: TaskId,
        run_id: RunId,
    ) -> RuntimeRequest:
        return cls(
            agent_id=envelope.agent_id or "",
            user_id=envelope.user_id,
            session_id=envelope.session_id or f"sess_{envelope.tenant_id}",
            message=envelope.message,
            task_id=task_id,
            run_id=run_id,
            workspace_id=envelope.workspace_id,
            tenant_id=envelope.tenant_id,
            metadata=dict(envelope.metadata),
            canonical_identity=envelope.canonical_identity,
        )


def canonical_runtime_request_tenant_id(request: RuntimeRequest) -> str:
    """Return typed ``RuntimeRequest.tenant_id``; metadata cannot override or substitute."""
    if request.tenant_id is None or not str(request.tenant_id).strip():
        raise ValueError("tenant_id is required for RuntimeRequest")
    tenant = str(request.tenant_id).strip()
    meta_tenant = request.metadata.get("tenant_id")
    if meta_tenant is not None and str(meta_tenant).strip():
        if str(meta_tenant).strip() != tenant:
            raise ValueError(
                "metadata tenant_id cannot override canonical RuntimeRequest.tenant_id"
            )
    return tenant


@dataclass
class RuntimeAnswer:
    answer: str
    stop_reason: StopReason = StopReason.COMPLETED
    run_id: str | None = None
    citations: list[Citation] = field(default_factory=list)
    route: RouteInfo = field(default_factory=RouteInfo)
    tool_calls: list[ToolCallInfo] = field(default_factory=list)
    stats: RuntimeStats = field(default_factory=RuntimeStats)
    llm_usage_report: LLMUsageReport | None = None
    raw_model_output: Any | None = None
    trace_events: list[TraceEvent] = field(default_factory=list)


__all__ = [
    "Citation",
    "HistoryCompressionStrategy",
    "RouteInfo",
    "RuntimeAnswer",
    "RuntimeRequest",
    "RuntimeStats",
    "StopReason",
    "ToolCallInfo",
]
