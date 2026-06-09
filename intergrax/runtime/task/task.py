# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.task_envelope import TaskEnvelope
from intergrax.runtime.task.task_contract import (
    TaskExecutionOptions,
    TaskResultSummary,
    TaskRuntimeState,
)


class TaskState(str, Enum):
    CREATED = "created"
    CLASSIFIED = "classified"
    PLANNED = "planned"
    WAITING_FOR_RESOURCES = "waiting_for_resources"
    WAITING_FOR_HUMAN = "waiting_for_human"
    RUNNING = "running"
    VALIDATING = "validating"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    NEEDS_MORE_INFORMATION = "needs_more_information"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class TaskContext(BaseModel):
    """Lightweight task context for capability routing."""

    capability: Optional[str] = None
    intent: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    """Normalized task object (canonical architecture §23, §41)."""

    task_id: str = Field(default_factory=lambda: f"task_{uuid4().hex}")
    tenant_id: str
    user_id: str
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    message: str = ""
    state: TaskState = TaskState.CREATED
    context: TaskContext = Field(default_factory=TaskContext)
    options: TaskExecutionOptions = Field(default_factory=TaskExecutionOptions)
    runtime: TaskRuntimeState = Field(default_factory=TaskRuntimeState)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    _registry: Optional[Any] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _hydrate_from_legacy_metadata(self) -> Task:
        from intergrax.runtime.task.task_metadata_bridge import (
            hydrate_task_from_metadata,
            metadata_needs_hydration,
        )

        if self.metadata.get("_hydrate_legacy") is False:
            return self
        if self.metadata.get("_hydrate_legacy") is True or metadata_needs_hydration(self):
            hydrate_task_from_metadata(self)
        return self

    @property
    def classification(self) -> Optional[str]:
        return self.runtime.classification.value

    def sync_metadata(self) -> None:
        from intergrax.runtime.task.task_metadata_bridge import sync_task_metadata

        sync_task_metadata(self)

    def to_envelope(self) -> TaskEnvelope:
        from intergrax.contracts.task_envelope import TaskRiskTier, TaskSlaClass

        sla_raw = self.metadata.get("sla_class", TaskSlaClass.INTERACTIVE.value)
        risk_raw = self.metadata.get("risk_tier", TaskRiskTier.LOW.value)
        constraints = self.metadata.get("constraints", {})
        meta = {
            k: v
            for k, v in self.metadata.items()
            if k not in {"workspace_id", "sla_class", "risk_tier", "constraints"}
        }
        return TaskEnvelope(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            message=self.message,
            session_id=self.session_id,
            agent_id=self.agent_id,
            workspace_id=self.metadata.get("workspace_id"),
            sla_class=TaskSlaClass(sla_raw),
            risk_tier=TaskRiskTier(risk_raw),
            constraints=dict(constraints) if isinstance(constraints, dict) else {},
            metadata=meta,
        )

    @classmethod
    def from_envelope(cls, envelope: TaskEnvelope) -> Task:
        metadata = dict(envelope.metadata)
        metadata["sla_class"] = envelope.sla_class.value
        metadata["risk_tier"] = envelope.risk_tier.value
        if envelope.constraints:
            metadata["constraints"] = dict(envelope.constraints)
        if envelope.workspace_id:
            metadata["workspace_id"] = envelope.workspace_id
        return cls(
            tenant_id=envelope.tenant_id,
            user_id=envelope.user_id,
            message=envelope.message,
            session_id=envelope.session_id,
            agent_id=envelope.agent_id,
            metadata=metadata,
        )

    def to_runtime_request(self) -> "RuntimeRequest":
        from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
        from intergrax.runtime.task.task_metadata_bridge import task_to_request_metadata

        if not self.agent_id:
            raise ValueError("Task.agent_id must be set before execution.")

        metadata = task_to_request_metadata(self)
        metadata["run_id"] = self.task_id
        metadata["task_id"] = self.task_id

        return RuntimeRequest(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            session_id=self.session_id or f"sess_{uuid4().hex}",
            agent_id=self.agent_id,
            message=self.message,
            metadata=metadata,
        )


class TaskResult(BaseModel):
    task_id: str
    run_id: Optional[str] = None
    state: TaskState
    answer: str = ""
    agent_id: Optional[str] = None
    execution_result: Optional[AgentExecutionResult] = None
    summary: TaskResultSummary = Field(default_factory=TaskResultSummary)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _inject_summary_from_legacy(cls, data: Any) -> Any:
        if isinstance(data, dict) and "summary" not in data:
            metadata = data.get("metadata")
            if isinstance(metadata, dict) and (
                metadata.get("task_result.v1") is not None
                or metadata.get("validation_valid") is not None
            ):
                from intergrax.runtime.task.task_metadata_bridge import (
                    result_summary_from_metadata,
                )

                data = dict(data)
                data["summary"] = result_summary_from_metadata(metadata).model_dump()
        return data

    def sync_metadata(self) -> None:
        from intergrax.runtime.task.task_metadata_bridge import sync_result_metadata

        sync_result_metadata(self)
