# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from intergrax.contracts.agent_execution_result import AgentExecutionResult


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
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_runtime_request(self) -> "RuntimeRequest":
        from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest

        if not self.agent_id:
            raise ValueError("Task.agent_id must be set before execution.")

        return RuntimeRequest(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            session_id=self.session_id or f"sess_{uuid4().hex}",
            agent_id=self.agent_id,
            message=self.message,
            metadata={**self.metadata, "run_id": self.task_id},
        )


class TaskResult(BaseModel):
    task_id: str
    run_id: Optional[str] = None
    state: TaskState
    answer: str = ""
    agent_id: Optional[str] = None
    execution_result: Optional["AgentExecutionResult"] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
