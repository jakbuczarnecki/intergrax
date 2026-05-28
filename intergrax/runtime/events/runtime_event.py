# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical runtime events (architecture §42.1)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_phase import ExecutionPhase


class RuntimeEventType(str, Enum):
    TASK_CREATED = "task_created"
    TASK_CLASSIFIED = "task_classified"
    PLAN_CREATED = "plan_created"
    PLAN_UPDATED = "plan_updated"
    PLAN_FAILED = "plan_failed"
    AGENT_SELECTED = "agent_selected"
    CONTEXT_BUILT = "context_built"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    TOOL_REQUESTED = "tool_requested"
    TOOL_COMPLETED = "tool_completed"
    TOOL_DENIED = "tool_denied"
    TOOL_FAILED = "tool_failed"
    VALIDATION_STARTED = "validation_started"
    VALIDATION_PASSED = "validation_passed"
    VALIDATION_FAILED = "validation_failed"
    DECISION_EMITTED = "decision_emitted"
    INTERRUPT_REQUESTED = "interrupt_requested"
    INTERRUPT_HANDLED = "interrupt_handled"
    INTERRUPT_ESCALATED = "interrupt_escalated"
    HUMAN_APPROVAL_REQUESTED = "human_approval_requested"
    HUMAN_APPROVAL_RECEIVED = "human_approval_received"
    HUMAN_APPROVAL_TIMEOUT = "human_approval_timeout"
    PAUSE_REQUESTED = "pause_requested"
    PAUSED = "paused"
    RESUMED = "resumed"
    RETRY_SCHEDULED = "retry_scheduled"
    RETRY_STARTED = "retry_started"
    CANCELLATION_REQUESTED = "cancellation_requested"
    CANCELLED = "cancelled"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    HANDOFF_INITIATED = "handoff_initiated"
    HANDOFF_COMPLETED = "handoff_completed"
    TRACE_PERSISTED = "trace_persisted"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    RUNTIME_HANDLER_FAILED = "runtime_handler_failed"


class RuntimeEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"evt_{uuid4().hex}")
    tenant_id: Optional[str] = None
    task_id: str
    run_id: str
    node_id: Optional[str] = None
    agent_id: Optional[str] = None
    step_id: Optional[str] = None
    event_type: RuntimeEventType
    phase: ExecutionPhase
    severity: EventSeverity = EventSeverity.INFO
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str = ""
    parent_event_id: Optional[str] = None
    schema_version: str = "runtime_event.v1"

    def with_parent(self, parent: RuntimeEvent) -> RuntimeEvent:
        return self.model_copy(
            update={
                "parent_event_id": parent.event_id,
                "correlation_id": parent.correlation_id or parent.event_id,
            }
        )
