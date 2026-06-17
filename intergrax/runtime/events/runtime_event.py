# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical runtime events (architecture §42.1)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_taxonomy import EventCategory, category_for_event_kind


class RuntimeEventType(str, Enum):
    TASK_CREATED = "task_created"
    TASK_CLASSIFIED = "task_classified"
    PLAN_CREATED = "plan_created"
    PLAN_UPDATED = "plan_updated"
    PLAN_FAILED = "plan_failed"
    AGENT_SELECTED = "agent_selected"
    CONTEXT_BUILT = "context_built"
    CONTEXT_ASSEMBLED = "context_assembled"
    CONTEXT_TRIMMED = "context_trimmed"
    CONTEXT_CANDIDATE_COLLECTED = "context_candidate_collected"
    CONTEXT_CANDIDATE_DROPPED = "context_candidate_dropped"
    CONTEXT_VALIDATION_FAILED = "context_validation_failed"
    INGESTION_FAILED = "ingestion_failed"
    SKILL_RESOLVED = "skill_resolved"
    SKILL_IMPORT_FAILED = "skill_import_failed"
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
    DELEGATION_GRANTED = "delegation_granted"
    TRACE_PERSISTED = "trace_persisted"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    RUNTIME_HANDLER_FAILED = "runtime_handler_failed"
    LLM_CALL = "llm_call"
    POLICY_DECISION = "policy_decision"
    GRAPH_BACKPRESSURE = "graph_backpressure"
    GUARDRAIL_BLOCKED = "guardrail_blocked"
    BUDGET_THRESHOLD = "budget_threshold"
    BUDGET_EXCEEDED = "budget_exceeded"
    DOMAIN_SIGNAL = "domain_signal"


class RuntimeEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"evt_{uuid4().hex}")
    tenant_id: Optional[str] = None
    task_id: str
    run_id: str
    node_id: Optional[str] = None
    agent_id: Optional[str] = None
    step_id: Optional[str] = None
    event_type: RuntimeEventType
    event_kind: str = ""
    event_category: EventCategory | None = None
    ops_hint: str = ""
    phase: ExecutionPhase
    severity: EventSeverity = EventSeverity.INFO
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str = ""
    parent_event_id: Optional[str] = None
    schema_version: str = "runtime_event.v1"

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_spine(cls, data: Any) -> Any:
        from intergrax.runtime.events.spine_consolidation import migrate_legacy_spine_payload

        return migrate_legacy_spine_payload(data)

    def model_post_init(self, __context: Any) -> None:
        from intergrax.runtime.events.event_catalog import get_catalog_entry

        entry = get_catalog_entry(self.event_type)
        if entry is not None:
            if not self.event_kind:
                self.event_kind = entry.default_event_kind
            if self.event_category is None:
                self.event_category = entry.category
            if not self.ops_hint:
                self.ops_hint = entry.ops_hint
            return
        if not self.event_kind:
            self.event_kind = self.event_type.value
        if self.event_category is None:
            self.event_category = category_for_event_kind(self.event_kind)
        if not self.ops_hint and self.event_type == RuntimeEventType.DOMAIN_SIGNAL:
            self.ops_hint = "ops:domain_signal"

    def with_parent(self, parent: RuntimeEvent) -> RuntimeEvent:
        return self.model_copy(
            update={
                "parent_event_id": parent.event_id,
                "correlation_id": parent.correlation_id or parent.event_id,
            }
        )
