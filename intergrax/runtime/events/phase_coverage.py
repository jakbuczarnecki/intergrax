# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ExecutionPhase and ops-filter coverage for RuntimeEventType (§42.31, DX-5.7)."""

from __future__ import annotations

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEventType

# Ops scrape / alert routing hints (canon §42.1.5). Values are stable filter tokens.
OpsFilterHint = str

EVENT_PHASE_COVERAGE: dict[RuntimeEventType, ExecutionPhase] = {
    RuntimeEventType.TASK_CREATED: ExecutionPhase.INTAKE,
    RuntimeEventType.TASK_CLASSIFIED: ExecutionPhase.CLASSIFICATION,
    RuntimeEventType.PLAN_CREATED: ExecutionPhase.PLANNING,
    RuntimeEventType.PLAN_UPDATED: ExecutionPhase.PLANNING,
    RuntimeEventType.PLAN_FAILED: ExecutionPhase.PLANNING,
    RuntimeEventType.AGENT_SELECTED: ExecutionPhase.AGENT_SELECTION,
    RuntimeEventType.CONTEXT_BUILT: ExecutionPhase.CONTEXT_BUILDING,
    RuntimeEventType.CONTEXT_ASSEMBLED: ExecutionPhase.CONTEXT_BUILDING,
    RuntimeEventType.CONTEXT_TRIMMED: ExecutionPhase.CONTEXT_BUILDING,
    RuntimeEventType.INGESTION_FAILED: ExecutionPhase.CONTEXT_BUILDING,
    RuntimeEventType.SKILL_RESOLVED: ExecutionPhase.AGENT_SELECTION,
    RuntimeEventType.SKILL_IMPORT_FAILED: ExecutionPhase.AGENT_SELECTION,
    RuntimeEventType.STEP_STARTED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.STEP_COMPLETED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.STEP_FAILED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.TOOL_REQUESTED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.TOOL_COMPLETED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.TOOL_DENIED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.TOOL_FAILED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.VALIDATION_STARTED: ExecutionPhase.VALIDATION,
    RuntimeEventType.VALIDATION_PASSED: ExecutionPhase.VALIDATION,
    RuntimeEventType.VALIDATION_FAILED: ExecutionPhase.VALIDATION,
    RuntimeEventType.DECISION_EMITTED: ExecutionPhase.FINALIZATION,
    RuntimeEventType.INTERRUPT_REQUESTED: ExecutionPhase.INTERRUPT_HANDLING,
    RuntimeEventType.INTERRUPT_HANDLED: ExecutionPhase.INTERRUPT_HANDLING,
    RuntimeEventType.INTERRUPT_ESCALATED: ExecutionPhase.INTERRUPT_HANDLING,
    RuntimeEventType.HUMAN_APPROVAL_REQUESTED: ExecutionPhase.HUMAN_APPROVAL,
    RuntimeEventType.HUMAN_APPROVAL_RECEIVED: ExecutionPhase.HUMAN_APPROVAL,
    RuntimeEventType.HUMAN_APPROVAL_TIMEOUT: ExecutionPhase.HUMAN_APPROVAL,
    RuntimeEventType.PAUSE_REQUESTED: ExecutionPhase.HUMAN_APPROVAL,
    RuntimeEventType.PAUSED: ExecutionPhase.HUMAN_APPROVAL,
    RuntimeEventType.RESUMED: ExecutionPhase.HUMAN_APPROVAL,
    RuntimeEventType.RETRY_SCHEDULED: ExecutionPhase.RETRY_HANDLING,
    RuntimeEventType.RETRY_STARTED: ExecutionPhase.RETRY_HANDLING,
    RuntimeEventType.CANCELLATION_REQUESTED: ExecutionPhase.COMPLETION,
    RuntimeEventType.CANCELLED: ExecutionPhase.COMPLETION,
    RuntimeEventType.MEMORY_READ: ExecutionPhase.CONTEXT_BUILDING,
    RuntimeEventType.MEMORY_WRITE: ExecutionPhase.CONTEXT_BUILDING,
    RuntimeEventType.HANDOFF_INITIATED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.HANDOFF_COMPLETED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.TRACE_PERSISTED: ExecutionPhase.TRACE_PERSISTENCE,
    RuntimeEventType.TASK_PROGRESS: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.TASK_COMPLETED: ExecutionPhase.COMPLETION,
    RuntimeEventType.TASK_FAILED: ExecutionPhase.COMPLETION,
    RuntimeEventType.RUNTIME_HANDLER_FAILED: ExecutionPhase.INTERRUPT_HANDLING,
    RuntimeEventType.ADAPTIVE_SIGNAL_RECORDED: ExecutionPhase.TRACE_PERSISTENCE,
    RuntimeEventType.ADAPTIVE_PROPOSAL_SUBMITTED: ExecutionPhase.FINALIZATION,
    RuntimeEventType.ADAPTIVE_PROFILE_APPLIED: ExecutionPhase.FINALIZATION,
    RuntimeEventType.ADAPTIVE_PROFILE_ROLLBACK: ExecutionPhase.FINALIZATION,
    RuntimeEventType.ADAPTIVE_VERIFICATION_FAILED: ExecutionPhase.FINALIZATION,
    RuntimeEventType.ADAPTIVE_LOOP_BLOCKED: ExecutionPhase.FINALIZATION,
}

EVENT_OPS_FILTER_HINTS: dict[RuntimeEventType, OpsFilterHint] = {
    RuntimeEventType.TASK_CREATED: "trace:intake",
    RuntimeEventType.TASK_CLASSIFIED: "trace:classification",
    RuntimeEventType.PLAN_CREATED: "ops:planning",
    RuntimeEventType.PLAN_UPDATED: "ops:planning",
    RuntimeEventType.PLAN_FAILED: "ops:alert",
    RuntimeEventType.AGENT_SELECTED: "trace:selection",
    RuntimeEventType.CONTEXT_BUILT: "trace:context",
    RuntimeEventType.CONTEXT_ASSEMBLED: "trace:context",
    RuntimeEventType.CONTEXT_TRIMMED: "trace:context",
    RuntimeEventType.INGESTION_FAILED: "ops:alert",
    RuntimeEventType.SKILL_RESOLVED: "trace:skills",
    RuntimeEventType.SKILL_IMPORT_FAILED: "ops:alert",
    RuntimeEventType.STEP_STARTED: "trace:step",
    RuntimeEventType.STEP_COMPLETED: "trace:step",
    RuntimeEventType.STEP_FAILED: "ops:alert",
    RuntimeEventType.TOOL_REQUESTED: "ops:tool_audit",
    RuntimeEventType.TOOL_COMPLETED: "ops:tool_audit",
    RuntimeEventType.TOOL_DENIED: "ops:alert",
    RuntimeEventType.TOOL_FAILED: "ops:alert",
    RuntimeEventType.VALIDATION_STARTED: "trace:validation",
    RuntimeEventType.VALIDATION_PASSED: "trace:validation",
    RuntimeEventType.VALIDATION_FAILED: "ops:alert",
    RuntimeEventType.DECISION_EMITTED: "trace:decision",
    RuntimeEventType.INTERRUPT_REQUESTED: "ops:hitl",
    RuntimeEventType.INTERRUPT_HANDLED: "ops:hitl",
    RuntimeEventType.INTERRUPT_ESCALATED: "ops:alert",
    RuntimeEventType.HUMAN_APPROVAL_REQUESTED: "ops:hitl",
    RuntimeEventType.HUMAN_APPROVAL_RECEIVED: "ops:hitl",
    RuntimeEventType.HUMAN_APPROVAL_TIMEOUT: "ops:alert",
    RuntimeEventType.PAUSE_REQUESTED: "ops:hitl",
    RuntimeEventType.PAUSED: "ops:hitl",
    RuntimeEventType.RESUMED: "ops:hitl",
    RuntimeEventType.RETRY_SCHEDULED: "ops:retry",
    RuntimeEventType.RETRY_STARTED: "ops:retry",
    RuntimeEventType.CANCELLATION_REQUESTED: "ops:completion",
    RuntimeEventType.CANCELLED: "ops:completion",
    RuntimeEventType.MEMORY_READ: "ops:memory",
    RuntimeEventType.MEMORY_WRITE: "ops:memory",
    RuntimeEventType.HANDOFF_INITIATED: "ops:handoff",
    RuntimeEventType.HANDOFF_COMPLETED: "ops:handoff",
    RuntimeEventType.TRACE_PERSISTED: "trace:persistence",
    RuntimeEventType.TASK_PROGRESS: "ops:progress",
    RuntimeEventType.TASK_COMPLETED: "ops:completion",
    RuntimeEventType.TASK_FAILED: "ops:alert",
    RuntimeEventType.RUNTIME_HANDLER_FAILED: "ops:alert",
    RuntimeEventType.ADAPTIVE_SIGNAL_RECORDED: "ops:adaptive",
    RuntimeEventType.ADAPTIVE_PROPOSAL_SUBMITTED: "ops:adaptive",
    RuntimeEventType.ADAPTIVE_PROFILE_APPLIED: "ops:adaptive",
    RuntimeEventType.ADAPTIVE_PROFILE_ROLLBACK: "ops:alert",
    RuntimeEventType.ADAPTIVE_VERIFICATION_FAILED: "ops:alert",
    RuntimeEventType.ADAPTIVE_LOOP_BLOCKED: "ops:alert",
}


def phase_for_event(event_type: RuntimeEventType) -> ExecutionPhase | None:
    return EVENT_PHASE_COVERAGE.get(event_type)


def ops_filter_hint_for_event(event_type: RuntimeEventType) -> OpsFilterHint | None:
    return EVENT_OPS_FILTER_HINTS.get(event_type)


def list_unmapped_event_types() -> list[RuntimeEventType]:
    return [item for item in RuntimeEventType if item not in EVENT_PHASE_COVERAGE]


def list_unmapped_ops_filter_hints() -> list[RuntimeEventType]:
    return [item for item in RuntimeEventType if item not in EVENT_OPS_FILTER_HINTS]
