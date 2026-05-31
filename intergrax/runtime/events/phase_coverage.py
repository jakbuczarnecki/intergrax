# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ExecutionPhase coverage audit for RuntimeEventType (§42.31, B.07)."""

from __future__ import annotations

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEventType

EVENT_PHASE_COVERAGE: dict[RuntimeEventType, ExecutionPhase] = {
    RuntimeEventType.TASK_CREATED: ExecutionPhase.INTAKE,
    RuntimeEventType.TASK_CLASSIFIED: ExecutionPhase.CLASSIFICATION,
    RuntimeEventType.PLAN_CREATED: ExecutionPhase.PLANNING,
    RuntimeEventType.PLAN_UPDATED: ExecutionPhase.PLANNING,
    RuntimeEventType.PLAN_FAILED: ExecutionPhase.PLANNING,
    RuntimeEventType.AGENT_SELECTED: ExecutionPhase.AGENT_SELECTION,
    RuntimeEventType.CONTEXT_BUILT: ExecutionPhase.CONTEXT_BUILDING,
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
}


def phase_for_event(event_type: RuntimeEventType) -> ExecutionPhase | None:
    return EVENT_PHASE_COVERAGE.get(event_type)


def list_unmapped_event_types() -> list[RuntimeEventType]:
    return [item for item in RuntimeEventType if item not in EVENT_PHASE_COVERAGE]
