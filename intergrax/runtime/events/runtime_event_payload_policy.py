# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Single SSOT for RuntimeEventType payload policy (OBS-DIAG-EC1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.runtime.events.payloads.canonical import (
    AgentSelectionPayloadV1,
    ContextAssemblyPayloadV1,
    ContextAssemblyPayloadV2,
    ContextCandidatePayloadV1,
    DecisionPayloadV1,
    DelegationGrantedPayloadV1,
    ExecutionFailurePayloadV1,
    ExternalOperationFailurePayloadV1,
    GraphNodePayloadV1,
    HandoffPayloadV1,
    HumanPayloadV1,
    InterruptPayloadV1,
    LlmCallPayloadV1,
    SkillResolvedPayloadV1,
    TaskLifecyclePayloadV1,
    ToolPayloadV1,
    ValidationPayloadV1,
)
from intergrax.runtime.events.payloads.spine_families import (
    BudgetSignalPayloadV1,
    CancellationLifecyclePayloadV1,
    GraphBackpressurePayloadV1,
    GuardrailBlockedPayloadV1,
    HumanTimeoutPayloadV1,
    MemoryAccessPayloadV1,
    OperationalAlertPayloadV1,
    PauseLifecyclePayloadV1,
    PlanLifecyclePayloadV1,
    PolicyDecisionSpinePayloadV1,
    RetryLifecyclePayloadV1,
    TaskProgressPayloadV1,
    TracePersistedPayloadV1,
)


class RuntimeEventTypeClassification(str, Enum):
    CANONICAL_PRODUCTION = "A"
    EXTENSION_CUSTOM = "B"
    HISTORICAL_LEGACY = "C"
    NON_PRODUCTION = "D"
    DEPRECATED = "E"


class PayloadWriteMode(str, Enum):
    STRICT_SPINE_SCHEMA = "strict_spine_schema"
    EXTENSION_EVENT_KIND = "extension_event_kind"


@dataclass(frozen=True, slots=True)
class RuntimeEventPayloadPolicyEntry:
    classification: RuntimeEventTypeClassification
    schema_id: str | None
    write_mode: PayloadWriteMode = PayloadWriteMode.STRICT_SPINE_SCHEMA


def _strict(schema_id: str) -> RuntimeEventPayloadPolicyEntry:
    return RuntimeEventPayloadPolicyEntry(
        classification=RuntimeEventTypeClassification.CANONICAL_PRODUCTION,
        schema_id=schema_id,
    )


_RUNTIME_EVENT_PAYLOAD_POLICY: dict[RuntimeEventType, RuntimeEventPayloadPolicyEntry] = {
    RuntimeEventType.TASK_CREATED: _strict(TaskLifecyclePayloadV1.schema_id),
    RuntimeEventType.TASK_CLASSIFIED: _strict(TaskLifecyclePayloadV1.schema_id),
    RuntimeEventType.PLAN_CREATED: _strict(PlanLifecyclePayloadV1.schema_id),
    RuntimeEventType.PLAN_UPDATED: _strict(PlanLifecyclePayloadV1.schema_id),
    RuntimeEventType.PLAN_FAILED: _strict(PlanLifecyclePayloadV1.schema_id),
    RuntimeEventType.AGENT_SELECTED: _strict(AgentSelectionPayloadV1.schema_id),
    RuntimeEventType.CONTEXT_BUILT: _strict(ContextAssemblyPayloadV1.schema_id),
    RuntimeEventType.CONTEXT_ASSEMBLED: _strict(ContextAssemblyPayloadV2.schema_id),
    RuntimeEventType.CONTEXT_TRIMMED: _strict(ContextAssemblyPayloadV2.schema_id),
    RuntimeEventType.CONTEXT_CANDIDATE_COLLECTED: _strict(ContextCandidatePayloadV1.schema_id),
    RuntimeEventType.CONTEXT_CANDIDATE_DROPPED: _strict(ContextCandidatePayloadV1.schema_id),
    RuntimeEventType.CONTEXT_VALIDATION_FAILED: _strict(ValidationPayloadV1.schema_id),
    RuntimeEventType.INGESTION_FAILED: _strict(OperationalAlertPayloadV1.schema_id),
    RuntimeEventType.SKILL_RESOLVED: _strict(SkillResolvedPayloadV1.schema_id),
    RuntimeEventType.SKILL_IMPORT_FAILED: _strict(OperationalAlertPayloadV1.schema_id),
    RuntimeEventType.STEP_STARTED: _strict(GraphNodePayloadV1.schema_id),
    RuntimeEventType.STEP_COMPLETED: _strict(GraphNodePayloadV1.schema_id),
    RuntimeEventType.STEP_FAILED: _strict(ValidationPayloadV1.schema_id),
    RuntimeEventType.EXECUTION_FAILED: _strict(ExecutionFailurePayloadV1.schema_id),
    RuntimeEventType.EXTERNAL_OPERATION_FAILED: _strict(ExternalOperationFailurePayloadV1.schema_id),
    RuntimeEventType.TOOL_REQUESTED: _strict(ToolPayloadV1.schema_id),
    RuntimeEventType.TOOL_COMPLETED: _strict(ToolPayloadV1.schema_id),
    RuntimeEventType.TOOL_DENIED: _strict(ToolPayloadV1.schema_id),
    RuntimeEventType.TOOL_FAILED: _strict(ToolPayloadV1.schema_id),
    RuntimeEventType.VALIDATION_STARTED: _strict(ValidationPayloadV1.schema_id),
    RuntimeEventType.VALIDATION_PASSED: _strict(ValidationPayloadV1.schema_id),
    RuntimeEventType.VALIDATION_FAILED: _strict(ValidationPayloadV1.schema_id),
    RuntimeEventType.DECISION_EMITTED: _strict(DecisionPayloadV1.schema_id),
    RuntimeEventType.INTERRUPT_REQUESTED: _strict(InterruptPayloadV1.schema_id),
    RuntimeEventType.INTERRUPT_HANDLED: _strict(InterruptPayloadV1.schema_id),
    RuntimeEventType.INTERRUPT_ESCALATED: _strict(OperationalAlertPayloadV1.schema_id),
    RuntimeEventType.HUMAN_APPROVAL_REQUESTED: _strict(HumanPayloadV1.schema_id),
    RuntimeEventType.HUMAN_APPROVAL_RECEIVED: _strict(HumanPayloadV1.schema_id),
    RuntimeEventType.HUMAN_APPROVAL_TIMEOUT: _strict(HumanTimeoutPayloadV1.schema_id),
    RuntimeEventType.PAUSE_REQUESTED: _strict(PauseLifecyclePayloadV1.schema_id),
    RuntimeEventType.PAUSED: _strict(PauseLifecyclePayloadV1.schema_id),
    RuntimeEventType.RESUMED: _strict(PauseLifecyclePayloadV1.schema_id),
    RuntimeEventType.RETRY_SCHEDULED: _strict(RetryLifecyclePayloadV1.schema_id),
    RuntimeEventType.RETRY_STARTED: _strict(RetryLifecyclePayloadV1.schema_id),
    RuntimeEventType.CANCELLATION_REQUESTED: _strict(CancellationLifecyclePayloadV1.schema_id),
    RuntimeEventType.CANCELLED: _strict(CancellationLifecyclePayloadV1.schema_id),
    RuntimeEventType.MEMORY_READ: _strict(MemoryAccessPayloadV1.schema_id),
    RuntimeEventType.MEMORY_WRITE: _strict(MemoryAccessPayloadV1.schema_id),
    RuntimeEventType.HANDOFF_INITIATED: _strict(HandoffPayloadV1.schema_id),
    RuntimeEventType.HANDOFF_COMPLETED: _strict(HandoffPayloadV1.schema_id),
    RuntimeEventType.DELEGATION_GRANTED: _strict(DelegationGrantedPayloadV1.schema_id),
    RuntimeEventType.TRACE_PERSISTED: _strict(TracePersistedPayloadV1.schema_id),
    RuntimeEventType.TASK_PROGRESS: _strict(TaskProgressPayloadV1.schema_id),
    RuntimeEventType.TASK_COMPLETED: _strict(TaskLifecyclePayloadV1.schema_id),
    RuntimeEventType.TASK_FAILED: _strict(TaskLifecyclePayloadV1.schema_id),
    RuntimeEventType.RUNTIME_HANDLER_FAILED: _strict(OperationalAlertPayloadV1.schema_id),
    RuntimeEventType.LLM_CALL: _strict(LlmCallPayloadV1.schema_id),
    RuntimeEventType.POLICY_DECISION: _strict(PolicyDecisionSpinePayloadV1.schema_id),
    RuntimeEventType.GRAPH_BACKPRESSURE: _strict(GraphBackpressurePayloadV1.schema_id),
    RuntimeEventType.GUARDRAIL_BLOCKED: _strict(GuardrailBlockedPayloadV1.schema_id),
    RuntimeEventType.BUDGET_THRESHOLD: _strict(BudgetSignalPayloadV1.schema_id),
    RuntimeEventType.BUDGET_EXCEEDED: _strict(BudgetSignalPayloadV1.schema_id),
    RuntimeEventType.DOMAIN_SIGNAL: RuntimeEventPayloadPolicyEntry(
        classification=RuntimeEventTypeClassification.CANONICAL_PRODUCTION,
        schema_id=None,
        write_mode=PayloadWriteMode.EXTENSION_EVENT_KIND,
    ),
}


def _validate_policy_completeness() -> None:
    missing = [member for member in RuntimeEventType if member not in _RUNTIME_EVENT_PAYLOAD_POLICY]
    if missing:
        raise RuntimeError(
            "incomplete runtime event payload policy: "
            + ", ".join(member.value for member in missing)
        )


_validate_policy_completeness()

CANONICAL_PRODUCTION_STRICT_SPINE_EVENT_TYPES: Final[frozenset[RuntimeEventType]] = frozenset(
    event_type
    for event_type, entry in _RUNTIME_EVENT_PAYLOAD_POLICY.items()
    if entry.classification == RuntimeEventTypeClassification.CANONICAL_PRODUCTION
    and entry.write_mode == PayloadWriteMode.STRICT_SPINE_SCHEMA
    and entry.schema_id is not None
)

EVENT_TYPE_PREFERRED_SCHEMA: Final[dict[RuntimeEventType, str]] = {
    event_type: entry.schema_id
    for event_type, entry in _RUNTIME_EVENT_PAYLOAD_POLICY.items()
    if entry.schema_id is not None
}


def get_runtime_event_payload_policy(
    event_type: RuntimeEventType,
) -> RuntimeEventPayloadPolicyEntry:
    return _RUNTIME_EVENT_PAYLOAD_POLICY[event_type]


def iter_runtime_event_payload_policies() -> tuple[
    tuple[RuntimeEventType, RuntimeEventPayloadPolicyEntry],
    ...,
]:
    return tuple(_RUNTIME_EVENT_PAYLOAD_POLICY.items())


__all__ = [
    "CANONICAL_PRODUCTION_STRICT_SPINE_EVENT_TYPES",
    "EVENT_TYPE_PREFERRED_SCHEMA",
    "PayloadWriteMode",
    "RuntimeEventPayloadPolicyEntry",
    "RuntimeEventTypeClassification",
    "get_runtime_event_payload_policy",
    "iter_runtime_event_payload_policies",
]
