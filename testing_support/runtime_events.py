# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.contracts.execution_identity import (
    AttemptId,
    EventId,
    ExecutionId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.events.payload_registry import (
    EVENT_TYPE_PREFERRED_SCHEMA,
    runtime_event_with_payload,
)
from intergrax.runtime.events.payloads import RuntimeEventPayload
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
from intergrax.runtime.events.runtime_event import RuntimeEvent


def _minimal_payload_for_schema_id(schema_id: str) -> RuntimeEventPayload:
    builders: dict[str, RuntimeEventPayload] = {
        "agent_selection.v1": AgentSelectionPayloadV1(selected_agent_id="agent.test"),
        "context_assembly.v1": ContextAssemblyPayloadV1(
            node_id="node",
            context_original_chars=0,
            context_final_chars=0,
        ),
        "context_assembly.v2": ContextAssemblyPayloadV2(
            node_id="node",
            context_original_chars=0,
            context_final_chars=0,
        ),
        "context_candidate.v1": ContextCandidatePayloadV1(provider_id="provider"),
        "decision.v1": DecisionPayloadV1(decision_type="test", reason="test"),
        "delegation_granted.v1": DelegationGrantedPayloadV1(
            parent_agent_id="p",
            child_agent_id="c",
            node_id="n",
        ),
        "execution_failure.v1": ExecutionFailurePayloadV1(
            failure_kind="delegate_exception",
            safe_summary="test",
        ),
        "external_operation_failure.v1": ExternalOperationFailurePayloadV1(
            execution_id=mint_execution_id(),
            operation_attempt_id="attempt-1",
            provider_id="provider",
            operation_type="op",
            failure_kind="external_operation.timeout",
        ),
        "graph_node.v1": GraphNodePayloadV1(node_id="node", status="started"),
        "handoff.v1": HandoffPayloadV1(from_agent="a", to_agent="b"),
        "human.v1": HumanPayloadV1(request_id="req", option_selected="approve"),
        "interrupt.v1": InterruptPayloadV1(
            interrupt_type="test",
            blocking=False,
            recommended_action="continue",
        ),
        "llm_call.v1": LlmCallPayloadV1(),
        "skill_resolved.v1": SkillResolvedPayloadV1(
            skill_ids=("skill",),
            tool_ids=(),
            prompt_instruction_ids=(),
            policy_fragment_ids=(),
            risk_tier="low",
        ),
        "task_lifecycle.v1": TaskLifecyclePayloadV1(task_state="created"),
        "tool.v1": ToolPayloadV1(tool_name="tool.test", status="requested"),
        "validation.v1": ValidationPayloadV1(valid=True),
        "plan_lifecycle.v1": PlanLifecyclePayloadV1(plan_id="plan.test", step_count=1, task_state="planned"),
        "pause_lifecycle.v1": PauseLifecyclePayloadV1(lifecycle_state="paused"),
        "retry_lifecycle.v1": RetryLifecyclePayloadV1(scope="run", attempt=1),
        "cancellation_lifecycle.v1": CancellationLifecyclePayloadV1(reason="test"),
        "memory_access.v1": MemoryAccessPayloadV1(namespace="ns", key="k", found=True),
        "operational_alert.v1": OperationalAlertPayloadV1(alert_kind="test"),
        "human_timeout.v1": HumanTimeoutPayloadV1(request_id="req"),
        "policy_decision_spine.v1": PolicyDecisionSpinePayloadV1(evidence_id="evidence"),
        "budget_signal.v1": BudgetSignalPayloadV1(scope="agent", signal_kind="threshold"),
        "graph_backpressure.v1": GraphBackpressurePayloadV1(max_inflight_nodes=1),
        "guardrail_blocked.v1": GuardrailBlockedPayloadV1(reason="blocked"),
        "trace_persisted.v1": TracePersistedPayloadV1(trace_ref="trace"),
        "task_progress.v1": TaskProgressPayloadV1(progress_kind="test"),
    }
    payload = builders.get(schema_id)
    if payload is None:
        raise ValueError(f"no minimal test payload registered for schema_id={schema_id!r}")
    return payload


def with_preferred_canonical_payload(event: RuntimeEvent) -> RuntimeEvent:
    """Attach minimal typed envelope for spine events listed in EVENT_TYPE_PREFERRED_SCHEMA."""
    schema_id = EVENT_TYPE_PREFERRED_SCHEMA.get(event.event_type)
    if schema_id is None:
        return event
    typed = _minimal_payload_for_schema_id(schema_id)
    return runtime_event_with_payload(event, typed)


def runtime_event_test_identity(
    *,
    task_id: TaskId | str | None = None,
    run_id: RunId | str | None = None,
    attempt_id: AttemptId | str | None = None,
    execution_id: ExecutionId | str | None = None,
) -> dict[str, TaskId | RunId | AttemptId | ExecutionId]:
    return {
        "task_id": TaskId(task_id) if task_id is not None else mint_task_id(),
        "run_id": RunId(run_id) if run_id is not None else mint_run_id(),
        "attempt_id": AttemptId(attempt_id) if attempt_id is not None else mint_attempt_id(),
        "execution_id": ExecutionId(execution_id)
        if execution_id is not None
        else mint_execution_id(),
    }


def emit_context_test_identity(
    *,
    task_id: TaskId | str | None = None,
    run_id: RunId | str | None = None,
    attempt_id: AttemptId | str | None = None,
    execution_id: ExecutionId | str | None = None,
    tenant_id: str | None = None,
    correlation_id: str = "",
    parent_event_id: EventId | None = None,
    traceparent: str | None = None,
    tracestate: str | None = None,
    bus: object | None = None,
    production_mode: bool = False,
) -> "EmitContext":
    from intergrax.runtime.events.emit_context import EmitContext
    from intergrax.runtime.events.event_bus import RuntimeEventBus

    identity = runtime_event_test_identity(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    return EmitContext(
        task_id=identity["task_id"],
        run_id=identity["run_id"],
        attempt_id=identity["attempt_id"],
        execution_id=identity["execution_id"],
        tenant_id=tenant_id,
        correlation_id=correlation_id,
        parent_event_id=parent_event_id,
        traceparent=traceparent,
        tracestate=tracestate,
        bus=bus if isinstance(bus, RuntimeEventBus) else None,
        production_mode=production_mode,
    )
