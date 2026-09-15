# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Graph host Decision exposure collection and terminal publication (P0-B-D1-I1-B)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.decision_authoritative_exposure import (
    AuthoritativeDecisionExposure,
    DecisionEvaluationScope,
    ExposureUnevaluated,
    ExposureUnevaluatedReason,
)
from intergrax.contracts.decision_exposure_selection import (
    DecisionExposureCandidateAppend,
    DecisionExposurePublicationPolicy,
    DecisionExposureSelectionFailure,
    HostPublicationClass,
)
from intergrax.contracts.execution_identity import RunId, validate_run_id
from intergrax.runtime.decision_exposure_mapping import (
    decision_flow_result_to_authoritative_exposure,
    decision_flow_scope_to_evaluation_scope,
)
from intergrax.runtime.decision_flow import (
    DecisionFlowIdentitySeed,
    DecisionFlowResult,
    DecisionFlowScope,
)
from intergrax.runtime.execution.attempt_lifecycle import AttemptLifecycleService
from intergrax.runtime.execution.decision_exposure_collector import (
    DecisionExposureCandidateCollector,
)
from intergrax.runtime.execution.decision_exposure_selection_composition import (
    DecisionExposureSelectionComposition,
)
from intergrax.runtime.execution.decision_exposure_selection_validation import (
    run_validated_decision_exposure_selection,
)
from intergrax.runtime.task.task_state import TaskState, task_state_requires_authoritative_exposure


class NexusDecisionExposureError(RuntimeError):
    """Fail-closed authoritative exposure resolution failure."""


GRAPH_HOST_DECISION_EXPOSURE_PUBLICATION_POLICY = DecisionExposurePublicationPolicy(
    eligible_terminal_scopes=frozenset({DecisionEvaluationScope.GRAPH_FINAL}),
)


_NON_TERMINAL_EXPOSURE_STATES = frozenset(
    {
        TaskState.WAITING_FOR_HUMAN,
        TaskState.NEEDS_MORE_INFORMATION,
        TaskState.WAITING_FOR_RESOURCES,
        TaskState.RUNNING,
        TaskState.VALIDATING,
        TaskState.CREATED,
        TaskState.CLASSIFIED,
        TaskState.PLANNED,
    },
)


def graph_host_publication_class_for_flow_scope(
    flow_scope: DecisionFlowScope,
) -> HostPublicationClass:
    if type(flow_scope) is not DecisionFlowScope:
        raise TypeError("flow_scope must be DecisionFlowScope")
    if flow_scope is DecisionFlowScope.GRAPH_FINAL:
        return HostPublicationClass.HOST_TERMINAL_CANDIDATE
    if flow_scope is DecisionFlowScope.UAEP_STEP:
        return HostPublicationClass.INTERMEDIATE
    raise ValueError(f"unsupported DecisionFlowScope for graph host: {flow_scope!r}")


@dataclass(slots=True)
class NexusDecisionExposureRunSession:
    """Per-run collector and selection context (not checkpoint-durable)."""

    collector: DecisionExposureCandidateCollector[object]
    selection: DecisionExposureSelectionComposition
    graph_final_gate_enabled: bool
    policy: DecisionExposurePublicationPolicy = GRAPH_HOST_DECISION_EXPOSURE_PUBLICATION_POLICY
    graph_final_evaluation_occurred: bool = field(default=False)


def append_graph_decision_flow_exposure_candidate(
    session: NexusDecisionExposureRunSession,
    *,
    flow_result: DecisionFlowResult[object],
    identity_seed: DecisionFlowIdentitySeed,
) -> None:
    if type(session) is not NexusDecisionExposureRunSession:
        raise TypeError("session must be NexusDecisionExposureRunSession")
    if type(flow_result) is not DecisionFlowResult:
        raise TypeError("flow_result must be DecisionFlowResult")
    if type(identity_seed) is not DecisionFlowIdentitySeed:
        raise TypeError("identity_seed must be DecisionFlowIdentitySeed")
    if flow_result.flow_scope is DecisionFlowScope.GRAPH_FINAL:
        session.graph_final_evaluation_occurred = True
    exposure = decision_flow_result_to_authoritative_exposure(flow_result)
    if exposure is None:
        return
    fragment = DecisionExposureCandidateAppend(
        evaluation_scope=decision_flow_scope_to_evaluation_scope(flow_result.flow_scope),
        decision_scope=identity_seed.scope,
        execution_lineage=identity_seed.execution,
        host_publication_class=graph_host_publication_class_for_flow_scope(
            flow_result.flow_scope,
        ),
        exposure=exposure,
    )
    session.collector.append(fragment)


def resolve_authoritative_decision_exposure_for_task(
    *,
    task_state: TaskState,
    tenant_id: str,
    run_id: RunId | str,
    attempt_lifecycle: AttemptLifecycleService,
    session: NexusDecisionExposureRunSession | None,
) -> AuthoritativeDecisionExposure[object] | None:
    if type(task_state) is not TaskState:
        raise TypeError("task_state must be TaskState")
    if task_state in _NON_TERMINAL_EXPOSURE_STATES:
        return None
    if not task_state_requires_authoritative_exposure(task_state):
        raise NexusDecisionExposureError(
            f"unsupported task state for authoritative exposure: {task_state!r}",
        )
    resolved_run_id = validate_run_id(run_id)
    if session is None or not session.graph_final_gate_enabled:
        return ExposureUnevaluated(
            scope=None,
            reason=ExposureUnevaluatedReason.NO_DECISION_GATE,
        )
    if task_state is TaskState.CANCELLED and not session.graph_final_evaluation_occurred:
        return ExposureUnevaluated(
            scope=DecisionEvaluationScope.GRAPH_FINAL,
            reason=ExposureUnevaluatedReason.EXECUTION_CANCELLED_BEFORE_DECISION,
        )
    if task_state is TaskState.FAILED and not session.graph_final_evaluation_occurred:
        return ExposureUnevaluated(
            scope=DecisionEvaluationScope.GRAPH_FINAL,
            reason=ExposureUnevaluatedReason.EXECUTION_FAILED_BEFORE_DECISION,
        )
    effective_attempt_id = attempt_lifecycle.get_active_attempt_id(
        tenant_id=tenant_id,
        run_id=resolved_run_id,
    )
    if effective_attempt_id is None:
        raise NexusDecisionExposureError(
            "terminal authoritative exposure requires effective attempt id",
        )
    candidates = session.collector.candidates_for_attempt(effective_attempt_id)
    if not candidates and task_state in {
        TaskState.COMPLETED,
        TaskState.PARTIALLY_COMPLETED,
    }:
        return ExposureUnevaluated(
            scope=DecisionEvaluationScope.GRAPH_FINAL,
            reason=ExposureUnevaluatedReason.SCOPE_NOT_EVALUATED,
        )
    outcome = run_validated_decision_exposure_selection(
        session.selection.strategy,
        session.policy,
        candidates,
    )
    if type(outcome) is DecisionExposureSelectionFailure:
        raise NexusDecisionExposureError(
            f"authoritative decision exposure selection failed: "
            f"{outcome.reason_code.value}: {outcome.detail}",
        )
    return outcome.selected

