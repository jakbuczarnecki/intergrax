# © Artur Czarnecki. All rights reserved.

"""P0-B-D1-I1-B authoritative Decision exposure propagation tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.contracts.decision_authoritative_exposure import (
    DecisionEvaluationScope,
    ExposureAccepted,
    ExposureUnevaluated,
    ExposureUnevaluatedReason,
)
from intergrax.contracts.decision_exposure_selection import (
    DecisionExposureCandidateAppend,
    HostPublicationClass,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    DecisionVersionLineage,
    candidate_decision,
    decision_lineage_ref,
    validate_decision_artifact_kind,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.execution.attempt_lifecycle import (
    AttemptLifecycleService,
    InMemoryAttemptLifecycleStore,
)
from intergrax.runtime.execution.decision_exposure_collector import (
    DecisionExposureCandidateCollector,
)
from intergrax.runtime.execution.decision_exposure_selection_composition import (
    compose_decision_exposure_selection,
)
from intergrax.runtime.nexus.orchestration.nexus_decision_exposure import (
    NexusDecisionExposureError,
    NexusDecisionExposureRunSession,
    resolve_authoritative_decision_exposure_for_task,
)
from intergrax.runtime.task.task import TaskResult
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from intergrax.runtime.task.task_state import TaskState

pytestmark = pytest.mark.unit


@dataclass
class _Payload:
    text: str


def _session() -> NexusDecisionExposureRunSession:
    return NexusDecisionExposureRunSession(
        collector=DecisionExposureCandidateCollector(),
        selection=compose_decision_exposure_selection(),
        graph_final_gate_enabled=True,
    )


def _accepted_exposure(
    subject: str,
    *,
    attempt_id: AttemptId,
    run_id: RunId,
) -> ExposureAccepted[_Payload]:
    artifact_kind = validate_decision_artifact_kind("agent.execution.result")
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="ns", subject=subject),
        tenant_id="tenant-1",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=run_id,
            attempt_id=attempt_id,
        ),
    )
    candidate = candidate_decision(
        identity=identity,
        artifact_kind=artifact_kind,
        payload=_Payload(text=subject),
    )
    return ExposureAccepted(
        scope=DecisionEvaluationScope.GRAPH_FINAL,
        accepted=AuthoritativeAcceptedDecision(
            identity=candidate.identity,
            artifact=candidate.artifact,
            lineage=DecisionVersionLineage(
                current=decision_lineage_ref(candidate.identity.version),
            ),
        ),
    )


def _append_graph_final(
    session: NexusDecisionExposureRunSession,
    *,
    exposure: ExposureAccepted[_Payload],
    attempt_id: AttemptId,
    run_id: RunId,
    subject: str,
) -> None:
    session.collector.append(
        DecisionExposureCandidateAppend(
            evaluation_scope=DecisionEvaluationScope.GRAPH_FINAL,
            decision_scope=DecisionScope(namespace="graph.final", subject=subject),
            execution_lineage=DecisionExecutionLineage(
                task_id=mint_task_id(),
                run_id=run_id,
                attempt_id=attempt_id,
            ),
            host_publication_class=HostPublicationClass.HOST_TERMINAL_CANDIDATE,
            exposure=exposure,
        ),
    )


def _lifecycle_with_attempt(
    *,
    tenant_id: str,
    run_id: RunId,
    attempt_id: AttemptId,
) -> AttemptLifecycleService:
    service = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    service.record_initial_attempt(
        tenant_id=tenant_id,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    return service


def test_b1_graph_final_accepted_selected() -> None:
    session = _session()
    attempt = mint_attempt_id()
    run_id = mint_run_id()
    exposure = _accepted_exposure("ok", attempt_id=attempt, run_id=run_id)
    _append_graph_final(
        session,
        exposure=exposure,
        attempt_id=attempt,
        run_id=run_id,
        subject="g1",
    )
    lifecycle = _lifecycle_with_attempt(tenant_id="t1", run_id=run_id, attempt_id=attempt)
    resolved = resolve_authoritative_decision_exposure_for_task(
        task_state=TaskState.COMPLETED,
        tenant_id="t1",
        run_id=run_id,
        attempt_lifecycle=lifecycle,
        session=session,
    )
    assert resolved is exposure


def test_b9_terminal_without_gate_unevaluated() -> None:
    lifecycle = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    resolved = resolve_authoritative_decision_exposure_for_task(
        task_state=TaskState.COMPLETED,
        tenant_id="t1",
        run_id=mint_run_id(),
        attempt_lifecycle=lifecycle,
        session=None,
    )
    assert type(resolved) is ExposureUnevaluated
    assert resolved.reason is ExposureUnevaluatedReason.NO_DECISION_GATE


def test_b12_waiting_human_exposure_none() -> None:
    session = _session()
    resolved = resolve_authoritative_decision_exposure_for_task(
        task_state=TaskState.WAITING_FOR_HUMAN,
        tenant_id="t1",
        run_id=mint_run_id(),
        attempt_lifecycle=AttemptLifecycleService(InMemoryAttemptLifecycleStore()),
        session=session,
    )
    assert resolved is None


def test_b18_terminal_without_exposure_raises() -> None:
    with pytest.raises(ValueError, match="authoritative_decision_exposure"):
        TaskResult(
            task_id="task-1",
            state=TaskState.COMPLETED,
            authoritative_decision_exposure=None,
        )


def test_b18_terminal_with_default_helper_passes() -> None:
    result = TaskResult(
        task_id="task-1",
        state=TaskState.FAILED,
        authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
    )
    assert result.authoritative_decision_exposure is not None


def test_b13_ambiguous_candidates_fail_closed() -> None:
    session = _session()
    attempt = mint_attempt_id()
    run_id = mint_run_id()
    _append_graph_final(
        session,
        exposure=_accepted_exposure("a", attempt_id=attempt, run_id=run_id),
        attempt_id=attempt,
        run_id=run_id,
        subject="subj-a",
    )
    _append_graph_final(
        session,
        exposure=_accepted_exposure("b", attempt_id=attempt, run_id=run_id),
        attempt_id=attempt,
        run_id=run_id,
        subject="subj-b",
    )
    lifecycle = _lifecycle_with_attempt(tenant_id="t1", run_id=run_id, attempt_id=attempt)
    with pytest.raises(NexusDecisionExposureError):
        resolve_authoritative_decision_exposure_for_task(
            task_state=TaskState.COMPLETED,
            tenant_id="t1",
            run_id=run_id,
            attempt_lifecycle=lifecycle,
            session=session,
        )
