# © Artur Czarnecki. All rights reserved.

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
    DecisionExposureCandidate,
    DecisionExposurePublicationPolicy,
    DecisionExposureSelectionFailure,
    DecisionExposureSelectionFailureCode,
    DecisionExposureSelectionSuccessReason,
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
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
    validate_attempt_id,
)
from intergrax.runtime.execution.host_terminal_decision_exposure_selector import (
    HostTerminalDecisionExposureSelector,
)

pytestmark = pytest.mark.unit


@dataclass
class _Payload:
    business: str


def _lineage(attempt_id: AttemptId | str) -> DecisionExecutionLineage:
    return DecisionExecutionLineage(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=validate_attempt_id(attempt_id),
    )


def _accepted(payload: str) -> ExposureAccepted[_Payload]:
    artifact_kind = validate_decision_artifact_kind("agent.execution.result")
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="ns", subject="subj-a"),
        tenant_id="tenant-1",
        execution=_lineage(mint_attempt_id()),
    )
    candidate = candidate_decision(
        identity=identity,
        artifact_kind=artifact_kind,
        payload=_Payload(business=payload),
    )
    accepted = AuthoritativeAcceptedDecision(
        identity=identity,
        artifact=candidate.artifact,
        lineage=DecisionVersionLineage(current=decision_lineage_ref(identity.version)),
    )
    return ExposureAccepted(scope=DecisionEvaluationScope.GRAPH_FINAL, accepted=accepted)


def _candidate(
    *,
    evaluation_scope: DecisionEvaluationScope,
    publication_class: HostPublicationClass,
    exposure: ExposureAccepted[_Payload],
    ordinal: int,
    subject: str = "subj-a",
    attempt_id: AttemptId | None = None,
) -> DecisionExposureCandidate[_Payload]:
    lineage = _lineage(attempt_id or mint_attempt_id())
    return DecisionExposureCandidate(
        evaluation_scope=evaluation_scope,
        decision_scope=DecisionScope(namespace="ns", subject=subject),
        execution_lineage=lineage,
        host_publication_class=publication_class,
        exposure=exposure,
        evaluation_ordinal=ordinal,
    )


def _graph_policy() -> DecisionExposurePublicationPolicy:
    return DecisionExposurePublicationPolicy(
        eligible_terminal_scopes=frozenset({DecisionEvaluationScope.GRAPH_FINAL}),
    )


def _uaep_policy() -> DecisionExposurePublicationPolicy:
    return DecisionExposurePublicationPolicy(
        eligible_terminal_scopes=frozenset({DecisionEvaluationScope.UAEP_STEP}),
    )


def test_s1_graph_policy_selects_graph_candidate() -> None:
    selector = HostTerminalDecisionExposureSelector()
    exposure = _accepted("one")
    attempt = mint_attempt_id()
    candidates = (
        _candidate(
            evaluation_scope=DecisionEvaluationScope.GRAPH_FINAL,
            publication_class=HostPublicationClass.HOST_TERMINAL_CANDIDATE,
            exposure=exposure,
            ordinal=0,
            attempt_id=attempt,
        ),
    )
    outcome = selector.select(_graph_policy(), candidates)
    assert type(outcome).__name__ == "DecisionExposureSelectionDecision"
    assert outcome.selected is exposure
    assert (
        outcome.reason_code
        is DecisionExposureSelectionSuccessReason.HOST_TERMINAL_SINGLE_ELIGIBLE_CANDIDATE
    )


def test_s2_graph_policy_ignores_uaep_intermediate() -> None:
    selector = HostTerminalDecisionExposureSelector()
    graph_exposure = _accepted("graph")
    uaep_exposure = ExposureAccepted(
        scope=DecisionEvaluationScope.UAEP_STEP,
        accepted=graph_exposure.accepted,
    )
    attempt = mint_attempt_id()
    candidates = (
        _candidate(
            evaluation_scope=DecisionEvaluationScope.UAEP_STEP,
            publication_class=HostPublicationClass.INTERMEDIATE,
            exposure=uaep_exposure,
            ordinal=0,
            attempt_id=attempt,
        ),
        _candidate(
            evaluation_scope=DecisionEvaluationScope.GRAPH_FINAL,
            publication_class=HostPublicationClass.HOST_TERMINAL_CANDIDATE,
            exposure=graph_exposure,
            ordinal=1,
            attempt_id=attempt,
        ),
    )
    outcome = selector.select(_graph_policy(), candidates)
    assert outcome.selected is graph_exposure


def test_s3_uaep_only_policy_selects_uaep() -> None:
    selector = HostTerminalDecisionExposureSelector()
    uaep_exposure = ExposureAccepted(
        scope=DecisionEvaluationScope.UAEP_STEP,
        accepted=_accepted("uaep").accepted,
    )
    attempt = mint_attempt_id()
    candidates = (
        _candidate(
            evaluation_scope=DecisionEvaluationScope.UAEP_STEP,
            publication_class=HostPublicationClass.HOST_TERMINAL_CANDIDATE,
            exposure=uaep_exposure,
            ordinal=0,
            attempt_id=attempt,
        ),
    )
    outcome = selector.select(_uaep_policy(), candidates)
    assert outcome.selected is uaep_exposure


def test_s4_two_equivalent_eligible_finals_fail_closed() -> None:
    selector = HostTerminalDecisionExposureSelector()
    attempt = mint_attempt_id()
    exposure_a = _accepted("a")
    exposure_b = _accepted("b")
    candidates = (
        _candidate(
            evaluation_scope=DecisionEvaluationScope.GRAPH_FINAL,
            publication_class=HostPublicationClass.HOST_TERMINAL_CANDIDATE,
            exposure=exposure_a,
            ordinal=0,
            attempt_id=attempt,
        ),
        _candidate(
            evaluation_scope=DecisionEvaluationScope.GRAPH_FINAL,
            publication_class=HostPublicationClass.HOST_TERMINAL_CANDIDATE,
            exposure=exposure_b,
            ordinal=1,
            attempt_id=attempt,
        ),
    )
    outcome = selector.select(_graph_policy(), candidates)
    assert type(outcome) is DecisionExposureSelectionFailure
    assert outcome.reason_code is DecisionExposureSelectionFailureCode.AMBIGUOUS_TERMINAL_CANDIDATES


def test_s5_no_eligible_terminal_candidate_fails_typed() -> None:
    selector = HostTerminalDecisionExposureSelector()
    attempt = mint_attempt_id()
    unevaluated = ExposureUnevaluated(
        scope=DecisionEvaluationScope.GRAPH_FINAL,
        reason=ExposureUnevaluatedReason.SCOPE_NOT_EVALUATED,
    )
    candidates = (
        DecisionExposureCandidate(
            evaluation_scope=DecisionEvaluationScope.GRAPH_FINAL,
            decision_scope=DecisionScope(namespace="ns", subject="subj-a"),
            execution_lineage=_lineage(attempt),
            host_publication_class=HostPublicationClass.NON_PUBLISHABLE,
            exposure=unevaluated,
            evaluation_ordinal=0,
        ),
    )
    outcome = selector.select(_graph_policy(), candidates)
    assert type(outcome) is DecisionExposureSelectionFailure
    assert outcome.reason_code is DecisionExposureSelectionFailureCode.NO_ELIGIBLE_TERMINAL_CANDIDATE


def test_s6_business_payload_does_not_affect_selection() -> None:
    selector = HostTerminalDecisionExposureSelector()
    attempt = mint_attempt_id()
    exposure = _accepted("semantic-a")
    candidates = (
        _candidate(
            evaluation_scope=DecisionEvaluationScope.GRAPH_FINAL,
            publication_class=HostPublicationClass.HOST_TERMINAL_CANDIDATE,
            exposure=exposure,
            ordinal=0,
            attempt_id=attempt,
        ),
    )
    first = selector.select(_graph_policy(), candidates)
    exposure_alt = _accepted("semantic-b")
    candidates_alt = (
        _candidate(
            evaluation_scope=DecisionEvaluationScope.GRAPH_FINAL,
            publication_class=HostPublicationClass.HOST_TERMINAL_CANDIDATE,
            exposure=exposure_alt,
            ordinal=0,
            attempt_id=attempt,
        ),
    )
    second = selector.select(_graph_policy(), candidates_alt)
    assert first.reason_code == second.reason_code


def test_s7_candidate_ordering_is_deterministic() -> None:
    selector = HostTerminalDecisionExposureSelector()
    attempt = mint_attempt_id()
    exposure = _accepted("stable")
    only = _candidate(
        evaluation_scope=DecisionEvaluationScope.GRAPH_FINAL,
        publication_class=HostPublicationClass.HOST_TERMINAL_CANDIDATE,
        exposure=exposure,
        ordinal=3,
        attempt_id=attempt,
    )
    outcome_first = selector.select(_graph_policy(), (only,))
    outcome_second = selector.select(_graph_policy(), (only,))
    assert outcome_first.selected is outcome_second.selected


def test_s8_attempt_id_lexicography_does_not_affect_selection() -> None:
    selector = HostTerminalDecisionExposureSelector()
    exposure = _accepted("stable")
    attempt_low = AttemptId("attempt_00000000000000000000000000000001")
    attempt_high = AttemptId("attempt_ffffffffffffffffffffffffffffffff")
    low_candidates = (
        _candidate(
            evaluation_scope=DecisionEvaluationScope.GRAPH_FINAL,
            publication_class=HostPublicationClass.HOST_TERMINAL_CANDIDATE,
            exposure=exposure,
            ordinal=0,
            attempt_id=attempt_low,
        ),
    )
    high_candidates = (
        _candidate(
            evaluation_scope=DecisionEvaluationScope.GRAPH_FINAL,
            publication_class=HostPublicationClass.HOST_TERMINAL_CANDIDATE,
            exposure=exposure,
            ordinal=0,
            attempt_id=attempt_high,
        ),
    )
    low_outcome = selector.select(_graph_policy(), low_candidates)
    high_outcome = selector.select(_graph_policy(), high_candidates)
    assert low_outcome.selected is high_outcome.selected


def test_attempt_boundary_mixed_attempt_ids_fail_closed() -> None:
    selector = HostTerminalDecisionExposureSelector()
    exposure = _accepted("x")
    candidates = (
        _candidate(
            evaluation_scope=DecisionEvaluationScope.GRAPH_FINAL,
            publication_class=HostPublicationClass.HOST_TERMINAL_CANDIDATE,
            exposure=exposure,
            ordinal=0,
            attempt_id=mint_attempt_id(),
        ),
        _candidate(
            evaluation_scope=DecisionEvaluationScope.GRAPH_FINAL,
            publication_class=HostPublicationClass.HOST_TERMINAL_CANDIDATE,
            exposure=exposure,
            ordinal=1,
            attempt_id=mint_attempt_id(),
        ),
    )
    outcome = selector.select(_graph_policy(), candidates)
    assert type(outcome) is DecisionExposureSelectionFailure
    assert outcome.reason_code is DecisionExposureSelectionFailureCode.INVALID_CANDIDATE_SET
