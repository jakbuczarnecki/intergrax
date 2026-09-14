# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.contracts.decision_authoritative_exposure import (
    DecisionEvaluationScope,
    ExposureAccepted,
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
from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id, mint_task_id
from intergrax.runtime.execution.decision_exposure_collector import (
    DecisionExposureCandidateCollector,
    DecisionExposureCollectorError,
)

pytestmark = pytest.mark.unit


@dataclass
class _Payload:
    value: str


def _append_fragment(
    *,
    attempt_id: str,
    subject: str = "subj",
    scope: DecisionEvaluationScope = DecisionEvaluationScope.GRAPH_FINAL,
) -> DecisionExposureCandidateAppend[_Payload]:
    artifact_kind = validate_decision_artifact_kind("agent.execution.result")
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="ns", subject=subject),
        tenant_id="tenant-1",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=attempt_id,
        ),
    )
    candidate = candidate_decision(
        identity=identity,
        artifact_kind=artifact_kind,
        payload=_Payload(value="v"),
    )
    accepted = AuthoritativeAcceptedDecision(
        identity=identity,
        artifact=candidate.artifact,
        lineage=DecisionVersionLineage(current=decision_lineage_ref(identity.version)),
    )
    exposure = ExposureAccepted(scope=scope, accepted=accepted)
    return DecisionExposureCandidateAppend(
        evaluation_scope=scope,
        decision_scope=identity.scope,
        execution_lineage=identity.execution,
        host_publication_class=HostPublicationClass.HOST_TERMINAL_CANDIDATE,
        exposure=exposure,
    )


def test_c1_append_candidate() -> None:
    collector = DecisionExposureCandidateCollector[_Payload]()
    attempt = mint_attempt_id()
    collector.append(_append_fragment(attempt_id=attempt))
    assert collector.count(attempt) == 1


def test_c2_partition_by_attempt_id() -> None:
    collector = DecisionExposureCandidateCollector[_Payload]()
    attempt_a = mint_attempt_id()
    attempt_b = mint_attempt_id()
    collector.append(_append_fragment(attempt_id=attempt_a, subject="a"))
    collector.append(_append_fragment(attempt_id=attempt_b, subject="b"))
    assert collector.count(attempt_a) == 1
    assert collector.count(attempt_b) == 1


def test_c3_returns_immutable_snapshot() -> None:
    collector = DecisionExposureCandidateCollector[_Payload]()
    attempt = mint_attempt_id()
    collector.append(_append_fragment(attempt_id=attempt))
    snapshot = collector.candidates_for_attempt(attempt)
    collector.append(_append_fragment(attempt_id=attempt, subject="other"))
    assert len(snapshot) == 1
    assert collector.count(attempt) == 2


def test_c4_old_attempt_partition_does_not_mix_with_new() -> None:
    collector = DecisionExposureCandidateCollector[_Payload]()
    old_attempt = mint_attempt_id()
    new_attempt = mint_attempt_id()
    collector.append(_append_fragment(attempt_id=old_attempt))
    collector.append(_append_fragment(attempt_id=new_attempt))
    assert collector.candidates_for_attempt(old_attempt) != collector.candidates_for_attempt(
        new_attempt,
    )


def test_c5_duplicate_scope_subject_rejected() -> None:
    collector = DecisionExposureCandidateCollector[_Payload]()
    attempt = mint_attempt_id()
    fragment = _append_fragment(attempt_id=attempt)
    collector.append(fragment)
    with pytest.raises(DecisionExposureCollectorError):
        collector.append(fragment)


def test_c6_ordinals_are_monotonic_per_attempt() -> None:
    collector = DecisionExposureCandidateCollector[_Payload]()
    attempt = mint_attempt_id()
    collector.append(_append_fragment(attempt_id=attempt, subject="one"))
    collector.append(_append_fragment(attempt_id=attempt, subject="two"))
    ordinals = tuple(
        item.evaluation_ordinal for item in collector.candidates_for_attempt(attempt)
    )
    assert ordinals == (0, 1)


def test_c7_attempt_id_ordering_irrelevant_to_partitions() -> None:
    collector = DecisionExposureCandidateCollector[_Payload]()
    low = "attempt_00000000000000000000000000000001"
    high = "attempt_ffffffffffffffffffffffffffffffff"
    collector.append(_append_fragment(attempt_id=high))
    collector.append(_append_fragment(attempt_id=low))
    assert collector.count(low) == 1
    assert collector.count(high) == 1
