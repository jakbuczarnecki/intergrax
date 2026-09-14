# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass

import pytest

from intergrax.contracts.decision_authoritative_exposure import (
    DecisionEvaluationScope,
    ExposureAccepted,
    ExposureResolution,
    ExposureUnevaluated,
    ExposureUnevaluatedReason,
    validate_decision_evaluation_scope,
    validate_exposure_unevaluated_reason,
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
from intergrax.contracts.decision_resolution import (
    AuthoritativeResolutionRecord,
    DecisionResolution,
)
from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id, mint_task_id

pytestmark = pytest.mark.unit


@dataclass
class _Payload:
    value: str


def _accepted() -> AuthoritativeAcceptedDecision[_Payload]:
    artifact_kind = validate_decision_artifact_kind("agent.execution.result")
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="ns", subject="subj"),
        tenant_id="tenant-1",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
        ),
    )
    candidate = candidate_decision(
        identity=identity,
        artifact_kind=artifact_kind,
        payload=_Payload(value="ok"),
    )
    return AuthoritativeAcceptedDecision(
        identity=identity,
        artifact=candidate.artifact,
        lineage=DecisionVersionLineage(current=decision_lineage_ref(identity.version)),
    )


def _resolution() -> AuthoritativeResolutionRecord:
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="ns", subject="subj"),
        tenant_id="tenant-1",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
        ),
    )
    return AuthoritativeResolutionRecord(
        identity=identity,
        resolution=DecisionResolution.REJECTED,
    )


def test_e1_exposure_accepted_preserves_accepted_object() -> None:
    accepted = _accepted()
    exposure = ExposureAccepted(
        scope=DecisionEvaluationScope.GRAPH_FINAL,
        accepted=accepted,
    )
    assert exposure.accepted is accepted


def test_e2_exposure_resolution_preserves_resolution_object() -> None:
    resolution = _resolution()
    exposure = ExposureResolution(
        scope=DecisionEvaluationScope.UAEP_STEP,
        resolution=resolution,
    )
    assert exposure.resolution is resolution


def test_e3_exposure_unevaluated_reason_typed() -> None:
    unevaluated = ExposureUnevaluated(
        scope=None,
        reason=ExposureUnevaluatedReason.NO_DECISION_GATE,
    )
    assert unevaluated.reason is ExposureUnevaluatedReason.NO_DECISION_GATE


def test_e4_exposure_types_are_immutable() -> None:
    accepted = ExposureAccepted(
        scope=DecisionEvaluationScope.GRAPH_FINAL,
        accepted=_accepted(),
    )
    with pytest.raises(FrozenInstanceError):
        accepted.scope = DecisionEvaluationScope.UAEP_STEP  # type: ignore[misc]


def test_e5_invalid_scope_and_reason_rejected() -> None:
    with pytest.raises(ValueError):
        validate_decision_evaluation_scope("not_a_scope")
    with pytest.raises(ValueError):
        validate_exposure_unevaluated_reason("not_a_reason")
    with pytest.raises(TypeError):
        ExposureAccepted(scope="graph_final", accepted=_accepted())  # type: ignore[arg-type]


def test_e6_manual_construction_allowed_structurally() -> None:
    accepted = ExposureAccepted(
        scope=DecisionEvaluationScope.GRAPH_FINAL,
        accepted=_accepted(),
    )
    assert type(accepted) is ExposureAccepted
