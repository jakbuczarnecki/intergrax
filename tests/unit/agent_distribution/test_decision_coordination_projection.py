# © Artur Czarnecki. All rights reserved.

"""NPSC-5C/R2 — Decision coordination projection adapter tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.agent_distribution.capability_matching import CapabilityId
from intergrax.agent_distribution.coordination_intent import (
    CoordinationContribution,
    CoordinationContributionId,
    CoordinationExecutionMode,
    CoordinationIntent,
    CoordinationIntentId,
    validate_coordination_intent,
)
from intergrax.agent_distribution.decision_coordination_projection import (
    DecisionCoordinationProjectionContext,
    DecisionCoordinationProjectionError,
    coordination_intent_id_from_decision_identity,
    project_authoritative_accepted_decision_coordination,
    project_decision_coordination_artifact,
)
from intergrax.agent_distribution.task_capability_resolution import (
    AgentDistributionCapabilityNeedKind,
)
from intergrax.contracts.decision_coordination import (
    DecisionCapabilityRequirement,
    DecisionCoordinationContribution,
    DecisionCoordinationSemantic,
    DecisionCoordinationShape,
    decision_coordination_artifact,
    decision_coordination_artifact_kind,
    validate_decision_capability_id,
    validate_decision_contribution_id,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionId,
    DecisionIdentity,
    DecisionScope,
    DecisionVersion,
    initial_decision_version,
    mint_decision_id,
    next_decision_version,
)
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    DecisionArtifact,
    decision_lineage_ref,
    decision_version_lineage,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.decision_record import validate_decision_artifact_kind

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@dataclass(frozen=True, slots=True)
class ExamplePayload:
    detail: str


def _identity(
    *,
    decision_id: DecisionId | None = None,
    version: DecisionVersion | None = None,
) -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=decision_id or mint_decision_id(),
        version=version or initial_decision_version(),
        scope=DecisionScope(namespace="coordination", subject="case-1"),
        tenant_id="tenant-a",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )


def _capability(capability_id: str) -> DecisionCapabilityRequirement:
    return DecisionCapabilityRequirement(
        capability_id=validate_decision_capability_id(capability_id),
    )


def _contribution(
    contribution_id: str,
    *,
    capability_id: str = "invoice_ocr",
    payload: ExamplePayload | None = None,
) -> DecisionCoordinationContribution[ExamplePayload]:
    return DecisionCoordinationContribution(
        contribution_id=validate_decision_contribution_id(contribution_id),
        capability_requirement=_capability(capability_id),
        payload=payload or ExamplePayload(detail=contribution_id),
    )


def _semantic(
    shape: DecisionCoordinationShape,
    contributions: tuple[DecisionCoordinationContribution[ExamplePayload], ...],
) -> DecisionCoordinationSemantic[ExamplePayload]:
    return DecisionCoordinationSemantic(shape=shape, contributions=contributions)


def _artifact(
    semantic: DecisionCoordinationSemantic[ExamplePayload],
) -> DecisionArtifact[DecisionCoordinationSemantic[ExamplePayload]]:
    return decision_coordination_artifact(semantic)


def _validate_projected_intent(intent: CoordinationIntent[ExamplePayload]) -> None:
    object_contributions: tuple[CoordinationContribution[object], ...] = tuple(
        CoordinationContribution(
            contribution_id=contribution.contribution_id,
            payload=contribution.payload,
            capability_need=contribution.capability_need,
            policy=contribution.policy,
        )
        for contribution in intent.contributions
    )
    validate_coordination_intent(
        CoordinationIntent(
            intent_id=intent.intent_id,
            mode=intent.mode,
            contributions=object_contributions,
            requested_max_concurrency=intent.requested_max_concurrency,
        ),
    )


def _accepted(
    semantic: DecisionCoordinationSemantic[ExamplePayload],
    *,
    identity: DecisionIdentity | None = None,
) -> AuthoritativeAcceptedDecision[DecisionCoordinationSemantic[ExamplePayload]]:
    resolved_identity = identity or _identity()
    return AuthoritativeAcceptedDecision(
        identity=resolved_identity,
        artifact=_artifact(semantic),
        lineage=decision_version_lineage(
            current=decision_lineage_ref(resolved_identity.version),
        ),
    )


def test_valid_single_projection() -> None:
    semantic = _semantic(DecisionCoordinationShape.SINGLE, (_contribution("contrib-a"),))
    accepted = _accepted(semantic)
    intent = project_authoritative_accepted_decision_coordination(accepted)

    assert intent.mode is CoordinationExecutionMode.SINGLE
    assert len(intent.contributions) == 1
    assert intent.requested_max_concurrency is None
    assert intent.contributions[0].contribution_id == CoordinationContributionId("contrib-a")
    assert intent.contributions[0].payload.detail == "contrib-a"
    assert (
        intent.contributions[0].capability_need.kind
        is AgentDistributionCapabilityNeedKind.RESOLVED_REQUIREMENT
    )
    _validate_projected_intent(intent)


def test_valid_fan_out_projection_preserves_order() -> None:
    semantic = _semantic(
        DecisionCoordinationShape.FAN_OUT,
        (
            _contribution("contrib-c", capability_id="invoice_ocr"),
            _contribution("contrib-a", capability_id="fraud_analysis"),
            _contribution("contrib-b", capability_id="catalog_lookup"),
        ),
    )
    accepted = _accepted(semantic)
    intent = project_authoritative_accepted_decision_coordination(accepted)

    assert intent.mode is CoordinationExecutionMode.FAN_OUT
    assert [item.contribution_id for item in intent.contributions] == [
        CoordinationContributionId("contrib-c"),
        CoordinationContributionId("contrib-a"),
        CoordinationContributionId("contrib-b"),
    ]
    _validate_projected_intent(intent)


def test_wrong_artifact_kind_fails_closed() -> None:
    semantic = _semantic(DecisionCoordinationShape.SINGLE, (_contribution("contrib-a"),))
    wrong_kind_artifact = DecisionArtifact(
        kind=validate_decision_artifact_kind("decision.other"),
        content=semantic,
    )
    context = DecisionCoordinationProjectionContext(
        coordination_intent_id=CoordinationIntentId("intent-explicit"),
    )
    with pytest.raises(
        DecisionCoordinationProjectionError,
        match="artifact kind must be 'decision.coordination'",
    ):
        project_decision_coordination_artifact(wrong_kind_artifact, context)


def test_contribution_identity_is_literal_projection() -> None:
    semantic = _semantic(
        DecisionCoordinationShape.FAN_OUT,
        (
            _contribution("contrib-a"),
            _contribution("contrib-b"),
        ),
    )
    intent = project_authoritative_accepted_decision_coordination(_accepted(semantic))
    assert intent.contributions[0].contribution_id == CoordinationContributionId("contrib-a")
    assert intent.contributions[1].contribution_id == CoordinationContributionId("contrib-b")


def test_payload_preserved_by_identity() -> None:
    payload = ExamplePayload(detail="typed-value")
    semantic = _semantic(
        DecisionCoordinationShape.SINGLE,
        (_contribution("contrib-a", payload=payload),),
    )
    intent = project_authoritative_accepted_decision_coordination(_accepted(semantic))
    assert intent.contributions[0].payload == payload
    assert intent.contributions[0].payload is payload


def test_capability_projects_to_canonical_required_requirement() -> None:
    semantic = _semantic(
        DecisionCoordinationShape.SINGLE,
        (_contribution("contrib-a", capability_id="invoice_ocr"),),
    )
    intent = project_authoritative_accepted_decision_coordination(_accepted(semantic))
    requirement = intent.contributions[0].capability_need.resolved_requirement
    assert requirement is not None
    required_ids = {item.value for item in requirement.required_capability_ids}
    assert required_ids == {"invoice_ocr"}
    assert CapabilityId(value="invoice_ocr") in requirement.required_capability_ids


def test_intent_id_is_deterministic_from_decision_identity() -> None:
    identity = _identity()
    semantic = _semantic(DecisionCoordinationShape.SINGLE, (_contribution("contrib-a"),))
    first = project_authoritative_accepted_decision_coordination(
        _accepted(semantic, identity=identity),
    )
    second = project_authoritative_accepted_decision_coordination(
        _accepted(semantic, identity=identity),
    )
    expected = coordination_intent_id_from_decision_identity(identity)
    assert first.intent_id == expected
    assert second.intent_id == expected


def test_intent_id_differs_for_different_decision_identity() -> None:
    semantic = _semantic(DecisionCoordinationShape.SINGLE, (_contribution("contrib-a"),))
    first = project_authoritative_accepted_decision_coordination(_accepted(semantic))
    second = project_authoritative_accepted_decision_coordination(_accepted(semantic))
    assert first.intent_id != second.intent_id


def test_intent_id_differs_for_different_decision_version() -> None:
    decision_id = mint_decision_id()
    identity_v1 = _identity(decision_id=decision_id, version=initial_decision_version())
    identity_v2 = DecisionIdentity(
        decision_id=decision_id,
        version=next_decision_version(identity_v1.version),
        scope=identity_v1.scope,
        tenant_id=identity_v1.tenant_id,
        execution=identity_v1.execution,
    )
    semantic = _semantic(DecisionCoordinationShape.SINGLE, (_contribution("contrib-a"),))
    intent_v1 = project_authoritative_accepted_decision_coordination(
        _accepted(semantic, identity=identity_v1),
    )
    intent_v2 = project_authoritative_accepted_decision_coordination(
        AuthoritativeAcceptedDecision(
            identity=identity_v2,
            artifact=_artifact(semantic),
            lineage=decision_version_lineage(
                current=decision_lineage_ref(identity_v2.version),
                parents=(decision_lineage_ref(identity_v1.version),),
            ),
        ),
    )
    assert intent_v1.intent_id != intent_v2.intent_id


def test_explicit_projection_context_intent_id() -> None:
    semantic = _semantic(DecisionCoordinationShape.SINGLE, (_contribution("contrib-a"),))
    artifact = _artifact(semantic)
    context = DecisionCoordinationProjectionContext(
        coordination_intent_id=CoordinationIntentId("intent-explicit"),
    )
    intent = project_decision_coordination_artifact(artifact, context)
    assert intent.intent_id == CoordinationIntentId("intent-explicit")
    _validate_projected_intent(intent)


def test_cross_system_contract_projection_proof() -> None:
    semantic = _semantic(
        DecisionCoordinationShape.FAN_OUT,
        (
            _contribution("contrib-a", capability_id="invoice_ocr"),
            _contribution("contrib-b", capability_id="fraud_analysis"),
        ),
    )
    accepted = _accepted(semantic)
    intent = project_authoritative_accepted_decision_coordination(accepted)
    _validate_projected_intent(intent)
    assert accepted.artifact.kind == decision_coordination_artifact_kind()
    assert intent.mode is CoordinationExecutionMode.FAN_OUT
