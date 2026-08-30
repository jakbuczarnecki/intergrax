# © Artur Czarnecki. All rights reserved.

from dataclasses import dataclass

import pytest

from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    DecisionVersion,
    initial_decision_version,
    mint_decision_id,
    next_decision_version,
)
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    CandidateDecision,
    DecisionArtifact,
    DecisionArtifactKind,
    DecisionVersionLineage,
    validate_decision_artifact_kind,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)


@dataclass(frozen=True, slots=True)
class IncidentDecisionPayload:
    recommendation: str


def _execution_lineage() -> DecisionExecutionLineage:
    return DecisionExecutionLineage(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _identity(
    *,
    tenant_id: str = "tenant-a",
    namespace: str = "incident",
    subject: str = "incident-123",
    version: DecisionVersion | None = None,
) -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=mint_decision_id(),
        version=version or initial_decision_version(),
        scope=DecisionScope(namespace=namespace, subject=subject),
        tenant_id=tenant_id,
        execution=_execution_lineage(),
    )


def _artifact(
    *,
    kind: str = "incident_resolution",
    recommendation: str = "escalate",
) -> DecisionArtifact[IncidentDecisionPayload]:
    return DecisionArtifact(
        kind=validate_decision_artifact_kind(kind),
        content=IncidentDecisionPayload(recommendation=recommendation),
    )


def _candidate(
    *,
    identity: DecisionIdentity | None = None,
    artifact: DecisionArtifact[IncidentDecisionPayload] | None = None,
    lineage: DecisionVersionLineage | None = None,
) -> CandidateDecision[IncidentDecisionPayload]:
    resolved_identity = identity or _identity()
    resolved_lineage = lineage or DecisionVersionLineage(resolved_identity.version)
    return CandidateDecision(
        identity=resolved_identity,
        artifact=artifact or _artifact(),
        lineage=resolved_lineage,
    )


@pytest.mark.unit
@pytest.mark.gate
def test_decision_artifact_accepts_valid_typed_payload() -> None:
    artifact = _artifact(recommendation="contain")
    assert artifact.content.recommendation == "contain"
    assert artifact.kind == "incident_resolution"


@pytest.mark.unit
@pytest.mark.gate
def test_decision_artifact_kind_valid() -> None:
    kind = validate_decision_artifact_kind("contract_review")
    assert kind == "contract_review"


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("kind", ["", "   "])
def test_decision_artifact_kind_rejects_empty(kind: str) -> None:
    with pytest.raises(ValueError):
        validate_decision_artifact_kind(kind)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("kind", [" incident_resolution", "incident_resolution "])
def test_decision_artifact_kind_rejects_whitespace(kind: str) -> None:
    with pytest.raises(ValueError):
        validate_decision_artifact_kind(kind)


@pytest.mark.unit
@pytest.mark.gate
def test_decision_artifact_is_immutable() -> None:
    artifact = _artifact()
    with pytest.raises(AttributeError):
        artifact.kind = validate_decision_artifact_kind("other")


@pytest.mark.unit
@pytest.mark.gate
def test_decision_artifact_kind_not_silently_normalized() -> None:
    with pytest.raises(ValueError):
        validate_decision_artifact_kind(" incident_resolution")


@pytest.mark.unit
@pytest.mark.gate
def test_lineage_v1_parent_none_valid() -> None:
    lineage = DecisionVersionLineage(DecisionVersion(1))
    assert lineage.current_version.value == 1
    assert lineage.parent_version is None


@pytest.mark.unit
@pytest.mark.gate
def test_lineage_v2_parent_v1_valid() -> None:
    lineage = DecisionVersionLineage(DecisionVersion(2), DecisionVersion(1))
    assert lineage.current_version.value == 2
    assert lineage.parent_version is not None
    assert lineage.parent_version.value == 1


@pytest.mark.unit
@pytest.mark.gate
def test_lineage_v3_parent_v2_valid() -> None:
    lineage = DecisionVersionLineage(DecisionVersion(3), DecisionVersion(2))
    assert lineage.current_version.value == 3
    assert lineage.parent_version is not None
    assert lineage.parent_version.value == 2


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize(
    ("current", "parent"),
    [
        (2, 2),
        (2, 3),
        (3, 5),
    ],
)
def test_lineage_rejects_current_not_after_parent(current: int, parent: int) -> None:
    with pytest.raises(ValueError):
        DecisionVersionLineage(DecisionVersion(current), DecisionVersion(parent))


@pytest.mark.unit
@pytest.mark.gate
def test_lineage_rejects_invalid_parent_type() -> None:
    with pytest.raises(TypeError):
        DecisionVersionLineage(DecisionVersion(2), 1)


@pytest.mark.unit
@pytest.mark.gate
def test_lineage_is_immutable() -> None:
    lineage = DecisionVersionLineage(DecisionVersion(1))
    with pytest.raises(AttributeError):
        lineage.current_version = DecisionVersion(2)


@pytest.mark.unit
@pytest.mark.gate
def test_candidate_valid_accepted() -> None:
    candidate = _candidate()
    assert candidate.identity.tenant_id == "tenant-a"
    assert candidate.artifact.content.recommendation == "escalate"


@pytest.mark.unit
@pytest.mark.gate
def test_candidate_requires_identity_version_matches_lineage_current() -> None:
    identity = _identity(version=DecisionVersion(3))
    lineage = DecisionVersionLineage(DecisionVersion(2), DecisionVersion(1))
    with pytest.raises(ValueError, match="current_version"):
        _candidate(identity=identity, lineage=lineage)


@pytest.mark.unit
@pytest.mark.gate
def test_candidate_is_immutable() -> None:
    candidate = _candidate()
    with pytest.raises(AttributeError):
        candidate.identity = _identity()


@pytest.mark.unit
@pytest.mark.gate
def test_candidate_preserves_artifact() -> None:
    artifact = _artifact(recommendation="isolate")
    candidate = _candidate(artifact=artifact)
    assert candidate.artifact is artifact
    assert candidate.artifact.content.recommendation == "isolate"


@pytest.mark.unit
@pytest.mark.gate
def test_candidate_equality_semantics() -> None:
    identity = _identity()
    artifact = _artifact()
    lineage = DecisionVersionLineage(identity.version)
    first = CandidateDecision(identity=identity, artifact=artifact, lineage=lineage)
    second = CandidateDecision(identity=identity, artifact=artifact, lineage=lineage)
    assert first == second

    other_version = _candidate(
        identity=_identity(version=DecisionVersion(2)),
        lineage=DecisionVersionLineage(DecisionVersion(2), DecisionVersion(1)),
    )
    other_tenant = _candidate(identity=_identity(tenant_id="tenant-b"))
    other_scope = _candidate(
        identity=_identity(namespace="routing", subject="route-1"),
    )
    other_artifact = _candidate(artifact=_artifact(recommendation="rollback"))
    assert first != other_version
    assert first != other_tenant
    assert first != other_scope
    assert first != other_artifact


@pytest.mark.unit
@pytest.mark.gate
def test_authoritative_accepted_valid_contract() -> None:
    identity = _identity()
    artifact = _artifact()
    lineage = DecisionVersionLineage(identity.version)
    accepted = AuthoritativeAcceptedDecision(
        identity=identity,
        artifact=artifact,
        lineage=lineage,
    )
    assert accepted.identity is identity
    assert accepted.artifact is artifact
    assert accepted.lineage is lineage


@pytest.mark.unit
@pytest.mark.gate
def test_authoritative_accepted_enforces_identity_lineage_consistency() -> None:
    identity = _identity(version=DecisionVersion(2))
    lineage = DecisionVersionLineage(DecisionVersion(1))
    with pytest.raises(ValueError, match="current_version"):
        AuthoritativeAcceptedDecision(
            identity=identity,
            artifact=_artifact(),
            lineage=lineage,
        )


@pytest.mark.unit
@pytest.mark.gate
def test_authoritative_accepted_is_immutable() -> None:
    identity = _identity()
    accepted = AuthoritativeAcceptedDecision(
        identity=identity,
        artifact=_artifact(),
        lineage=DecisionVersionLineage(identity.version),
    )
    with pytest.raises(AttributeError):
        accepted.artifact = _artifact(recommendation="other")


@pytest.mark.unit
@pytest.mark.gate
def test_authoritative_accepted_retains_artifact() -> None:
    artifact = _artifact(recommendation="patch")
    identity = _identity()
    accepted = AuthoritativeAcceptedDecision(
        identity=identity,
        artifact=artifact,
        lineage=DecisionVersionLineage(identity.version),
    )
    assert accepted.artifact.content.recommendation == "patch"


@pytest.mark.unit
@pytest.mark.gate
def test_authoritative_accepted_differs_by_version_tenant_scope() -> None:
    base_identity = _identity()
    base = AuthoritativeAcceptedDecision(
        identity=base_identity,
        artifact=_artifact(),
        lineage=DecisionVersionLineage(base_identity.version),
    )
    other_version = AuthoritativeAcceptedDecision(
        identity=_identity(version=DecisionVersion(2)),
        artifact=_artifact(),
        lineage=DecisionVersionLineage(DecisionVersion(2), DecisionVersion(1)),
    )
    other_tenant = AuthoritativeAcceptedDecision(
        identity=_identity(tenant_id="tenant-b"),
        artifact=_artifact(),
        lineage=DecisionVersionLineage(initial_decision_version()),
    )
    other_scope = AuthoritativeAcceptedDecision(
        identity=_identity(namespace="routing", subject="route-9"),
        artifact=_artifact(),
        lineage=DecisionVersionLineage(initial_decision_version()),
    )
    assert base != other_version
    assert base != other_tenant
    assert base != other_scope


@pytest.mark.unit
@pytest.mark.gate
def test_adversarial_identity_v2_lineage_current_v3_rejected() -> None:
    identity = _identity(version=DecisionVersion(2))
    lineage = DecisionVersionLineage(DecisionVersion(3), DecisionVersion(2))
    with pytest.raises(ValueError, match="current_version"):
        CandidateDecision(
            identity=identity,
            artifact=_artifact(),
            lineage=lineage,
        )


@pytest.mark.unit
@pytest.mark.gate
def test_adversarial_lineage_escalation_parent_v99_rejected() -> None:
    with pytest.raises(ValueError):
        DecisionVersionLineage(DecisionVersion(2), DecisionVersion(99))


@pytest.mark.unit
@pytest.mark.gate
def test_adversarial_tenant_substitution_records_not_equal() -> None:
    trusted = _candidate(identity=_identity(tenant_id="tenant-trusted"))
    substituted = _candidate(identity=_identity(tenant_id="tenant-attacker"))
    assert trusted != substituted


@pytest.mark.unit
@pytest.mark.gate
def test_adversarial_artifact_substitution_records_not_equal() -> None:
    trusted = _candidate(artifact=_artifact(recommendation="trusted"))
    substituted = _candidate(artifact=_artifact(recommendation="malicious"))
    assert trusted != substituted


@pytest.mark.unit
@pytest.mark.gate
def test_adversarial_scope_substitution_records_not_equal() -> None:
    trusted = _candidate(identity=_identity(namespace="incident", subject="incident-123"))
    substituted = _candidate(
        identity=_identity(namespace="incident", subject="incident-999"),
    )
    assert trusted != substituted
