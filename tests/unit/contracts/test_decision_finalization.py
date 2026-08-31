# © Artur Czarnecki. All rights reserved.

from pathlib import Path

from dataclasses import FrozenInstanceError, dataclass, fields
from typing import get_type_hints

import pytest

from intergrax.contracts.decision_finalization import (
    DecisionFinalizationConflictError,
    DecisionFinalizationKey,
    DecisionFinalizeDisposition,
    DecisionFinalizeGuardResult,
    DecisionFinalizeGuardState,
    decision_finalization_key,
    guard_decision_finalization,
    initial_decision_finalize_guard,
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
    DecisionVersionLineage,
    validate_decision_artifact_kind,
)
from intergrax.contracts.decision_resolution import (
    AuthoritativeResolutionRecord,
    DecisionResolution,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)

_FORBIDDEN_RUNTIME_FIELD_NAMES = frozenset(
    {
        "retry_count",
        "lock",
        "transaction",
        "checkpoint",
        "execution_status",
        "policy_status",
    },
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


def _alternate_execution_lineage() -> DecisionExecutionLineage:
    return DecisionExecutionLineage(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _identity(
    *,
    decision_id: DecisionId | None = None,
    tenant_id: str = "tenant-a",
    namespace: str = "incident",
    subject: str = "incident-123",
    version: DecisionVersion | None = None,
    execution: DecisionExecutionLineage | None = None,
) -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=mint_decision_id() if decision_id is None else decision_id,
        version=version or initial_decision_version(),
        scope=DecisionScope(namespace=namespace, subject=subject),
        tenant_id=tenant_id,
        execution=execution or _execution_lineage(),
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


def _accepted(
    *,
    identity: DecisionIdentity | None = None,
    artifact: DecisionArtifact[IncidentDecisionPayload] | None = None,
    lineage: DecisionVersionLineage | None = None,
) -> AuthoritativeAcceptedDecision[IncidentDecisionPayload]:
    resolved_identity = identity or _identity()
    resolved_lineage = lineage or DecisionVersionLineage(resolved_identity.version)
    return AuthoritativeAcceptedDecision(
        identity=resolved_identity,
        artifact=artifact or _artifact(),
        lineage=resolved_lineage,
    )


def _resolution(
    *,
    identity: DecisionIdentity | None = None,
    resolution: DecisionResolution = DecisionResolution.REJECTED,
) -> AuthoritativeResolutionRecord:
    return AuthoritativeResolutionRecord(
        identity=identity or _identity(),
        resolution=resolution,
    )


def _guard_for_identity(identity: DecisionIdentity) -> DecisionFinalizeGuardState[IncidentDecisionPayload]:
    return initial_decision_finalize_guard(decision_finalization_key(identity))


@pytest.mark.unit
@pytest.mark.gate
def test_decision_finalization_module_does_not_import_private_decision_identity_names() -> None:
    source = Path("intergrax/contracts/decision_finalization.py").read_text(encoding="utf-8")
    assert "_validate_tenant_id" not in source
    assert "from intergrax.contracts.decision_identity import" in source


@pytest.mark.unit
@pytest.mark.gate
def test_decision_finalization_key_ignores_version_and_execution_lineage() -> None:
    fixed_id = mint_decision_id()
    scope = DecisionScope(namespace="incident", subject="incident-123")
    tenant_id = "tenant-a"
    execution_a = _execution_lineage()
    execution_b = _alternate_execution_lineage()

    identity_v1 = _identity(
        decision_id=fixed_id,
        tenant_id=tenant_id,
        namespace=scope.namespace,
        subject=scope.subject,
        version=initial_decision_version(),
        execution=execution_a,
    )
    identity_v2 = _identity(
        decision_id=fixed_id,
        tenant_id=tenant_id,
        namespace=scope.namespace,
        subject=scope.subject,
        version=next_decision_version(initial_decision_version()),
        execution=execution_b,
    )

    key_v1 = decision_finalization_key(identity_v1)
    key_v2 = decision_finalization_key(identity_v2)

    assert key_v1 == key_v2
    assert key_v1.decision_id == fixed_id
    assert key_v1.scope == scope
    assert key_v1.tenant_id == tenant_id
    assert identity_v1.version != identity_v2.version
    assert identity_v1.execution != identity_v2.execution


@pytest.mark.unit
@pytest.mark.gate
def test_decision_finalization_key_differs_by_decision_id() -> None:
    base = _identity()
    other = _identity(decision_id=mint_decision_id())
    assert decision_finalization_key(base) != decision_finalization_key(other)


@pytest.mark.unit
@pytest.mark.gate
def test_decision_finalization_key_differs_by_tenant() -> None:
    base = _identity()
    other = _identity(tenant_id="tenant-b")
    assert decision_finalization_key(base) != decision_finalization_key(other)


@pytest.mark.unit
@pytest.mark.gate
def test_decision_finalization_key_differs_by_namespace() -> None:
    base = _identity()
    other = _identity(namespace="policy")
    assert decision_finalization_key(base) != decision_finalization_key(other)


@pytest.mark.unit
@pytest.mark.gate
def test_decision_finalization_key_differs_by_subject() -> None:
    base = _identity()
    other = _identity(subject="incident-999")
    assert decision_finalization_key(base) != decision_finalization_key(other)


@pytest.mark.unit
@pytest.mark.gate
def test_first_accepted_finalization() -> None:
    accepted = _accepted()
    guard = _guard_for_identity(accepted.identity)

    result = guard_decision_finalization(guard, accepted)

    assert result.disposition is DecisionFinalizeDisposition.FIRST_FINALIZATION
    assert result.state.authoritative_outcome == accepted
    assert result.state.authoritative_outcome is accepted


@pytest.mark.unit
@pytest.mark.gate
def test_first_rejected_finalization() -> None:
    rejected = _resolution(resolution=DecisionResolution.REJECTED)
    guard = _guard_for_identity(rejected.identity)

    result = guard_decision_finalization(guard, rejected)

    assert result.disposition is DecisionFinalizeDisposition.FIRST_FINALIZATION
    assert result.state.authoritative_outcome == rejected


@pytest.mark.unit
@pytest.mark.gate
def test_first_unresolved_finalization() -> None:
    unresolved = _resolution(resolution=DecisionResolution.UNRESOLVED)
    guard = _guard_for_identity(unresolved.identity)

    result = guard_decision_finalization(guard, unresolved)

    assert result.disposition is DecisionFinalizeDisposition.FIRST_FINALIZATION
    assert result.state.authoritative_outcome == unresolved


@pytest.mark.unit
@pytest.mark.gate
def test_idempotent_accepted_replay() -> None:
    accepted = _accepted()
    guard = _guard_for_identity(accepted.identity)
    first = guard_decision_finalization(guard, accepted)

    replay = guard_decision_finalization(first.state, accepted)

    assert replay.disposition is DecisionFinalizeDisposition.IDEMPOTENT_REPLAY
    assert replay.state is first.state
    assert replay.state.authoritative_outcome is accepted


@pytest.mark.unit
@pytest.mark.gate
def test_idempotent_rejected_replay() -> None:
    rejected = _resolution(resolution=DecisionResolution.REJECTED)
    guard = _guard_for_identity(rejected.identity)
    first = guard_decision_finalization(guard, rejected)

    replay = guard_decision_finalization(first.state, rejected)

    assert replay.disposition is DecisionFinalizeDisposition.IDEMPOTENT_REPLAY
    assert replay.state is first.state


@pytest.mark.unit
@pytest.mark.gate
def test_idempotent_unresolved_replay() -> None:
    unresolved = _resolution(resolution=DecisionResolution.UNRESOLVED)
    guard = _guard_for_identity(unresolved.identity)
    first = guard_decision_finalization(guard, unresolved)

    replay = guard_decision_finalization(first.state, unresolved)

    assert replay.disposition is DecisionFinalizeDisposition.IDEMPOTENT_REPLAY
    assert replay.state is first.state


@pytest.mark.unit
@pytest.mark.gate
def test_accepted_version_conflict() -> None:
    fixed_id = mint_decision_id()
    scope = DecisionScope(namespace="incident", subject="incident-123")
    identity_v1 = _identity(
        decision_id=fixed_id,
        namespace=scope.namespace,
        subject=scope.subject,
        version=initial_decision_version(),
    )
    identity_v2 = _identity(
        decision_id=fixed_id,
        namespace=scope.namespace,
        subject=scope.subject,
        version=next_decision_version(initial_decision_version()),
    )
    accepted_v1 = _accepted(
        identity=identity_v1,
        lineage=DecisionVersionLineage(identity_v1.version),
    )
    accepted_v2 = _accepted(
        identity=identity_v2,
        lineage=DecisionVersionLineage(
            identity_v2.version,
            identity_v1.version,
        ),
    )
    guard = _guard_for_identity(identity_v1)
    finalized = guard_decision_finalization(guard, accepted_v1)

    with pytest.raises(DecisionFinalizationConflictError, match="decision_id="):
        guard_decision_finalization(finalized.state, accepted_v2)


@pytest.mark.unit
@pytest.mark.gate
def test_accepted_payload_conflict() -> None:
    identity = _identity()
    accepted_a = _accepted(
        identity=identity,
        artifact=_artifact(recommendation="contain"),
        lineage=DecisionVersionLineage(identity.version),
    )
    accepted_b = _accepted(
        identity=identity,
        artifact=_artifact(recommendation="rollback"),
        lineage=DecisionVersionLineage(identity.version),
    )
    guard = _guard_for_identity(identity)
    finalized = guard_decision_finalization(guard, accepted_a)

    with pytest.raises(DecisionFinalizationConflictError):
        guard_decision_finalization(finalized.state, accepted_b)


@pytest.mark.unit
@pytest.mark.gate
def test_accepted_vs_rejected_conflict() -> None:
    identity = _identity()
    accepted = _accepted(identity=identity)
    rejected = _resolution(identity=identity, resolution=DecisionResolution.REJECTED)
    guard = _guard_for_identity(identity)
    finalized = guard_decision_finalization(guard, accepted)

    with pytest.raises(DecisionFinalizationConflictError):
        guard_decision_finalization(finalized.state, rejected)


@pytest.mark.unit
@pytest.mark.gate
def test_rejected_vs_unresolved_conflict() -> None:
    identity = _identity()
    rejected = _resolution(identity=identity, resolution=DecisionResolution.REJECTED)
    unresolved = _resolution(identity=identity, resolution=DecisionResolution.UNRESOLVED)
    guard = _guard_for_identity(identity)
    finalized = guard_decision_finalization(guard, rejected)

    with pytest.raises(DecisionFinalizationConflictError):
        guard_decision_finalization(finalized.state, unresolved)


@pytest.mark.unit
@pytest.mark.gate
def test_same_resolution_different_record_context_conflict() -> None:
    fixed_id = mint_decision_id()
    scope = DecisionScope(namespace="incident", subject="incident-123")
    identity_v1 = _identity(
        decision_id=fixed_id,
        namespace=scope.namespace,
        subject=scope.subject,
        version=initial_decision_version(),
    )
    identity_v2 = _identity(
        decision_id=fixed_id,
        namespace=scope.namespace,
        subject=scope.subject,
        version=next_decision_version(initial_decision_version()),
    )
    rejected_v1 = _resolution(identity=identity_v1, resolution=DecisionResolution.REJECTED)
    rejected_v2 = _resolution(identity=identity_v2, resolution=DecisionResolution.REJECTED)
    guard = _guard_for_identity(identity_v1)
    finalized = guard_decision_finalization(guard, rejected_v1)

    with pytest.raises(DecisionFinalizationConflictError):
        guard_decision_finalization(finalized.state, rejected_v2)


@pytest.mark.unit
@pytest.mark.gate
def test_cross_scope_tenant_mismatch_fails_fast() -> None:
    identity_a = _identity(tenant_id="tenant-a")
    identity_b = _identity(tenant_id="tenant-b")
    guard = _guard_for_identity(identity_a)
    outcome_b = _accepted(identity=identity_b)

    with pytest.raises(ValueError, match="does not match guard state key"):
        guard_decision_finalization(guard, outcome_b)


@pytest.mark.unit
@pytest.mark.gate
def test_cross_scope_decision_id_mismatch_fails_fast() -> None:
    identity_a = _identity()
    identity_b = _identity(decision_id=mint_decision_id())
    guard = _guard_for_identity(identity_a)
    outcome_b = _accepted(identity=identity_b)

    with pytest.raises(ValueError, match="does not match guard state key"):
        guard_decision_finalization(guard, outcome_b)


@pytest.mark.unit
@pytest.mark.gate
def test_cross_scope_namespace_mismatch_fails_fast() -> None:
    identity_a = _identity(namespace="incident")
    identity_b = _identity(namespace="policy")
    guard = _guard_for_identity(identity_a)
    outcome_b = _accepted(identity=identity_b)

    with pytest.raises(ValueError, match="does not match guard state key"):
        guard_decision_finalization(guard, outcome_b)


@pytest.mark.unit
@pytest.mark.gate
def test_reconstructed_outcome_with_new_execution_lineage_is_not_idempotent() -> None:
    """Retry after crash must replay the persisted record, not a reconstruction."""
    fixed_id = mint_decision_id()
    scope = DecisionScope(namespace="incident", subject="incident-123")
    original_identity = _identity(
        decision_id=fixed_id,
        namespace=scope.namespace,
        subject=scope.subject,
        execution=_execution_lineage(),
    )
    accepted_original = _accepted(identity=original_identity)
    guard = _guard_for_identity(original_identity)
    finalized = guard_decision_finalization(guard, accepted_original)

    reconstructed_identity = _identity(
        decision_id=fixed_id,
        namespace=scope.namespace,
        subject=scope.subject,
        execution=_alternate_execution_lineage(),
    )
    accepted_reconstructed = _accepted(
        identity=reconstructed_identity,
        artifact=accepted_original.artifact,
        lineage=accepted_original.lineage,
    )
    assert accepted_original != accepted_reconstructed

    with pytest.raises(DecisionFinalizationConflictError):
        guard_decision_finalization(finalized.state, accepted_reconstructed)


@pytest.mark.unit
@pytest.mark.gate
def test_finalization_key_is_immutable() -> None:
    key = decision_finalization_key(_identity())
    with pytest.raises(FrozenInstanceError):
        setattr(key, "tenant_id", "tenant-b")


@pytest.mark.unit
@pytest.mark.gate
def test_guard_state_is_immutable() -> None:
    guard = _guard_for_identity(_identity())
    with pytest.raises(FrozenInstanceError):
        setattr(guard, "authoritative_outcome", _accepted())


@pytest.mark.unit
@pytest.mark.gate
def test_guard_result_is_immutable() -> None:
    accepted = _accepted()
    guard = _guard_for_identity(accepted.identity)
    result = guard_decision_finalization(guard, accepted)
    with pytest.raises(FrozenInstanceError):
        setattr(result, "disposition", DecisionFinalizeDisposition.IDEMPOTENT_REPLAY)


@pytest.mark.unit
@pytest.mark.gate
def test_guard_contracts_have_no_runtime_storage_fields() -> None:
    contract_types = (
        DecisionFinalizationKey,
        DecisionFinalizeGuardState,
        DecisionFinalizeGuardResult,
    )
    for contract_type in contract_types:
        field_names = frozenset(field.name for field in fields(contract_type))
        assert field_names.isdisjoint(_FORBIDDEN_RUNTIME_FIELD_NAMES)


@pytest.mark.unit
@pytest.mark.gate
def test_decision_finalization_key_field_types() -> None:
    hints = get_type_hints(DecisionFinalizationKey)
    assert hints["decision_id"].__name__ == "DecisionId"
    assert hints["scope"] is DecisionScope
    assert hints["tenant_id"] is str


@pytest.mark.unit
@pytest.mark.gate
def test_decision_finalize_guard_state_outcome_union_typing() -> None:
    hints = get_type_hints(DecisionFinalizeGuardState)
    outcome_hint = hints["authoritative_outcome"]
    assert "AuthoritativeAcceptedDecision" in str(outcome_hint)
    assert "AuthoritativeResolutionRecord" in str(outcome_hint)
    assert "None" in str(outcome_hint)
