# © Artur Czarnecki. All rights reserved.

from dataclasses import FrozenInstanceError, dataclass

import pytest

from intergrax.contracts.decision_checkpoint import (
    DecisionCheckpointState,
    decision_checkpoint_state,
    restore_decision_checkpoint_state,
)
from intergrax.contracts.decision_finalization import (
    DecisionFinalizationConflictError,
    DecisionFinalizeDisposition,
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
from intergrax.contracts.decision_lifecycle import (
    DecisionLifecycleStage,
    DecisionLifecycleState,
    initial_decision_lifecycle_state,
    transition_decision_lifecycle,
)
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    DecisionArtifact,
    DecisionVersionLineage,
    decision_lineage_ref,
    decision_version_lineage,
    validate_decision_artifact_kind,
    validate_decision_branch_id,
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
from intergrax.runtime.execution.decision_checkpoint_persistence import (
    DecisionCheckpointPersistence,
    load_decision_checkpoint,
    save_decision_checkpoint,
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


def _root_lineage() -> DecisionVersionLineage:
    return DecisionVersionLineage(current=decision_lineage_ref(initial_decision_version()))


def _linear_lineage(
    current: DecisionVersion,
    parent_version: DecisionVersion,
) -> DecisionVersionLineage:
    return DecisionVersionLineage(
        current=decision_lineage_ref(current),
        parents=(decision_lineage_ref(parent_version),),
    )


def _accepted(
    *,
    identity: DecisionIdentity | None = None,
    artifact: DecisionArtifact[IncidentDecisionPayload] | None = None,
    lineage: DecisionVersionLineage | None = None,
) -> AuthoritativeAcceptedDecision[IncidentDecisionPayload]:
    resolved_identity = identity or _identity()
    resolved_lineage = lineage or DecisionVersionLineage(
        current=decision_lineage_ref(resolved_identity.version),
    )
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


def _lifecycle(
    identity: DecisionIdentity,
    *,
    stage: DecisionLifecycleStage = DecisionLifecycleStage.PROPOSAL,
    transition_index: int = 0,
) -> DecisionLifecycleState:
    state = initial_decision_lifecycle_state(identity)
    if stage is DecisionLifecycleStage.PROPOSAL and transition_index == 0:
        return state
    current = state
    if stage is not DecisionLifecycleStage.PROPOSAL:
        path = _path_to_stage(stage)
        for next_stage in path:
            current = transition_decision_lifecycle(current, next_stage)
    while current.transition_index < transition_index:
        current = DecisionLifecycleState(
            identity=current.identity,
            stage=current.stage,
            transition_index=current.transition_index + 1,
        )
    return current


def _path_to_stage(stage: DecisionLifecycleStage) -> tuple[DecisionLifecycleStage, ...]:
    if stage is DecisionLifecycleStage.REVISION:
        return (
            DecisionLifecycleStage.VERIFICATION,
            DecisionLifecycleStage.REVISION,
        )
    if stage is DecisionLifecycleStage.FINALIZATION:
        return (
            DecisionLifecycleStage.VERIFICATION,
            DecisionLifecycleStage.RESOLUTION,
            DecisionLifecycleStage.FINALIZATION,
        )
    if stage is DecisionLifecycleStage.TERMINAL:
        return (
            DecisionLifecycleStage.VERIFICATION,
            DecisionLifecycleStage.RESOLUTION,
            DecisionLifecycleStage.FINALIZATION,
            DecisionLifecycleStage.TERMINAL,
        )
    if stage is DecisionLifecycleStage.VERIFICATION:
        return (DecisionLifecycleStage.VERIFICATION,)
    raise ValueError(f"unsupported direct path for stage {stage.value!r}")


def _guard_for_identity(
    identity: DecisionIdentity,
    *,
    outcome: (
        AuthoritativeAcceptedDecision[IncidentDecisionPayload]
        | AuthoritativeResolutionRecord
        | None
    ) = None,
) -> DecisionFinalizeGuardState[IncidentDecisionPayload]:
    guard = initial_decision_finalize_guard(decision_finalization_key(identity))
    if outcome is None:
        return guard
    return guard_decision_finalization(guard, outcome).state


class _FakeDecisionCheckpointPersistence(
    DecisionCheckpointPersistence[IncidentDecisionPayload],
):
    def __init__(self) -> None:
        self._store: dict[
            object,
            DecisionCheckpointState[IncidentDecisionPayload],
        ] = {}

    def load(
        self,
        *,
        key: object,
    ) -> DecisionCheckpointState[IncidentDecisionPayload] | None:
        return self._store.get(key)

    def save(
        self,
        *,
        checkpoint: DecisionCheckpointState[IncidentDecisionPayload],
    ) -> None:
        self._store[checkpoint.finalization.key] = checkpoint


@pytest.mark.unit
@pytest.mark.gate
def test_valid_pre_finalization_checkpoint() -> None:
    identity = _identity()
    lifecycle = _lifecycle(
        identity,
        stage=DecisionLifecycleStage.VERIFICATION,
        transition_index=2,
    )
    finalization = _guard_for_identity(identity)

    checkpoint = decision_checkpoint_state(lifecycle=lifecycle, finalization=finalization)

    assert checkpoint.lifecycle.stage is DecisionLifecycleStage.VERIFICATION
    assert checkpoint.lifecycle.transition_index == 2
    assert checkpoint.finalization.authoritative_outcome is None


@pytest.mark.unit
@pytest.mark.gate
def test_valid_finalization_checkpoint_without_outcome() -> None:
    identity = _identity()
    lifecycle = _lifecycle(identity, stage=DecisionLifecycleStage.FINALIZATION)
    finalization = _guard_for_identity(identity)

    checkpoint = decision_checkpoint_state(lifecycle=lifecycle, finalization=finalization)

    assert checkpoint.lifecycle.stage is DecisionLifecycleStage.FINALIZATION
    assert checkpoint.finalization.authoritative_outcome is None


@pytest.mark.unit
@pytest.mark.gate
def test_valid_terminal_accepted_checkpoint() -> None:
    identity = _identity(version=next_decision_version(initial_decision_version()))
    accepted = _accepted(
        identity=identity,
        lineage=_linear_lineage(identity.version, initial_decision_version()),
    )
    lifecycle = _lifecycle(identity, stage=DecisionLifecycleStage.TERMINAL, transition_index=5)
    finalization = _guard_for_identity(identity, outcome=accepted)

    checkpoint = decision_checkpoint_state(lifecycle=lifecycle, finalization=finalization)

    assert checkpoint.lifecycle.stage is DecisionLifecycleStage.TERMINAL
    assert checkpoint.finalization.authoritative_outcome == accepted


@pytest.mark.unit
@pytest.mark.gate
def test_valid_terminal_rejected_checkpoint() -> None:
    identity = _identity()
    rejected = _resolution(identity=identity, resolution=DecisionResolution.REJECTED)
    lifecycle = _lifecycle(identity, stage=DecisionLifecycleStage.TERMINAL)
    finalization = _guard_for_identity(identity, outcome=rejected)

    checkpoint = decision_checkpoint_state(lifecycle=lifecycle, finalization=finalization)

    assert checkpoint.lifecycle.stage is DecisionLifecycleStage.TERMINAL
    outcome = checkpoint.finalization.authoritative_outcome
    assert isinstance(outcome, AuthoritativeResolutionRecord)
    assert outcome.resolution is DecisionResolution.REJECTED


@pytest.mark.unit
@pytest.mark.gate
def test_valid_terminal_unresolved_checkpoint() -> None:
    identity = _identity()
    unresolved = _resolution(identity=identity, resolution=DecisionResolution.UNRESOLVED)
    lifecycle = _lifecycle(identity, stage=DecisionLifecycleStage.TERMINAL)
    finalization = _guard_for_identity(identity, outcome=unresolved)

    checkpoint = decision_checkpoint_state(lifecycle=lifecycle, finalization=finalization)

    outcome = checkpoint.finalization.authoritative_outcome
    assert isinstance(outcome, AuthoritativeResolutionRecord)
    assert outcome.resolution is DecisionResolution.UNRESOLVED


@pytest.mark.unit
@pytest.mark.gate
def test_lifecycle_finalization_key_mismatch_rejected() -> None:
    identity_a = _identity()
    identity_b = _identity(decision_id=mint_decision_id())
    lifecycle = _lifecycle(identity_a, stage=DecisionLifecycleStage.VERIFICATION)
    finalization = _guard_for_identity(identity_b)

    with pytest.raises(ValueError, match="does not match finalization key"):
        decision_checkpoint_state(lifecycle=lifecycle, finalization=finalization)


@pytest.mark.unit
@pytest.mark.gate
def test_tenant_mismatch_rejected() -> None:
    identity_a = _identity(tenant_id="tenant-a")
    identity_b = _identity(tenant_id="tenant-b")
    lifecycle = _lifecycle(identity_a, stage=DecisionLifecycleStage.VERIFICATION)
    finalization = _guard_for_identity(identity_b)

    with pytest.raises(ValueError, match="does not match finalization key"):
        decision_checkpoint_state(lifecycle=lifecycle, finalization=finalization)


@pytest.mark.unit
@pytest.mark.gate
def test_terminal_without_outcome_rejected() -> None:
    identity = _identity()
    lifecycle = _lifecycle(identity, stage=DecisionLifecycleStage.TERMINAL)
    finalization = _guard_for_identity(identity)

    with pytest.raises(ValueError, match="terminal stage requires authoritative outcome"):
        decision_checkpoint_state(lifecycle=lifecycle, finalization=finalization)


@pytest.mark.unit
@pytest.mark.gate
def test_early_lifecycle_with_authoritative_outcome_rejected() -> None:
    identity = _identity()
    accepted = _accepted(identity=identity)
    lifecycle = _lifecycle(identity, stage=DecisionLifecycleStage.PROPOSAL)
    finalization = _guard_for_identity(identity, outcome=accepted)

    with pytest.raises(ValueError, match="authoritative outcome requires stage"):
        decision_checkpoint_state(lifecycle=lifecycle, finalization=finalization)


@pytest.mark.unit
@pytest.mark.gate
def test_verification_stage_with_authoritative_outcome_rejected() -> None:
    identity = _identity()
    rejected = _resolution(identity=identity, resolution=DecisionResolution.REJECTED)
    lifecycle = _lifecycle(identity, stage=DecisionLifecycleStage.VERIFICATION)
    finalization = _guard_for_identity(identity, outcome=rejected)

    with pytest.raises(ValueError, match="authoritative outcome requires stage"):
        decision_checkpoint_state(lifecycle=lifecycle, finalization=finalization)


@pytest.mark.unit
@pytest.mark.gate
def test_exact_lifecycle_state_preserved_through_persistence_boundary() -> None:
    identity = _identity(version=next_decision_version(initial_decision_version()))
    lifecycle = _lifecycle(
        identity,
        stage=DecisionLifecycleStage.REVISION,
        transition_index=4,
    )
    finalization = _guard_for_identity(identity)
    original = decision_checkpoint_state(lifecycle=lifecycle, finalization=finalization)
    persistence = _FakeDecisionCheckpointPersistence()

    save_decision_checkpoint(persistence, checkpoint=original)
    restored = load_decision_checkpoint(
        persistence,
        key=decision_finalization_key(identity),
    )

    assert restored is not None
    assert restored.lifecycle == original.lifecycle
    assert restored.lifecycle.transition_index == 4
    assert restored.lifecycle.stage is DecisionLifecycleStage.REVISION
    assert restored.finalization.authoritative_outcome is None


@pytest.mark.unit
@pytest.mark.gate
def test_exact_version_preserved_through_restore() -> None:
    version = next_decision_version(initial_decision_version())
    identity = _identity(version=version)
    lifecycle = _lifecycle(identity, stage=DecisionLifecycleStage.VERIFICATION)
    checkpoint = decision_checkpoint_state(
        lifecycle=lifecycle,
        finalization=_guard_for_identity(identity),
    )

    restored = restore_decision_checkpoint_state(checkpoint)

    assert restored.lifecycle.identity.version == version
    assert restored.lifecycle.identity.version.value == 2


@pytest.mark.unit
@pytest.mark.gate
def test_execution_lineage_preserved_through_restore() -> None:
    execution = _execution_lineage()
    identity = _identity(execution=execution)
    lifecycle = _lifecycle(identity, stage=DecisionLifecycleStage.VERIFICATION)
    checkpoint = decision_checkpoint_state(
        lifecycle=lifecycle,
        finalization=_guard_for_identity(identity),
    )

    restored = restore_decision_checkpoint_state(checkpoint)

    assert restored.lifecycle.identity.execution == execution


@pytest.mark.unit
@pytest.mark.gate
def test_accepted_artifact_preserved_through_restore() -> None:
    identity = _identity()
    artifact = _artifact(recommendation="contain")
    accepted = _accepted(identity=identity, artifact=artifact)
    lifecycle = _lifecycle(identity, stage=DecisionLifecycleStage.TERMINAL)
    checkpoint = decision_checkpoint_state(
        lifecycle=lifecycle,
        finalization=_guard_for_identity(identity, outcome=accepted),
    )
    persistence = _FakeDecisionCheckpointPersistence()

    save_decision_checkpoint(persistence, checkpoint=checkpoint)
    restored = load_decision_checkpoint(
        persistence,
        key=decision_finalization_key(identity),
    )

    assert restored is not None
    outcome = restored.finalization.authoritative_outcome
    assert isinstance(outcome, AuthoritativeAcceptedDecision)
    assert outcome.artifact == artifact
    assert outcome.lineage == accepted.lineage
    assert outcome.identity == accepted.identity


@pytest.mark.unit
@pytest.mark.gate
def test_authoritative_resolution_preserved_through_restore() -> None:
    identity = _identity()
    rejected = _resolution(identity=identity, resolution=DecisionResolution.REJECTED)
    lifecycle = _lifecycle(identity, stage=DecisionLifecycleStage.TERMINAL)
    checkpoint = decision_checkpoint_state(
        lifecycle=lifecycle,
        finalization=_guard_for_identity(identity, outcome=rejected),
    )

    restored = restore_decision_checkpoint_state(checkpoint)

    assert restored.finalization.authoritative_outcome == rejected


@pytest.mark.unit
@pytest.mark.gate
def test_idempotent_replay_after_restore() -> None:
    identity = _identity()
    accepted = _accepted(identity=identity)
    lifecycle = _lifecycle(identity, stage=DecisionLifecycleStage.TERMINAL)
    checkpoint = decision_checkpoint_state(
        lifecycle=lifecycle,
        finalization=_guard_for_identity(identity, outcome=accepted),
    )
    persistence = _FakeDecisionCheckpointPersistence()
    save_decision_checkpoint(persistence, checkpoint=checkpoint)
    restored = load_decision_checkpoint(
        persistence,
        key=decision_finalization_key(identity),
    )

    assert restored is not None
    outcome = restored.finalization.authoritative_outcome
    assert outcome is not None
    replay = guard_decision_finalization(restored.finalization, outcome)

    assert replay.disposition is DecisionFinalizeDisposition.IDEMPOTENT_REPLAY
    assert replay.state is restored.finalization


@pytest.mark.unit
@pytest.mark.gate
def test_competing_outcome_conflict_after_restore() -> None:
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
        lineage=_root_lineage(),
    )
    accepted_v2 = _accepted(
        identity=identity_v2,
        lineage=_linear_lineage(identity_v2.version, identity_v1.version),
    )
    lifecycle = _lifecycle(identity_v1, stage=DecisionLifecycleStage.TERMINAL)
    checkpoint = decision_checkpoint_state(
        lifecycle=lifecycle,
        finalization=_guard_for_identity(identity_v1, outcome=accepted_v1),
    )
    persistence = _FakeDecisionCheckpointPersistence()
    save_decision_checkpoint(persistence, checkpoint=checkpoint)
    restored = load_decision_checkpoint(
        persistence,
        key=decision_finalization_key(identity_v1),
    )

    assert restored is not None
    with pytest.raises(DecisionFinalizationConflictError):
        guard_decision_finalization(restored.finalization, accepted_v2)


@pytest.mark.unit
@pytest.mark.gate
def test_checkpoint_state_is_immutable() -> None:
    identity = _identity()
    checkpoint = decision_checkpoint_state(
        lifecycle=_lifecycle(identity, stage=DecisionLifecycleStage.VERIFICATION),
        finalization=_guard_for_identity(identity),
    )

    with pytest.raises(FrozenInstanceError):
        setattr(checkpoint, "lifecycle", checkpoint.lifecycle)


@pytest.mark.unit
@pytest.mark.gate
def test_branched_synthesis_lineage_preserved_through_restore() -> None:
    v2a = decision_lineage_ref(
        next_decision_version(initial_decision_version()),
        validate_decision_branch_id("A"),
    )
    v2b = decision_lineage_ref(
        next_decision_version(initial_decision_version()),
        validate_decision_branch_id("B"),
    )
    synthesis_lineage = decision_version_lineage(
        current=decision_lineage_ref(DecisionVersion(3)),
        parents=(v2a, v2b),
    )
    identity = _identity(version=DecisionVersion(3))
    accepted = _accepted(identity=identity, lineage=synthesis_lineage)
    lifecycle = _lifecycle(identity, stage=DecisionLifecycleStage.TERMINAL)
    checkpoint = decision_checkpoint_state(
        lifecycle=lifecycle,
        finalization=_guard_for_identity(identity, outcome=accepted),
    )
    persistence = _FakeDecisionCheckpointPersistence()

    save_decision_checkpoint(persistence, checkpoint=checkpoint)
    restored = load_decision_checkpoint(
        persistence,
        key=decision_finalization_key(identity),
    )

    assert restored is not None
    outcome = restored.finalization.authoritative_outcome
    assert isinstance(outcome, AuthoritativeAcceptedDecision)
    assert outcome.lineage == synthesis_lineage
    assert outcome.lineage.parents == (v2a, v2b)
    assert restore_decision_checkpoint_state(restored).finalization.authoritative_outcome == outcome
