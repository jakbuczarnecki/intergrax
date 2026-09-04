# © Artur Czarnecki. All rights reserved.

"""DS-REC-02/03 — durable Decision recovery and revision budget preservation."""

from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.contracts.decision_checkpoint import (
    decision_checkpoint_state,
)
from intergrax.contracts.decision_finalization import (
    decision_finalization_key,
    initial_decision_finalize_guard,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.decision_lifecycle import (
    DecisionLifecycleStage,
    initial_decision_lifecycle_state,
    transition_decision_lifecycle,
)
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    DecisionArtifact,
    DecisionProposalRef,
    DecisionVersionLineage,
    decision_lineage_ref,
    validate_decision_artifact_kind,
)
from intergrax.contracts.decision_resolution import (
    AuthoritativeResolutionRecord,
    DecisionResolution,
)
from intergrax.contracts.decision_revision import (
    DecisionRevisionCheckpointState,
    DecisionRevisionDisposition,
    DecisionRevisionPolicyMismatchError,
    decision_revision_checkpoint_state,
    decision_revision_policy,
    evaluate_decision_revision,
    revision_policy_from_checkpoint,
    revision_state_from_checkpoint,
)
from intergrax.contracts.decision_verification import (
    VerificationDisposition,
    VerificationResult,
    VerificationStageOutcome,
    validate_verification_finding_code,
    validate_verification_requirement_code,
    validate_verification_stage_kind,
    verification_challenge,
    verification_finding,
    verification_result,
    verification_stage_record,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.execution.decision_checkpoint_persistence import (
    save_decision_checkpoint,
)
from intergrax.runtime.execution.decision_finalization_persistence import (
    DecisionDurableFinalizationDisposition,
)
from intergrax.runtime.execution.decision_recovery import (
    DecisionCheckpointCorruptionError,
    is_terminal_resumable_checkpoint,
    persist_terminal_decision_state,
    reconcile_checkpoint_with_durable_finalization,
    resume_decision_from_durable_state,
)
from intergrax.runtime.execution.in_memory_decision_checkpoint_persistence import (
    InMemoryDecisionCheckpointPersistence,
)
from intergrax.runtime.execution.in_memory_decision_finalization_persistence import (
    InMemoryDecisionFinalizationPersistence,
)
from intergrax.runtime.execution.sqlite_decision_checkpoint_persistence import (
    SQLiteDecisionCheckpointPersistence,
)
from intergrax.runtime.execution.sqlite_decision_finalization_persistence import (
    SQLiteDecisionFinalizationPersistence,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


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


def _identity() -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="incident", subject="incident-123"),
        tenant_id="tenant-a",
        execution=_execution_lineage(),
    )


def _proposal_ref(identity: DecisionIdentity) -> DecisionProposalRef:
    return DecisionProposalRef(
        identity=identity,
        lineage_ref=decision_lineage_ref(identity.version),
    )


def _verification_result(proposal_ref: DecisionProposalRef) -> VerificationResult:
    finding = verification_finding(
        code=validate_verification_finding_code("verification.semantic.below_requirement"),
        message="need more evidence",
    )
    stage = validate_verification_stage_kind("semantic")
    return verification_result(
        proposal_ref=proposal_ref,
        disposition=VerificationDisposition.CHALLENGED,
        stage_records=(
            verification_stage_record(
                proposal_ref=proposal_ref,
                stage=stage,
                outcome=VerificationStageOutcome.CHALLENGED,
                challenge=verification_challenge(
                    proposal_ref=proposal_ref,
                    stage=stage,
                    requirement_code=validate_verification_requirement_code(
                        "verification.semantic.below_requirement",
                    ),
                    finding=finding,
                ),
            ),
        ),
    )


def _revision_checkpoint(
    *,
    identity: DecisionIdentity,
    revision_count: int,
    max_revisions: int,
) -> DecisionRevisionCheckpointState:
    return decision_revision_checkpoint_state(
        proposal_ref=_proposal_ref(identity),
        revision_count=revision_count,
        max_revisions=max_revisions,
    )


def _lifecycle_at_revision(identity: DecisionIdentity) -> object:
    state = initial_decision_lifecycle_state(identity)
    state = transition_decision_lifecycle(state, DecisionLifecycleStage.VERIFICATION)
    return transition_decision_lifecycle(state, DecisionLifecycleStage.REVISION)


def _lifecycle_at_finalization(identity: DecisionIdentity) -> object:
    state = initial_decision_lifecycle_state(identity)
    state = transition_decision_lifecycle(state, DecisionLifecycleStage.VERIFICATION)
    state = transition_decision_lifecycle(state, DecisionLifecycleStage.RESOLUTION)
    return transition_decision_lifecycle(state, DecisionLifecycleStage.FINALIZATION)


def test_revision_checkpoint_survives_restart() -> None:
    identity = _identity()
    lifecycle = _lifecycle_at_revision(identity)
    revision = _revision_checkpoint(identity=identity, revision_count=2, max_revisions=3)
    checkpoint = decision_checkpoint_state(
        lifecycle=lifecycle,
        finalization=initial_decision_finalize_guard(decision_finalization_key(identity)),
        revision=revision,
    )
    checkpoint_store = InMemoryDecisionCheckpointPersistence[IncidentDecisionPayload]()
    finalization_store = InMemoryDecisionFinalizationPersistence[IncidentDecisionPayload]()
    save_decision_checkpoint(checkpoint_store, checkpoint=checkpoint)

    resumed = resume_decision_from_durable_state(
        checkpoint_persistence=checkpoint_store,
        finalization_persistence=finalization_store,
        key=decision_finalization_key(identity),
        runtime_revision_policy=decision_revision_policy(max_revisions=3),
    )

    assert resumed is not None
    assert resumed.revision is not None
    assert resumed.revision.revision_count == 2
    assert resumed.revision.max_revisions == 3
    restored_state = revision_state_from_checkpoint(resumed.revision)
    restored_policy = revision_policy_from_checkpoint(resumed.revision)
    assert restored_state.revision_count == 2
    assert restored_policy.max_revisions == 3
    assert restored_policy.max_revisions - restored_state.revision_count == 1


def test_exhausted_revision_remains_exhausted_after_restart() -> None:
    identity = _identity()
    revision = _revision_checkpoint(identity=identity, revision_count=3, max_revisions=3)
    lifecycle = _lifecycle_at_revision(identity)
    checkpoint = decision_checkpoint_state(
        lifecycle=lifecycle,
        finalization=initial_decision_finalize_guard(decision_finalization_key(identity)),
        revision=revision,
    )
    checkpoint_store = InMemoryDecisionCheckpointPersistence[IncidentDecisionPayload]()
    save_decision_checkpoint(checkpoint_store, checkpoint=checkpoint)
    resumed = resume_decision_from_durable_state(
        checkpoint_persistence=checkpoint_store,
        finalization_persistence=InMemoryDecisionFinalizationPersistence(),
        key=decision_finalization_key(identity),
        runtime_revision_policy=decision_revision_policy(max_revisions=3),
    )
    assert resumed is not None
    assert resumed.revision is not None
    restored_state = revision_state_from_checkpoint(resumed.revision)
    restored_policy = revision_policy_from_checkpoint(resumed.revision)
    challenged = evaluate_decision_revision(
        policy=restored_policy,
        state=restored_state,
        verification_result=_verification_result(_proposal_ref(identity)),
    )
    assert challenged.disposition is DecisionRevisionDisposition.EXHAUSTED


def test_resume_policy_mismatch_fails_closed() -> None:
    identity = _identity()
    revision = _revision_checkpoint(identity=identity, revision_count=1, max_revisions=3)
    lifecycle = initial_decision_lifecycle_state(identity)
    checkpoint = decision_checkpoint_state(
        lifecycle=lifecycle,
        finalization=initial_decision_finalize_guard(decision_finalization_key(identity)),
        revision=revision,
    )
    checkpoint_store = InMemoryDecisionCheckpointPersistence[IncidentDecisionPayload]()
    save_decision_checkpoint(checkpoint_store, checkpoint=checkpoint)
    with pytest.raises(DecisionRevisionPolicyMismatchError):
        resume_decision_from_durable_state(
            checkpoint_persistence=checkpoint_store,
            finalization_persistence=InMemoryDecisionFinalizationPersistence(),
            key=decision_finalization_key(identity),
            runtime_revision_policy=decision_revision_policy(max_revisions=5),
        )


def test_crash_after_durable_commit_converges_terminal() -> None:
    identity = _identity()
    accepted = AuthoritativeAcceptedDecision(
        identity=identity,
        artifact=DecisionArtifact(
            kind=validate_decision_artifact_kind("incident_resolution"),
            content=IncidentDecisionPayload(recommendation="contain"),
        ),
        lineage=DecisionVersionLineage(current=decision_lineage_ref(identity.version)),
    )
    lifecycle = _lifecycle_at_finalization(identity)
    guard = initial_decision_finalize_guard(decision_finalization_key(identity))
    from intergrax.contracts.decision_finalization import guard_decision_finalization

    guard = guard_decision_finalization(guard, accepted).state
    pre_terminal_checkpoint = decision_checkpoint_state(
        lifecycle=lifecycle,
        finalization=guard,
    )
    checkpoint_store = InMemoryDecisionCheckpointPersistence[IncidentDecisionPayload]()
    finalization_store = InMemoryDecisionFinalizationPersistence[IncidentDecisionPayload]()
    finalization_store.commit_authoritative_outcome(
        key=decision_finalization_key(identity),
        requested_outcome=accepted,
    )

    resumed = resume_decision_from_durable_state(
        checkpoint_persistence=checkpoint_store,
        finalization_persistence=finalization_store,
        key=decision_finalization_key(identity),
    )
    assert resumed is not None
    assert resumed.lifecycle.stage is DecisionLifecycleStage.TERMINAL
    assert resumed.finalization.authoritative_outcome == accepted

    save_decision_checkpoint(checkpoint_store, checkpoint=pre_terminal_checkpoint)
    resumed = resume_decision_from_durable_state(
        checkpoint_persistence=checkpoint_store,
        finalization_persistence=finalization_store,
        key=decision_finalization_key(identity),
    )
    assert resumed is not None
    converged = reconcile_checkpoint_with_durable_finalization(
        resumed,
        durable_guard=finalization_store.load_guard_state(
            key=decision_finalization_key(identity),
        ),
    )
    assert converged.lifecycle.stage is DecisionLifecycleStage.TERMINAL
    assert converged.finalization.authoritative_outcome == accepted
    replay = finalization_store.commit_authoritative_outcome(
        key=decision_finalization_key(identity),
        requested_outcome=accepted,
    )
    assert replay.disposition is DecisionDurableFinalizationDisposition.IDEMPOTENT_REPLAY


def test_persist_terminal_decision_state_orders_commit_before_checkpoint() -> None:
    identity = _identity()
    accepted = AuthoritativeAcceptedDecision(
        identity=identity,
        artifact=DecisionArtifact(
            kind=validate_decision_artifact_kind("incident_resolution"),
            content=IncidentDecisionPayload(recommendation="rollback"),
        ),
        lineage=DecisionVersionLineage(current=decision_lineage_ref(identity.version)),
    )
    lifecycle = _lifecycle_at_finalization(identity)
    from intergrax.contracts.decision_finalization import guard_decision_finalization

    guard = guard_decision_finalization(
        initial_decision_finalize_guard(decision_finalization_key(identity)),
        accepted,
    ).state
    checkpoint = decision_checkpoint_state(lifecycle=lifecycle, finalization=guard)
    checkpoint_store = InMemoryDecisionCheckpointPersistence[IncidentDecisionPayload]()
    finalization_store = InMemoryDecisionFinalizationPersistence[IncidentDecisionPayload]()
    terminal = persist_terminal_decision_state(
        checkpoint_persistence=checkpoint_store,
        finalization_persistence=finalization_store,
        checkpoint=checkpoint,
    )
    assert is_terminal_resumable_checkpoint(terminal)
    loaded_guard = finalization_store.load_guard_state(key=decision_finalization_key(identity))
    assert loaded_guard is not None
    assert loaded_guard.authoritative_outcome == accepted


def test_durable_outcome_conflicting_checkpoint_fails_closed() -> None:
    identity = _identity()
    accepted = AuthoritativeAcceptedDecision(
        identity=identity,
        artifact=DecisionArtifact(
            kind=validate_decision_artifact_kind("incident_resolution"),
            content=IncidentDecisionPayload(recommendation="contain"),
        ),
        lineage=DecisionVersionLineage(current=decision_lineage_ref(identity.version)),
    )
    rejected = AuthoritativeResolutionRecord(
        identity=identity,
        resolution=DecisionResolution.REJECTED,
    )
    from intergrax.contracts.decision_finalization import guard_decision_finalization

    durable_guard = guard_decision_finalization(
        initial_decision_finalize_guard(decision_finalization_key(identity)),
        accepted,
    ).state
    checkpoint = decision_checkpoint_state(
        lifecycle=_lifecycle_at_finalization(identity),
        finalization=guard_decision_finalization(
            initial_decision_finalize_guard(decision_finalization_key(identity)),
            rejected,
        ).state,
    )
    checkpoint_store = InMemoryDecisionCheckpointPersistence[IncidentDecisionPayload]()
    save_decision_checkpoint(checkpoint_store, checkpoint=checkpoint)
    finalization_store = InMemoryDecisionFinalizationPersistence[IncidentDecisionPayload]()
    finalization_store.commit_authoritative_outcome(
        key=decision_finalization_key(identity),
        requested_outcome=accepted,
    )
    with pytest.raises(DecisionCheckpointCorruptionError):
        resume_decision_from_durable_state(
            checkpoint_persistence=checkpoint_store,
            finalization_persistence=finalization_store,
            key=decision_finalization_key(identity),
        )


def _subprocess_write_state(db_dir: str) -> None:
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="incident", subject="subprocess"),
        tenant_id="tenant-subprocess",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )
    revision = decision_revision_checkpoint_state(
        proposal_ref=DecisionProposalRef(
            identity=identity,
            lineage_ref=decision_lineage_ref(identity.version),
        ),
        revision_count=2,
        max_revisions=3,
    )
    lifecycle = _lifecycle_at_revision(identity)
    checkpoint = decision_checkpoint_state(
        lifecycle=lifecycle,
        finalization=initial_decision_finalize_guard(decision_finalization_key(identity)),
        revision=revision,
    )
    db_path = Path(db_dir)
    checkpoint_store = SQLiteDecisionCheckpointPersistence(
        db_path=db_path / "checkpoint.db",
    )
    save_decision_checkpoint(checkpoint_store, checkpoint=checkpoint)
    sidecar = {
        "tenant_id": identity.tenant_id,
        "decision_id": str(identity.decision_id),
        "namespace": identity.scope.namespace,
        "subject": identity.scope.subject,
    }
    (db_path / "identity.json").write_text(json.dumps(sidecar), encoding="utf-8")


def _subprocess_read_state(db_dir: str) -> tuple[int, int]:
    db_path = Path(db_dir)
    sidecar = json.loads((db_path / "identity.json").read_text(encoding="utf-8"))
    from intergrax.contracts.decision_finalization import DecisionFinalizationKey
    from intergrax.contracts.decision_identity import validate_decision_id

    key = DecisionFinalizationKey(
        decision_id=validate_decision_id(sidecar["decision_id"]),
        scope=DecisionScope(namespace=sidecar["namespace"], subject=sidecar["subject"]),
        tenant_id=sidecar["tenant_id"],
    )
    checkpoint_store = SQLiteDecisionCheckpointPersistence(
        db_path=db_path / "checkpoint.db",
    )
    loaded = resume_decision_from_durable_state(
        checkpoint_persistence=checkpoint_store,
        finalization_persistence=SQLiteDecisionFinalizationPersistence(
            db_path=db_path / "finalization.db",
        ),
        key=key,
        runtime_revision_policy=decision_revision_policy(max_revisions=3),
    )
    if loaded is None or loaded.revision is None:
        return (-1, -1)
    return (loaded.revision.revision_count, loaded.revision.max_revisions)


def _subprocess_reader_with_queue(db: str, queue: mp.Queue[tuple[int, int]]) -> None:
    queue.put(_subprocess_read_state(db))


def test_sqlite_checkpoint_survives_subprocess_restart(tmp_path: Path) -> None:
    db_dir = str(tmp_path)
    ctx = mp.get_context("spawn")
    writer = ctx.Process(target=_subprocess_write_state, args=(db_dir,))
    writer.start()
    writer.join()
    assert writer.exitcode == 0

    result_queue: mp.Queue[tuple[int, int]] = ctx.Queue()
    reader = ctx.Process(
        target=_subprocess_reader_with_queue,
        args=(db_dir, result_queue),
    )
    reader.start()
    reader.join()
    assert reader.exitcode == 0
    revision_count, max_revisions = result_queue.get(timeout=5)
    assert revision_count == 2
    assert max_revisions == 3
