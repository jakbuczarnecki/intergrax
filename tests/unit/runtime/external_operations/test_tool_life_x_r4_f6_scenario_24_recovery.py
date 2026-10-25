# © Artur Czarnecki. All rights reserved.

"""F6 / TL8 scenario 24 — isolated qual-only certified owner-loss recovery (no prod activation)."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.autonomous_work.in_memory_repository import InMemoryWorkerInstanceRepository
from intergrax.contracts.certified_owner_loss_evidence import (
    CertifiedOwnerLossEvidenceRejectedError,
    CertifiedOwnerLossReason,
)
from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationIntentState,
    ExternalOperationPhysicalState,
)
from intergrax.contracts.external_operation_redrive_authority import (
    ExternalOperationRedriveCompetingAuthorityError,
    ExternalOperationRedriveEffectClass,
    ExternalOperationRedriveEffectClassRejectedError,
)
from intergrax.runtime.external_operations.external_operation_ownership import (
    ExternalOperationFencedCommitRejectedError,
    ProcessLocalExternalOperationOwner,
    mark_physical_running,
    mark_physical_terminal_if_owner_matches,
    recover_running_operation_after_certified_owner_loss,
)
from intergrax.runtime.external_operations.recovery_external_operation_gate import (
    admit_unknown_external_operation_for_reconciliation,
)
from intergrax.runtime.external_operations.redrive_external_operation_authority import (
    transfer_redrive_authority_after_reconciliation,
)
from intergrax.runtime.external_operations.sqlite_tool_life_durable_stores import (
    SqliteToolLifeDurableStoresBundle,
)
from intergrax.runtime.external_operations.worker_execution_certified_owner_loss import (
    build_worker_execution_certified_owner_loss_certifier,
)
from tests.unit.autonomous_work.repository_contracts import worker_instance
from tests.unit.runtime.external_operations.test_tool_life_x_l3_r8_sqlite_durable_revocation_shared_store import (
    _BOUND_EPOCH,
    _OP,
    _TENANT,
    _revoke_bound,
    _running_with_worker,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.fixture
def durable_bundle(tmp_path: Path) -> SqliteToolLifeDurableStoresBundle:
    return SqliteToolLifeDurableStoresBundle(db_path=tmp_path / "scenario24.db")


def test_scenario_24_certified_loss_transitions_running_to_unknown(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    _running_with_worker(durable_bundle, repo)
    proof = _revoke_bound(durable_bundle)
    certifier = build_worker_execution_certified_owner_loss_certifier(
        store=durable_bundle.external_operation_state,
    )
    staged = certifier.stage_certified_owner_loss(
        tenant_id=_TENANT,
        operation_id=_OP,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
        liveness_proof_ref=proof,
    )
    evidence_id = staged.staged_certified_owner_loss.evidence.evidence_id
    unknown = recover_running_operation_after_certified_owner_loss(
        durable_bundle.external_operation_state,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=staged.revision,
        evidence_id=evidence_id,
    )
    assert unknown.physical_state is ExternalOperationPhysicalState.UNKNOWN
    assert unknown.intent_state is ExternalOperationIntentState.TERMINATING
    persisted = durable_bundle.external_operation_state.load(_TENANT, _OP)
    assert persisted is not None
    assert evidence_id in persisted.consumed_certified_loss_evidence_ids


def test_scenario_24_stale_epoch_cannot_resume_physical_effect(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    _worker, bound, _worker_id, _owner = _running_with_worker(durable_bundle, repo)
    _revoke_bound(durable_bundle)
    stale = ProcessLocalExternalOperationOwner.mint()
    with pytest.raises(
        (ExternalOperationFencedCommitRejectedError, CertifiedOwnerLossEvidenceRejectedError),
    ):
        mark_physical_running(
            durable_bundle.external_operation_state,
            tenant_id=_TENANT,
            operation_id=_OP,
            expected_revision=bound.revision,
            owner=stale,
        )


def test_scenario_24_mark_physical_blocked_without_redrive_authority_transfer(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    """Revoked epoch blocks physical restart until sanctioned authority transfer."""
    repo = InMemoryWorkerInstanceRepository()
    _running_with_worker(durable_bundle, repo)
    proof = _revoke_bound(durable_bundle)
    certifier = build_worker_execution_certified_owner_loss_certifier(
        store=durable_bundle.external_operation_state,
    )
    staged = certifier.stage_certified_owner_loss(
        tenant_id=_TENANT,
        operation_id=_OP,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
        liveness_proof_ref=proof,
    )
    evidence_id = staged.staged_certified_owner_loss.evidence.evidence_id
    unknown = recover_running_operation_after_certified_owner_loss(
        durable_bundle.external_operation_state,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=staged.revision,
        evidence_id=evidence_id,
    )
    assert unknown.physical_state is ExternalOperationPhysicalState.UNKNOWN
    assert unknown.worker_instance_id is not None
    assert unknown.worker_ownership_epoch is not None
    admit_unknown_external_operation_for_reconciliation(
        durable_bundle.external_operation_state,
        tenant_id=_TENANT,
        operation_id=_OP,
    )
    fresh_owner = ProcessLocalExternalOperationOwner.mint()
    with pytest.raises(ExternalOperationFencedCommitRejectedError, match="revoked"):
        mark_physical_running(
            durable_bundle.external_operation_state,
            tenant_id=_TENANT,
            operation_id=_OP,
            expected_revision=unknown.revision,
            owner=fresh_owner,
        )


def test_scenario_24_positive_redrive_after_authority_transfer(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    _running_with_worker(durable_bundle, repo)
    proof = _revoke_bound(durable_bundle)
    certifier = build_worker_execution_certified_owner_loss_certifier(
        store=durable_bundle.external_operation_state,
    )
    staged = certifier.stage_certified_owner_loss(
        tenant_id=_TENANT,
        operation_id=_OP,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
        liveness_proof_ref=proof,
    )
    evidence_id = staged.staged_certified_owner_loss.evidence.evidence_id
    unknown = recover_running_operation_after_certified_owner_loss(
        durable_bundle.external_operation_state,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=staged.revision,
        evidence_id=evidence_id,
    )
    assert unknown.physical_state is ExternalOperationPhysicalState.UNKNOWN
    recovery_worker = worker_instance()
    repo.create(recovery_worker)
    transferred = transfer_redrive_authority_after_reconciliation(
        durable_bundle.external_operation_state,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=unknown.revision,
        effect_class=ExternalOperationRedriveEffectClass.CLASS_B_IDEMPOTENT_OR_RECONCILED,
        worker_instance_id=str(recovery_worker.worker_instance_id),
        worker_ownership_epoch=_BOUND_EPOCH + 1,
    )
    assert transferred.physical_state is ExternalOperationPhysicalState.NOT_STARTED
    assert transferred.intent_state is ExternalOperationIntentState.ACTIVE
    assert transferred.consumed_redrive_admission_ids
    fresh_owner = ProcessLocalExternalOperationOwner.mint()
    running = mark_physical_running(
        durable_bundle.external_operation_state,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=transferred.revision,
        owner=fresh_owner,
    )
    assert running.physical_state is ExternalOperationPhysicalState.RUNNING
    terminal = mark_physical_terminal_if_owner_matches(
        durable_bundle.external_operation_state,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=running.revision,
        physical_state=ExternalOperationPhysicalState.SUCCEEDED,
        owner=fresh_owner,
    )
    assert terminal.physical_state is ExternalOperationPhysicalState.SUCCEEDED
    stale_owner = ProcessLocalExternalOperationOwner.mint()
    with pytest.raises(ExternalOperationFencedCommitRejectedError):
        mark_physical_terminal_if_owner_matches(
            durable_bundle.external_operation_state,
            tenant_id=_TENANT,
            operation_id=_OP,
            expected_revision=terminal.revision,
            physical_state=ExternalOperationPhysicalState.SUCCEEDED,
            owner=stale_owner,
        )


def test_scenario_24_class_c_effect_cannot_obtain_redrive_authority(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    _running_with_worker(durable_bundle, repo)
    proof = _revoke_bound(durable_bundle)
    certifier = build_worker_execution_certified_owner_loss_certifier(
        store=durable_bundle.external_operation_state,
    )
    staged = certifier.stage_certified_owner_loss(
        tenant_id=_TENANT,
        operation_id=_OP,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
        liveness_proof_ref=proof,
    )
    evidence_id = staged.staged_certified_owner_loss.evidence.evidence_id
    unknown = recover_running_operation_after_certified_owner_loss(
        durable_bundle.external_operation_state,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=staged.revision,
        evidence_id=evidence_id,
    )
    recovery_worker = worker_instance()
    repo.create(recovery_worker)
    with pytest.raises(ExternalOperationRedriveEffectClassRejectedError):
        transfer_redrive_authority_after_reconciliation(
            durable_bundle.external_operation_state,
            tenant_id=_TENANT,
            operation_id=_OP,
            expected_revision=unknown.revision,
            effect_class=ExternalOperationRedriveEffectClass.CLASS_C_FAIL_CLOSED,
            worker_instance_id=str(recovery_worker.worker_instance_id),
            worker_ownership_epoch=_BOUND_EPOCH + 1,
        )


def test_scenario_24_competing_redrive_transfer_rejected(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    _running_with_worker(durable_bundle, repo)
    proof = _revoke_bound(durable_bundle)
    certifier = build_worker_execution_certified_owner_loss_certifier(
        store=durable_bundle.external_operation_state,
    )
    staged = certifier.stage_certified_owner_loss(
        tenant_id=_TENANT,
        operation_id=_OP,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
        liveness_proof_ref=proof,
    )
    evidence_id = staged.staged_certified_owner_loss.evidence.evidence_id
    unknown = recover_running_operation_after_certified_owner_loss(
        durable_bundle.external_operation_state,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=staged.revision,
        evidence_id=evidence_id,
    )
    worker_a = worker_instance()
    worker_b = worker_instance()
    repo.create(worker_a)
    repo.create(worker_b)
    transfer_redrive_authority_after_reconciliation(
        durable_bundle.external_operation_state,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=unknown.revision,
        effect_class=ExternalOperationRedriveEffectClass.CLASS_B_IDEMPOTENT_OR_RECONCILED,
        worker_instance_id=str(worker_a.worker_instance_id),
        worker_ownership_epoch=_BOUND_EPOCH + 1,
    )
    loaded = durable_bundle.external_operation_state.load(_TENANT, _OP)
    assert loaded is not None
    with pytest.raises(ExternalOperationRedriveCompetingAuthorityError):
        durable_bundle.external_operation_state.transfer_redrive_worker_execution_authority(
            tenant_id=_TENANT,
            operation_id=_OP,
            expected_revision=loaded.revision,
            worker_instance_id=str(worker_b.worker_instance_id),
            worker_ownership_epoch=_BOUND_EPOCH + 2,
            redrive_admission_id="redrive-admit:competing-worker",
        )


def test_scenario_24_forged_recovery_without_staged_evidence_rejected(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    _running_with_worker(durable_bundle, repo)
    loaded = durable_bundle.external_operation_state.load(_TENANT, _OP)
    assert loaded is not None
    with pytest.raises(CertifiedOwnerLossEvidenceRejectedError):
        recover_running_operation_after_certified_owner_loss(
            durable_bundle.external_operation_state,
            tenant_id=_TENANT,
            operation_id=_OP,
            expected_revision=loaded.revision,
            evidence_id="forged-evidence",
        )
