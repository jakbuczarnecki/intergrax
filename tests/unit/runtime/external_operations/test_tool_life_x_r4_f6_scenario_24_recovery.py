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
from intergrax.runtime.external_operations.external_operation_ownership import (
    ExternalOperationFencedCommitRejectedError,
    ProcessLocalExternalOperationOwner,
    mark_physical_running,
    recover_running_operation_after_certified_owner_loss,
)
from intergrax.runtime.external_operations.sqlite_tool_life_durable_stores import (
    SqliteToolLifeDurableStoresBundle,
)
from intergrax.runtime.external_operations.worker_execution_certified_owner_loss import (
    build_worker_execution_certified_owner_loss_certifier,
)
from tests.unit.runtime.external_operations.test_tool_life_x_l3_r8_sqlite_durable_revocation_shared_store import (
    _OP,
    _TENANT,
    _revoke_bound,
    _running_with_worker,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.fixture
def durable_bundle(tmp_path: Path) -> SqliteToolLifeDurableStoresBundle:
    return SqliteToolLifeDurableStoresBundle(db_path=tmp_path / "scenario24.db")


def test_scenario_24_positive_recovery_after_revocation_and_certified_loss(
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
