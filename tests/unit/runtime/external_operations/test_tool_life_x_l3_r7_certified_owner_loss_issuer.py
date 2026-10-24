# © Artur Czarnecki. All rights reserved.

"""TOOL-LIFE-X L3-R7 — sanctioned worker-execution certified owner-loss issuer."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from intergrax.autonomous_work.in_memory_repository import InMemoryWorkerInstanceRepository
from intergrax.contracts.autonomous_work.lifecycle import WorkerLifecycleState
from intergrax.contracts.autonomous_work.revision import Revision, initial_revision
from intergrax.contracts.certified_owner_loss_evidence import (
    CANONICAL_WORKER_EXECUTION_OWNER_LOSS_ISSUER_ID,
    CertifiedOwnerLossDirectIssuerInvocationError,
    CertifiedOwnerLossEvidenceRejectedError,
    CertifiedOwnerLossIssuerNotConfiguredError,
    CertifiedOwnerLossReason,
    ExternalOperationOwnershipEpochV1,
)
from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationIntentState,
    ExternalOperationPhysicalState,
)
from intergrax.runtime.external_operations.certified_owner_loss_composition import (
    reset_certified_owner_loss_composition_for_tests,
)
from intergrax.runtime.external_operations.external_operation_ownership import (
    ProcessLocalExternalOperationOwner,
    mark_physical_running,
    recover_running_operation_after_certified_owner_loss,
)
from intergrax.runtime.external_operations.external_operation_state_store import (
    InMemoryExternalOperationStateStore,
)
from intergrax.runtime.external_operations.recovery_external_operation_gate import (
    admit_unknown_external_operation_for_reconciliation,
)
from intergrax.runtime.external_operations.worker_execution_certified_owner_loss import (
    WorkerExecutionCertifiedOwnerLossEvidenceIssuer,
    bind_worker_execution_ownership,
    build_worker_execution_certified_owner_loss_certifier,
    configure_worker_execution_certified_owner_loss,
)
from tests.unit.autonomous_work.repository_contracts import worker_instance

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant_l3r7"
_OP = "op-l3r7"


@pytest.fixture(autouse=True)
def _reset_composition() -> None:
    reset_certified_owner_loss_composition_for_tests()
    yield
    reset_certified_owner_loss_composition_for_tests()


def _running_with_worker(
    store: InMemoryExternalOperationStateStore,
    repo: InMemoryWorkerInstanceRepository,
    *,
    worker_revision: int = 0,
    lifecycle: WorkerLifecycleState = WorkerLifecycleState.WORKING,
) -> tuple[object, object, str, ProcessLocalExternalOperationOwner]:
    worker = worker_instance(
        lifecycle_state=lifecycle,
        revision=initial_revision() if worker_revision == 0 else Revision(worker_revision),
    )
    repo.create(worker)
    owner = ProcessLocalExternalOperationOwner.mint()
    record = store.create_if_absent(_TENANT, _OP, owner_token=owner.token)
    running = mark_physical_running(
        store,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=record.revision,
        owner=owner,
    )
    bound = bind_worker_execution_ownership(
        store,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=running.revision,
        worker_instance_id=str(worker.worker_instance_id),
        worker_ownership_epoch=worker.revision.value,
    )
    return worker, bound, str(worker.worker_instance_id), owner


def test_l3r7_direct_issuer_port_invocation_rejected() -> None:
    issuer = WorkerExecutionCertifiedOwnerLossEvidenceIssuer()
    epoch = ExternalOperationOwnershipEpochV1(
        operation_id=_OP,
        tenant_id=_TENANT,
        owner_token="owner_x",
        store_revision=1,
    )
    with pytest.raises(CertifiedOwnerLossDirectIssuerInvocationError):
        issuer.issue_owner_loss_evidence(
            epoch=epoch,
            loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
            liveness_proof_ref=None,
        )


def test_l3r7_positive_stage_and_apply_after_worker_loss() -> None:
    store = InMemoryExternalOperationStateStore()
    repo = InMemoryWorkerInstanceRepository()
    worker, bound, worker_id, _owner = _running_with_worker(store, repo)
    configure_worker_execution_certified_owner_loss(worker_instance_repository=repo)
    certifier = build_worker_execution_certified_owner_loss_certifier(store=store)
    lost = replace(
        worker,
        lifecycle_state=WorkerLifecycleState.STOPPED,
        updated_at=datetime.now(timezone.utc),
    )
    repo.replace(lost, expected_revision=worker.revision)
    staged = certifier.stage_certified_owner_loss(
        tenant_id=_TENANT,
        operation_id=_OP,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
    )
    assert staged.staged_certified_owner_loss is not None
    evidence_id = staged.staged_certified_owner_loss.evidence.evidence_id
    assert (
        staged.staged_certified_owner_loss.evidence.issuer_id
        == CANONICAL_WORKER_EXECUTION_OWNER_LOSS_ISSUER_ID
    )
    unknown = recover_running_operation_after_certified_owner_loss(
        store,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=staged.revision,
        evidence_id=evidence_id,
    )
    assert unknown.physical_state is ExternalOperationPhysicalState.UNKNOWN
    assert unknown.intent_state is ExternalOperationIntentState.TERMINATING
    assert unknown.owner_token is None
    admit_unknown_external_operation_for_reconciliation(
        store,
        tenant_id=_TENANT,
        operation_id=_OP,
    )


def test_l3r7_worker_alive_denies_certification() -> None:
    store = InMemoryExternalOperationStateStore()
    repo = InMemoryWorkerInstanceRepository()
    _worker, bound, _worker_id, _owner = _running_with_worker(store, repo)
    configure_worker_execution_certified_owner_loss(worker_instance_repository=repo)
    certifier = build_worker_execution_certified_owner_loss_certifier(store=store)
    with pytest.raises(CertifiedOwnerLossEvidenceRejectedError):
        certifier.stage_certified_owner_loss(
            tenant_id=_TENANT,
            operation_id=_OP,
            loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
        )
    loaded = store.load(_TENANT, _OP)
    assert loaded is not None
    assert loaded.revision == bound.revision


def test_l3r7_end_to_end_stage_and_apply_helper() -> None:
    store = InMemoryExternalOperationStateStore()
    repo = InMemoryWorkerInstanceRepository()
    worker, _bound, _worker_id, _owner = _running_with_worker(store, repo)
    configure_worker_execution_certified_owner_loss(worker_instance_repository=repo)
    certifier = build_worker_execution_certified_owner_loss_certifier(store=store)
    repo.replace(
        replace(
            worker,
            lifecycle_state=WorkerLifecycleState.RECOVERING,
            updated_at=datetime.now(timezone.utc),
        ),
        expected_revision=worker.revision,
    )
    result = certifier.stage_and_apply_certified_owner_loss(
        tenant_id=_TENANT,
        operation_id=_OP,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
    )
    assert result.physical_state is ExternalOperationPhysicalState.UNKNOWN


def test_l3r7_misconfigured_composition_fail_closed() -> None:
    store = InMemoryExternalOperationStateStore()
    with pytest.raises(CertifiedOwnerLossIssuerNotConfiguredError):
        build_worker_execution_certified_owner_loss_certifier(store=store)
