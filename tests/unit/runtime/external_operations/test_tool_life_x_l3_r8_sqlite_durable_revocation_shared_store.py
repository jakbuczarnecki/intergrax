# © Artur Czarnecki. All rights reserved.

"""TOOL-LIFE-X L3-R8 — SQLite durable revocation + shared external-operation store."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.autonomous_work.in_memory_repository import InMemoryWorkerInstanceRepository
from intergrax.contracts.certified_owner_loss_evidence import (
    CANONICAL_WORKER_EXECUTION_OWNER_LOSS_ISSUER_ID,
    CertifiedOwnerLossEvidenceRejectedError,
    CertifiedOwnerLossIssuerNotConfiguredError,
    CertifiedOwnerLossReason,
)
from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationIntentState,
    ExternalOperationPhysicalState,
)
from intergrax.contracts.persistence_topology import PersistenceTopology
from intergrax.runtime.external_operations.certified_owner_loss_composition import (
    reset_certified_owner_loss_composition_for_tests,
)
from intergrax.runtime.external_operations.external_operation_ownership import (
    ProcessLocalExternalOperationOwner,
    mark_physical_running,
    recover_running_operation_after_certified_owner_loss,
)
from intergrax.runtime.external_operations.external_operation_state_store import (
    StaleExternalOperationStateError,
)
from intergrax.runtime.external_operations.sqlite_tool_life_durable_stores import (
    SqliteToolLifeDurableStoresBundle,
    SQLiteExternalOperationStateStore,
    SQLiteWorkerExecutionOwnershipRevocationStore,
)
from intergrax.runtime.external_operations.worker_execution_certified_owner_loss import (
    bind_worker_execution_ownership,
    build_worker_execution_certified_owner_loss_certifier,
    configure_worker_execution_certified_owner_loss,
)
from tests.unit.autonomous_work.repository_contracts import worker_instance

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant_l3r8"
_OP = "op-l3r8"
_BOUND_EPOCH = 11


@pytest.fixture(autouse=True)
def _reset_composition() -> None:
    reset_certified_owner_loss_composition_for_tests()
    yield
    reset_certified_owner_loss_composition_for_tests()


@pytest.fixture
def durable_bundle(tmp_path: Path) -> SqliteToolLifeDurableStoresBundle:
    return SqliteToolLifeDurableStoresBundle(db_path=tmp_path / "tool_life_durable.db")


def _running_with_worker(
    bundle: SqliteToolLifeDurableStoresBundle,
    repo: InMemoryWorkerInstanceRepository,
    *,
    ownership_epoch: int = _BOUND_EPOCH,
) -> tuple[object, object, str]:
    store = bundle.external_operation_state
    revocation = bundle.worker_execution_ownership_revocation
    worker = worker_instance()
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
        worker_ownership_epoch=ownership_epoch,
    )
    configure_worker_execution_certified_owner_loss(
        worker_execution_ownership_revocation=revocation,
        worker_instance_repository=repo,
    )
    return worker, bound, str(worker.worker_instance_id)


def test_l3r8_bundle_topology_durable_single_host(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    assert durable_bundle.persistence_topology is PersistenceTopology.DURABLE_SINGLE_HOST
    assert (
        durable_bundle.external_operation_state.persistence_topology
        is PersistenceTopology.DURABLE_SINGLE_HOST
    )
    assert (
        durable_bundle.worker_execution_ownership_revocation.persistence_topology
        is PersistenceTopology.DURABLE_SINGLE_HOST
    )


def test_l3r8_positive_revoke_stage_apply_survives_store_reopen(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
    tmp_path: Path,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    _worker, bound, worker_id = _running_with_worker(durable_bundle, repo)
    proof = durable_bundle.worker_execution_ownership_revocation.revoke_worker_execution_ownership(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id=worker_id,
        ownership_epoch=_BOUND_EPOCH,
    ).proof_ref
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
    reopened = SqliteToolLifeDurableStoresBundle(db_path=durable_bundle.db_path)
    unknown = recover_running_operation_after_certified_owner_loss(
        reopened.external_operation_state,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=staged.revision,
        evidence_id=evidence_id,
    )
    assert unknown.physical_state is ExternalOperationPhysicalState.UNKNOWN
    assert unknown.intent_state is ExternalOperationIntentState.TERMINATING
    persisted = reopened.external_operation_state.load(_TENANT, _OP)
    assert persisted is not None
    assert persisted.revision == unknown.revision
    assert evidence_id in persisted.consumed_certified_loss_evidence_ids


def test_l3r8_idempotent_revocation_record(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    _worker, _bound, worker_id = _running_with_worker(durable_bundle, repo)
    port = durable_bundle.worker_execution_ownership_revocation
    first = port.revoke_worker_execution_ownership(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id=worker_id,
        ownership_epoch=_BOUND_EPOCH,
    )
    second = port.revoke_worker_execution_ownership(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id=worker_id,
        ownership_epoch=_BOUND_EPOCH,
    )
    assert second.proof_ref == first.proof_ref
    assert second.revocation_id == first.revocation_id


def test_l3r8_staging_without_revocation_rejected(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    _running_with_worker(durable_bundle, repo)
    certifier = build_worker_execution_certified_owner_loss_certifier(
        store=durable_bundle.external_operation_state,
    )
    with pytest.raises(CertifiedOwnerLossEvidenceRejectedError):
        certifier.stage_certified_owner_loss(
            tenant_id=_TENANT,
            operation_id=_OP,
            loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
        )


def test_l3r8_wrong_epoch_revocation_denied(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    _worker, _bound, worker_id = _running_with_worker(durable_bundle, repo)
    durable_bundle.worker_execution_ownership_revocation.revoke_worker_execution_ownership(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id=worker_id,
        ownership_epoch=_BOUND_EPOCH + 1,
    )
    certifier = build_worker_execution_certified_owner_loss_certifier(
        store=durable_bundle.external_operation_state,
    )
    with pytest.raises(CertifiedOwnerLossEvidenceRejectedError):
        certifier.stage_certified_owner_loss(
            tenant_id=_TENANT,
            operation_id=_OP,
            loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
        )


def test_l3r8_cross_tenant_revocation_not_visible(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    _worker, _bound, worker_id = _running_with_worker(durable_bundle, repo)
    durable_bundle.worker_execution_ownership_revocation.revoke_worker_execution_ownership(
        tenant_id="other-tenant",
        operation_id=_OP,
        worker_instance_id=worker_id,
        ownership_epoch=_BOUND_EPOCH,
    )
    proof = durable_bundle.worker_execution_ownership_revocation.authoritative_revocation_proof_ref(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id=worker_id,
        ownership_epoch=_BOUND_EPOCH,
    )
    assert proof is None


def test_l3r8_concurrent_cas_stale_revision(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    store = durable_bundle.external_operation_state
    record = store.create_if_absent(_TENANT, "op-cas")
    owner = ProcessLocalExternalOperationOwner.mint()
    running = mark_physical_running(
        store,
        tenant_id=_TENANT,
        operation_id="op-cas",
        expected_revision=record.revision,
        owner=owner,
    )
    with pytest.raises(StaleExternalOperationStateError):
        mark_physical_running(
            store,
            tenant_id=_TENANT,
            operation_id="op-cas",
            expected_revision=record.revision,
            owner=owner,
        )
    assert running.revision == record.revision + 1


def test_l3r8_separate_connection_revocation_visibility(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    """WAL-backed SQLite: independent connection observes committed revocation."""
    repo = InMemoryWorkerInstanceRepository()
    _worker, _bound, worker_id = _running_with_worker(durable_bundle, repo)
    proof = durable_bundle.worker_execution_ownership_revocation.revoke_worker_execution_ownership(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id=worker_id,
        ownership_epoch=_BOUND_EPOCH,
    ).proof_ref
    peer = SQLiteWorkerExecutionOwnershipRevocationStore(
        db_path=durable_bundle.db_path,
    )
    peer_proof = peer.authoritative_revocation_proof_ref(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id=worker_id,
        ownership_epoch=_BOUND_EPOCH,
    )
    assert peer_proof == proof


def test_l3r8_separate_connection_external_operation_state(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    _running_with_worker(durable_bundle, repo)
    peer = SQLiteExternalOperationStateStore(db_path=durable_bundle.db_path)
    loaded = peer.load(_TENANT, _OP)
    assert loaded is not None
    assert loaded.physical_state is ExternalOperationPhysicalState.RUNNING


def test_l3r8_issuer_allowlist_required_for_stage(
    durable_bundle: SqliteToolLifeDurableStoresBundle,
) -> None:
    repo = InMemoryWorkerInstanceRepository()
    _worker, _bound, worker_id = _running_with_worker(durable_bundle, repo)
    durable_bundle.worker_execution_ownership_revocation.revoke_worker_execution_ownership(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id=worker_id,
        ownership_epoch=_BOUND_EPOCH,
    )
    reset_certified_owner_loss_composition_for_tests()
    with pytest.raises(CertifiedOwnerLossIssuerNotConfiguredError):
        build_worker_execution_certified_owner_loss_certifier(
            store=durable_bundle.external_operation_state,
        ).stage_certified_owner_loss(
            tenant_id=_TENANT,
            operation_id=_OP,
            loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
        )


def test_l3r8_canonical_issuer_id_unchanged() -> None:
    assert (
        CANONICAL_WORKER_EXECUTION_OWNER_LOSS_ISSUER_ID
        == "tool-life-x.worker-execution-ownership.v1"
    )
