# © Artur Czarnecki. All rights reserved.

"""TOOL-LIFE-X L3-R9 — authenticated revocation supervision and production composition."""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from intergrax.autonomous_work.in_memory_repository import InMemoryWorkerInstanceRepository
from intergrax.contracts.certified_owner_loss_evidence import CertifiedOwnerLossReason
from intergrax.contracts.persistence_topology import PersistenceTopology
from intergrax.contracts.worker_execution_ownership_revocation import (
    WorkerExecutionDirectRevocationInvocationError,
    WorkerExecutionOwnershipRevocationAuthorizationRejectedError,
    WorkerExecutionOwnershipRevocationUnauthorizedError,
    compute_worker_execution_ownership_revocation_authorization_digest,
)
from intergrax.runtime.external_operations.certified_owner_loss_composition import (
    reset_certified_owner_loss_composition_for_tests,
)
from intergrax.runtime.external_operations.execution_recovery_external_operation_worker_revocation import (
    ExecutionRecoveryExternalOperationWorkerRevocation,
)
from intergrax.runtime.external_operations.execution_recovery_worker_execution_revocation_supervision import (
    ExecutionRecoveryWorkerExecutionRevocationSupervision,
    FailClosedWorkerExecutionCommitAuthorityLossPolicy,
)
from intergrax.runtime.external_operations.external_operation_ownership import (
    ProcessLocalExternalOperationOwner,
    mark_physical_running,
)
from intergrax.runtime.external_operations.tool_life_durable_single_host_composition import (
    ToolLifeDurableTopologyNotSupportedError,
    compose_tool_life_durable_single_host_certified_owner_loss,
)
from intergrax.runtime.external_operations.worker_execution_certified_owner_loss import (
    bind_worker_execution_ownership,
)
from intergrax.runtime.external_operations.worker_execution_ownership_revocation_trust import (
    WorkerExecutionOwnershipRevocationTrustMaterial,
    bind_worker_execution_ownership_revocation_trust_for_qualification,
    compute_worker_execution_ownership_revocation_authorization_mac,
    reset_worker_execution_ownership_revocation_trust_for_tests,
)
from tests.unit.runtime.external_operations.tool_life_x_qualification_revocation_supervision import (
    QualificationWorkerExecutionCommitAuthorityLossPolicy,
    qualification_worker_execution_revocation_supervision,
)
from tests.unit.runtime.external_operations.tool_life_x_revocation_test_forgery import (
    forged_worker_execution_ownership_revocation_authorization,
)
from tests.unit.autonomous_work.repository_contracts import worker_instance

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant_l3r9"
_OP = "op-l3r9"
_BOUND_EPOCH = 19
_MAC_ENV = "INTERGRAX_WORKER_EXECUTION_REVOCATION_MAC_KEY"


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    reset_certified_owner_loss_composition_for_tests()
    reset_worker_execution_ownership_revocation_trust_for_tests()
    yield
    reset_certified_owner_loss_composition_for_tests()
    reset_worker_execution_ownership_revocation_trust_for_tests()


def test_l3r9_public_digest_forgery_rejected_at_store(tmp_path: Path) -> None:
    os.environ[_MAC_ENV] = "production-mac-key-l3r9"
    composed = compose_tool_life_durable_single_host_certified_owner_loss(
        db_path=tmp_path / "tool_life.db",
        persistence_topology=PersistenceTopology.DURABLE_SINGLE_HOST,
    )
    store = composed.bundle.external_operation_state
    owner = ProcessLocalExternalOperationOwner.mint()
    record = store.create_if_absent(_TENANT, _OP, owner_token=owner.token)
    running = mark_physical_running(
        store,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=record.revision,
        owner=owner,
    )
    bind_worker_execution_ownership(
        store,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=running.revision,
        worker_instance_id="worker-1",
        worker_ownership_epoch=_BOUND_EPOCH,
    )
    forged = forged_worker_execution_ownership_revocation_authorization(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id="worker-1",
        ownership_epoch=_BOUND_EPOCH,
        bound_store_revision=running.revision,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
    )
    with pytest.raises(WorkerExecutionDirectRevocationInvocationError):
        composed.bundle.worker_execution_ownership_revocation.revoke_worker_execution_ownership(
            tenant_id=_TENANT,
            operation_id=_OP,
            worker_instance_id="worker-1",
            ownership_epoch=_BOUND_EPOCH,
            authorization=forged,
        )
    os.environ.pop(_MAC_ENV, None)


def test_l3r9_fail_closed_without_admission_scope(tmp_path: Path) -> None:
    os.environ[_MAC_ENV] = "production-mac-key-l3r9-admission"
    composed = compose_tool_life_durable_single_host_certified_owner_loss(
        db_path=tmp_path / "tool_life.db",
        persistence_topology=PersistenceTopology.DURABLE_SINGLE_HOST,
    )
    store = composed.bundle.external_operation_state
    repo = InMemoryWorkerInstanceRepository()
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
    bind_worker_execution_ownership(
        store,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=running.revision,
        worker_instance_id=str(worker.worker_instance_id),
        worker_ownership_epoch=_BOUND_EPOCH,
    )
    with pytest.raises(WorkerExecutionOwnershipRevocationUnauthorizedError):
        composed.worker_revocation.record_revocation_after_commit_authority_loss(
            tenant_id=_TENANT,
            operation_id=_OP,
            worker_instance_id=str(worker.worker_instance_id),
            ownership_epoch=_BOUND_EPOCH,
        )
    os.environ.pop(_MAC_ENV, None)


def test_l3r9_qualification_loss_policy_records_revocation(tmp_path: Path) -> None:
    os.environ[_MAC_ENV] = "production-mac-key-l3r9-positive"
    composed = compose_tool_life_durable_single_host_certified_owner_loss(
        db_path=tmp_path / "tool_life.db",
        persistence_topology=PersistenceTopology.DURABLE_SINGLE_HOST,
        loss_policy=QualificationWorkerExecutionCommitAuthorityLossPolicy(),
    )
    store = composed.bundle.external_operation_state
    worker_id = "worker-positive"
    owner = ProcessLocalExternalOperationOwner.mint()
    record = store.create_if_absent(_TENANT, _OP, owner_token=owner.token)
    running = mark_physical_running(
        store,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=record.revision,
        owner=owner,
    )
    bind_worker_execution_ownership(
        store,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=running.revision,
        worker_instance_id=worker_id,
        worker_ownership_epoch=_BOUND_EPOCH,
    )
    proof = composed.worker_revocation.record_revocation_after_commit_authority_loss(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id=worker_id,
        ownership_epoch=_BOUND_EPOCH,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
    ).proof_ref
    assert proof


def test_l3r9_production_composition_fail_closed_despite_mac_key(tmp_path: Path) -> None:
    """R9-F01: MAC/trust alone must not enable positive revocation without loss policy."""
    os.environ[_MAC_ENV] = "production-mac-key-l3r9-no-admission-bypass"
    composed = compose_tool_life_durable_single_host_certified_owner_loss(
        db_path=tmp_path / "tool_life.db",
        persistence_topology=PersistenceTopology.DURABLE_SINGLE_HOST,
    )
    store = composed.bundle.external_operation_state
    owner = ProcessLocalExternalOperationOwner.mint()
    record = store.create_if_absent(_TENANT, _OP, owner_token=owner.token)
    running = mark_physical_running(
        store,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=record.revision,
        owner=owner,
    )
    bind_worker_execution_ownership(
        store,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=running.revision,
        worker_instance_id="worker-blocked",
        worker_ownership_epoch=_BOUND_EPOCH,
    )
    with pytest.raises(WorkerExecutionOwnershipRevocationUnauthorizedError):
        composed.worker_revocation.record_revocation_after_commit_authority_loss(
            tenant_id=_TENANT,
            operation_id=_OP,
            worker_instance_id="worker-blocked",
            ownership_epoch=_BOUND_EPOCH,
        )
    os.environ.pop(_MAC_ENV, None)


def test_l3r9_idempotent_revocation_repeat_returns_same_proof(tmp_path: Path) -> None:
    os.environ[_MAC_ENV] = "production-mac-key-l3r9-idempotent"
    composed = compose_tool_life_durable_single_host_certified_owner_loss(
        db_path=tmp_path / "tool_life.db",
        persistence_topology=PersistenceTopology.DURABLE_SINGLE_HOST,
        loss_policy=QualificationWorkerExecutionCommitAuthorityLossPolicy(),
    )
    store = composed.bundle.external_operation_state
    owner = ProcessLocalExternalOperationOwner.mint()
    record = store.create_if_absent(_TENANT, _OP, owner_token=owner.token)
    running = mark_physical_running(
        store,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=record.revision,
        owner=owner,
    )
    bind_worker_execution_ownership(
        store,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=running.revision,
        worker_instance_id="worker-idem",
        worker_ownership_epoch=_BOUND_EPOCH,
    )
    first = composed.worker_revocation.record_revocation_after_commit_authority_loss(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id="worker-idem",
        ownership_epoch=_BOUND_EPOCH,
    )
    second = composed.worker_revocation.record_revocation_after_commit_authority_loss(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id="worker-idem",
        ownership_epoch=_BOUND_EPOCH,
    )
    assert first.proof_ref == second.proof_ref
    assert first.revocation_id == second.revocation_id
    os.environ.pop(_MAC_ENV, None)
    os.environ.pop(_MAC_ENV, None)


def test_l3r9_unsupported_topology_fail_closed() -> None:
    with pytest.raises(ToolLifeDurableTopologyNotSupportedError):
        compose_tool_life_durable_single_host_certified_owner_loss(
            db_path=Path("ignored.db"),
            persistence_topology=PersistenceTopology.SHARED_MULTI_HOST,
            trust_material=WorkerExecutionOwnershipRevocationTrustMaterial(
                mac_key=b"test",
            ),
        )


def test_l3r9_expired_authorization_rejected() -> None:
    bind_worker_execution_ownership_revocation_trust_for_qualification()
    supervision = qualification_worker_execution_revocation_supervision()
    when = datetime.now(timezone.utc) - timedelta(hours=1)
    auth = supervision.authorize_worker_execution_ownership_revocation(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id="w",
        ownership_epoch=1,
        bound_store_revision=1,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
    )
    expired = auth.__class__(
        authorization_id=auth.authorization_id,
        supervision_authority_id=auth.supervision_authority_id,
        tenant_id=auth.tenant_id,
        operation_id=auth.operation_id,
        worker_instance_id=auth.worker_instance_id,
        ownership_epoch=auth.ownership_epoch,
        bound_store_revision=auth.bound_store_revision,
        loss_reason=auth.loss_reason,
        authorized_at=when,
        valid_until=when + timedelta(seconds=30),
        authorization_digest=compute_worker_execution_ownership_revocation_authorization_digest(
            auth.__class__(
                authorization_id=auth.authorization_id,
                supervision_authority_id=auth.supervision_authority_id,
                tenant_id=auth.tenant_id,
                operation_id=auth.operation_id,
                worker_instance_id=auth.worker_instance_id,
                ownership_epoch=auth.ownership_epoch,
                bound_store_revision=auth.bound_store_revision,
                loss_reason=auth.loss_reason,
                authorized_at=when,
                valid_until=when + timedelta(seconds=30),
                authorization_digest="pending",
                authorization_mac="pending",
            ),
        ),
        authorization_mac=auth.authorization_mac,
    )
    from intergrax.runtime.external_operations.worker_execution_ownership_revocation import (
        verify_worker_execution_ownership_revocation_authorization,
    )

    with pytest.raises(WorkerExecutionOwnershipRevocationAuthorizationRejectedError):
        verify_worker_execution_ownership_revocation_authorization(
            expired,
            tenant_id=_TENANT,
            operation_id=_OP,
            worker_instance_id="w",
            ownership_epoch=1,
            bound_store_revision=1,
        )


def test_l3r9_fail_closed_supervision_policy_blocks_positive_mint() -> None:
    bind_worker_execution_ownership_revocation_trust_for_qualification()
    supervision = ExecutionRecoveryWorkerExecutionRevocationSupervision(
        loss_policy=FailClosedWorkerExecutionCommitAuthorityLossPolicy(),
        trust=WorkerExecutionOwnershipRevocationTrustMaterial(
            mac_key=b"tool-life-x-qualification-revocation-mac-v1",
        ),
    )
    with pytest.raises(WorkerExecutionOwnershipRevocationUnauthorizedError):
        supervision.authorize_worker_execution_ownership_revocation(
            tenant_id=_TENANT,
            operation_id=_OP,
            worker_instance_id="w",
            ownership_epoch=1,
            bound_store_revision=1,
            loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
        )


def test_l3r9_recomputed_mac_without_trust_rejected() -> None:
    bind_worker_execution_ownership_revocation_trust_for_qualification()
    supervision = ExecutionRecoveryWorkerExecutionRevocationSupervision(
        loss_policy=QualificationWorkerExecutionCommitAuthorityLossPolicy(),
        trust=WorkerExecutionOwnershipRevocationTrustMaterial(
            mac_key=b"tool-life-x-qualification-revocation-mac-v1",
        ),
    )
    auth = supervision.authorize_worker_execution_ownership_revocation(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id="w",
        ownership_epoch=1,
        bound_store_revision=1,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
    )
    wrong_trust = WorkerExecutionOwnershipRevocationTrustMaterial(mac_key=b"wrong-key")
    bad_mac = compute_worker_execution_ownership_revocation_authorization_mac(
        auth,
        trust=wrong_trust,
    )
    tampered = replace(auth, authorization_mac=bad_mac)
    from intergrax.runtime.external_operations.worker_execution_ownership_revocation import (
        verify_worker_execution_ownership_revocation_authorization,
    )

    with pytest.raises(WorkerExecutionOwnershipRevocationAuthorizationRejectedError):
        verify_worker_execution_ownership_revocation_authorization(
            tampered,
            tenant_id=_TENANT,
            operation_id=_OP,
            worker_instance_id="w",
            ownership_epoch=1,
            bound_store_revision=1,
        )
