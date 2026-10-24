# © Artur Czarnecki. All rights reserved.

"""TOOL-LIFE-X L3-R10 — canonical commit-authority-loss decision and host integration."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from intergrax.contracts.certified_owner_loss_evidence import CertifiedOwnerLossReason
from intergrax.contracts.persistence_topology import PersistenceTopology
from intergrax.contracts.worker_execution_commit_authority_loss_decision import (
    WorkerExecutionCommitAuthorityLossDecisionDirectInvocationError,
    WorkerExecutionCommitAuthorityLossDecisionNotFoundError,
    WorkerExecutionCommitAuthorityLossDecisionRejectedError,
    WorkerExecutionCommitAuthorityLossDecisionV1,
)
from intergrax.contracts.worker_execution_ownership_revocation import (
    WorkerExecutionDirectRevocationInvocationError,
    WorkerExecutionOwnershipRevocationUnauthorizedError,
)
from intergrax.runtime.external_operations.certified_owner_loss_composition import (
    reset_certified_owner_loss_composition_for_tests,
)
from intergrax.runtime.external_operations.execution_recovery_worker_execution_revocation_supervision import (
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
    reset_worker_execution_ownership_revocation_trust_for_tests,
)
from tests.unit.runtime.external_operations.tool_life_x_revocation_test_forgery import (
    forged_worker_execution_ownership_revocation_authorization,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant_l3r10"
_OP = "op-l3r10"
_EPOCH = 42
_MAC_ENV = "INTERGRAX_WORKER_EXECUTION_REVOCATION_MAC_KEY"


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    reset_certified_owner_loss_composition_for_tests()
    reset_worker_execution_ownership_revocation_trust_for_tests()
    yield
    reset_certified_owner_loss_composition_for_tests()
    reset_worker_execution_ownership_revocation_trust_for_tests()


def _running_bound_operation(tmp_path: Path, *, worker_id: str = "worker-r10"):
    os.environ[_MAC_ENV] = "production-mac-key-l3r10"
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
        worker_instance_id=worker_id,
        worker_ownership_epoch=_EPOCH,
    )
    return composed, worker_id


def test_l3r10_production_composition_records_decision_and_revocation(tmp_path: Path) -> None:
    composed, worker_id = _running_bound_operation(tmp_path)
    decision = composed.worker_revocation.decide_commit_authority_loss_for_bound_operation(
        tenant_id=_TENANT,
        operation_id=_OP,
    )
    assert decision.ownership_epoch == _EPOCH
    proof = composed.worker_revocation.record_revocation_after_commit_authority_loss(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id=worker_id,
        ownership_epoch=_EPOCH,
    ).proof_ref
    assert proof
    os.environ.pop(_MAC_ENV, None)


def test_l3r10_direct_decision_store_write_rejected(tmp_path: Path) -> None:
    composed, _ = _running_bound_operation(tmp_path)
    port = composed.bundle.worker_execution_ownership_revocation
    when = datetime.now(timezone.utc)
    forged = WorkerExecutionCommitAuthorityLossDecisionV1(
        decision_id="forged",
        decision_authority_id="forged-authority",
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id="worker-r10",
        ownership_epoch=_EPOCH,
        bound_store_revision=3,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
        decided_at=when,
        valid_until=when + timedelta(minutes=5),
        decision_digest="bad",
    )
    with pytest.raises(WorkerExecutionCommitAuthorityLossDecisionDirectInvocationError):
        port.record_commit_authority_loss_decision(decision=forged)
    os.environ.pop(_MAC_ENV, None)


def test_l3r10_revocation_without_durable_decision_rejected(tmp_path: Path) -> None:
    composed, worker_id = _running_bound_operation(tmp_path)
    store = composed.bundle.external_operation_state
    record = store.load(_TENANT, _OP)
    assert record is not None
    forged = forged_worker_execution_ownership_revocation_authorization(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id=worker_id,
        ownership_epoch=_EPOCH,
        bound_store_revision=record.revision,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
    )
    with pytest.raises(WorkerExecutionDirectRevocationInvocationError):
        composed.bundle.worker_execution_ownership_revocation.revoke_worker_execution_ownership(
            tenant_id=_TENANT,
            operation_id=_OP,
            worker_instance_id=worker_id,
            ownership_epoch=_EPOCH,
            authorization=forged,
        )
    os.environ.pop(_MAC_ENV, None)


def test_l3r10_fail_closed_policy_blocks_even_with_decision(tmp_path: Path) -> None:
    os.environ[_MAC_ENV] = "production-mac-key-l3r10-fail-closed"
    composed = compose_tool_life_durable_single_host_certified_owner_loss(
        db_path=tmp_path / "tool_life.db",
        persistence_topology=PersistenceTopology.DURABLE_SINGLE_HOST,
        loss_policy=FailClosedWorkerExecutionCommitAuthorityLossPolicy(),
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
        worker_instance_id="worker-r10",
        worker_ownership_epoch=_EPOCH,
    )
    composed.commit_authority_loss_decision_authority.issue_for_bound_operation(
        tenant_id=_TENANT,
        operation_id=_OP,
        loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
    )
    from intergrax.runtime.external_operations.certified_owner_loss_composition import (
        require_certified_owner_loss_composition,
    )

    authorizer = require_certified_owner_loss_composition().worker_execution_revocation_authorizer
    assert authorizer is not None
    with pytest.raises(WorkerExecutionOwnershipRevocationUnauthorizedError):
        authorizer.authorize_revocation_for_bound_operation(
            tenant_id=_TENANT,
            operation_id=_OP,
        )
    os.environ.pop(_MAC_ENV, None)


def test_l3r10_stale_decision_revision_rejected(tmp_path: Path) -> None:
    composed, worker_id = _running_bound_operation(tmp_path)
    decision = composed.worker_revocation.decide_commit_authority_loss_for_bound_operation(
        tenant_id=_TENANT,
        operation_id=_OP,
    )
    port = composed.bundle.worker_execution_ownership_revocation
    with pytest.raises(WorkerExecutionCommitAuthorityLossDecisionRejectedError):
        from intergrax.runtime.external_operations.worker_execution_commit_authority_loss import (
            require_authoritative_commit_authority_loss_decision,
        )

        require_authoritative_commit_authority_loss_decision(
            port,
            tenant_id=_TENANT,
            operation_id=_OP,
            worker_instance_id=worker_id,
            ownership_epoch=_EPOCH,
            bound_store_revision=decision.bound_store_revision + 1,
            loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
        )
    os.environ.pop(_MAC_ENV, None)


def test_l3r10_shared_multi_host_topology_fail_closed() -> None:
    with pytest.raises(ToolLifeDurableTopologyNotSupportedError):
        compose_tool_life_durable_single_host_certified_owner_loss(
            db_path=Path("ignored.db"),
            persistence_topology=PersistenceTopology.SHARED_MULTI_HOST,
        )


def test_l3r10_host_wiring_materializes_composition(tmp_path: Path) -> None:
    from intergrax.applications._shared.tool_life_execution_recovery_wiring import (
        wire_tool_life_durable_single_host_execution_recovery,
    )

    os.environ[_MAC_ENV] = "production-mac-key-l3r10-host"
    wiring = wire_tool_life_durable_single_host_execution_recovery(
        required_persistence_topology=PersistenceTopology.DURABLE_SINGLE_HOST,
        db_path=tmp_path / "host_tool_life.db",
    )
    assert wiring is not None
    assert wiring.composition.commit_authority_loss_decision_authority is not None
    os.environ.pop(_MAC_ENV, None)


def test_l3r10_idempotent_decision_and_revocation(tmp_path: Path) -> None:
    composed, worker_id = _running_bound_operation(tmp_path)
    first = composed.worker_revocation.record_revocation_after_commit_authority_loss(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id=worker_id,
        ownership_epoch=_EPOCH,
    )
    second = composed.worker_revocation.record_revocation_after_commit_authority_loss(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id=worker_id,
        ownership_epoch=_EPOCH,
    )
    assert first.proof_ref == second.proof_ref
    os.environ.pop(_MAC_ENV, None)


def test_l3r10_missing_decision_at_store_boundary(tmp_path: Path) -> None:
    composed, worker_id = _running_bound_operation(tmp_path)
    port = composed.bundle.worker_execution_ownership_revocation
    with pytest.raises(WorkerExecutionCommitAuthorityLossDecisionNotFoundError):
        from intergrax.runtime.external_operations.worker_execution_commit_authority_loss import (
            require_authoritative_commit_authority_loss_decision,
        )

        require_authoritative_commit_authority_loss_decision(
            port,
            tenant_id=_TENANT,
            operation_id=_OP,
            worker_instance_id=worker_id,
            ownership_epoch=_EPOCH,
            bound_store_revision=99,
            loss_reason=CertifiedOwnerLossReason.WORKER_LOST,
        )
    os.environ.pop(_MAC_ENV, None)
