# © Artur Czarnecki. All rights reserved.

"""Gate A — in-memory CAS serializes revocation vs mark_physical_running."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.contracts.certified_owner_loss_evidence import (
    CertifiedOwnerLossEvidenceRejectedError,
)
from intergrax.contracts.external_operation_cancellation import ExternalOperationPhysicalState
from intergrax.contracts.persistence_topology import PersistenceTopology
from intergrax.runtime.external_operations.external_operation_ownership import (
    ExternalOperationFencedCommitRejectedError,
    ProcessLocalExternalOperationOwner,
    mark_physical_running,
)
from intergrax.runtime.external_operations.tool_life_durable_single_host_composition import (
    compose_tool_life_durable_single_host_certified_owner_loss,
)
from intergrax.runtime.external_operations.worker_execution_certified_owner_loss import (
    bind_worker_execution_ownership,
)
from tests.unit.runtime.external_operations.tool_life_x_qualification_revocation_supervision import (
    QualificationWorkerExecutionCommitAuthorityLossPolicy,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-cas-race"
_OP = "op-cas-race"
_EPOCH = 3
_MAC = "INTERGRAX_WORKER_EXECUTION_REVOCATION_MAC_KEY"


def test_inmemory_mark_physical_running_denied_after_revocation_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Durable store CAS path must reject stale physical admission once epoch is revoked."""
    monkeypatch.setenv(_MAC, "cas-race-mac")
    composed = compose_tool_life_durable_single_host_certified_owner_loss(
        db_path=tmp_path / "cas_race.db",
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
    worker_id = "worker-cas-race"
    bound = bind_worker_execution_ownership(
        store,
        tenant_id=_TENANT,
        operation_id=_OP,
        expected_revision=running.revision,
        worker_instance_id=worker_id,
        worker_ownership_epoch=_EPOCH,
    )
    composed.worker_revocation.record_revocation_after_commit_authority_loss(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id=worker_id,
        ownership_epoch=_EPOCH,
    )
    stale = ProcessLocalExternalOperationOwner.mint()
    with pytest.raises(
        (CertifiedOwnerLossEvidenceRejectedError, ExternalOperationFencedCommitRejectedError),
    ):
        mark_physical_running(
            store,
            tenant_id=_TENANT,
            operation_id=_OP,
            expected_revision=bound.revision,
            owner=stale,
        )
    loaded = store.load(_TENANT, _OP)
    assert loaded is not None
    assert loaded.physical_state is ExternalOperationPhysicalState.RUNNING
    proof = composed.bundle.worker_execution_ownership_revocation.authoritative_revocation_proof_ref(
        tenant_id=_TENANT,
        operation_id=_OP,
        worker_instance_id=worker_id,
        ownership_epoch=_EPOCH,
    )
    assert proof is not None
