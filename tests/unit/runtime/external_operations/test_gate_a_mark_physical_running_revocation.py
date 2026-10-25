# © Artur Czarnecki. All rights reserved.

"""G-A02 — stale worker cannot start physical effect after epoch revocation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

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

_TENANT = "tenant-ga02"
_OP = "op-ga02"
_EPOCH = 1
_MAC = "INTERGRAX_WORKER_EXECUTION_REVOCATION_MAC_KEY"


def test_mark_physical_running_rejects_revoked_epoch(tmp_path: Path) -> None:
    os.environ[_MAC] = "ga02-mac-key"
    composed = compose_tool_life_durable_single_host_certified_owner_loss(
        db_path=tmp_path / "ga02.db",
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
    worker_id = "worker-ga02"
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
    stale_owner = ProcessLocalExternalOperationOwner.mint()
    with pytest.raises(ExternalOperationFencedCommitRejectedError, match="revoked"):
        mark_physical_running(
            store,
            tenant_id=_TENANT,
            operation_id=_OP,
            expected_revision=bound.revision,
            owner=stale_owner,
        )
    os.environ.pop(_MAC, None)
