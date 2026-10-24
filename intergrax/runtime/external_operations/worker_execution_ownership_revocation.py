# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Worker execution ownership revocation (reference + authoritative binding guards)."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationNotFoundError,
    ExternalOperationPhysicalState,
)
from intergrax.contracts.worker_execution_ownership_revocation import (
    WorkerExecutionOwnershipNotRevokedError,
    WorkerExecutionOwnershipRevocationPort,
    WorkerExecutionOwnershipRevocationUnauthorizedError,
    WorkerExecutionOwnershipRevocationV1,
    worker_execution_revocation_proof_ref,
)

if TYPE_CHECKING:
    from intergrax.contracts.external_operation_cancellation import ExternalOperationState
    from intergrax.runtime.external_operations.external_operation_state_store import (
        ExternalOperationStateStore,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def assert_authoritative_worker_execution_revocation_allowed(
    record: ExternalOperationState,
    *,
    tenant_id: str,
    operation_id: str,
    worker_instance_id: str,
    ownership_epoch: int,
) -> None:
    """Require durable RUNNING ownership binding before recording revocation."""
    if record.tenant_id != tenant_id:
        raise WorkerExecutionOwnershipRevocationUnauthorizedError(
            "revocation tenant does not match authoritative ownership record",
        )
    if record.operation_id != operation_id:
        raise WorkerExecutionOwnershipRevocationUnauthorizedError(
            "revocation operation does not match authoritative ownership record",
        )
    if record.physical_state is not ExternalOperationPhysicalState.RUNNING:
        raise WorkerExecutionOwnershipRevocationUnauthorizedError(
            "revocation requires physical RUNNING ownership record",
        )
    if record.worker_instance_id is None or record.worker_ownership_epoch is None:
        raise WorkerExecutionOwnershipRevocationUnauthorizedError(
            "worker execution ownership is not bound on authoritative record",
        )
    if record.worker_instance_id != worker_instance_id:
        raise WorkerExecutionOwnershipRevocationUnauthorizedError(
            "revocation worker_instance_id does not match authoritative binding",
        )
    if record.worker_ownership_epoch != ownership_epoch:
        raise WorkerExecutionOwnershipRevocationUnauthorizedError(
            "revocation ownership_epoch does not match authoritative binding",
        )


def load_and_assert_authoritative_worker_execution_revocation_allowed(
    store: ExternalOperationStateStore,
    *,
    tenant_id: str,
    operation_id: str,
    worker_instance_id: str,
    ownership_epoch: int,
) -> ExternalOperationState:
    record = store.load(tenant_id, operation_id)
    if record is None:
        raise ExternalOperationNotFoundError(operation_id)
    assert_authoritative_worker_execution_revocation_allowed(
        record,
        tenant_id=tenant_id,
        operation_id=operation_id,
        worker_instance_id=worker_instance_id,
        ownership_epoch=ownership_epoch,
    )
    return record


class InMemoryWorkerExecutionOwnershipRevocationStore(
    WorkerExecutionOwnershipRevocationPort,
):
    """Process-local revocation ledger colocated with an authoritative state store."""

    def __init__(
        self,
        *,
        authoritative_state_store: ExternalOperationStateStore,
    ) -> None:
        self._authoritative_state_store = authoritative_state_store
        self._lock = threading.Lock()
        self._records: dict[
            tuple[str, str, str, int],
            WorkerExecutionOwnershipRevocationV1,
        ] = {}

    def authoritative_revocation_proof_ref(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        worker_instance_id: str,
        ownership_epoch: int,
    ) -> str | None:
        key = (tenant_id, operation_id, worker_instance_id, ownership_epoch)
        with self._lock:
            record = self._records.get(key)
            return record.proof_ref if record is not None else None

    def revoke_worker_execution_ownership(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        worker_instance_id: str,
        ownership_epoch: int,
    ) -> WorkerExecutionOwnershipRevocationV1:
        key = (tenant_id, operation_id, worker_instance_id, ownership_epoch)
        with self._lock:
            existing = self._records.get(key)
            if existing is not None:
                return existing
            load_and_assert_authoritative_worker_execution_revocation_allowed(
                self._authoritative_state_store,
                tenant_id=tenant_id,
                operation_id=operation_id,
                worker_instance_id=worker_instance_id,
                ownership_epoch=ownership_epoch,
            )
            revocation_id = uuid.uuid4().hex
            proof = worker_execution_revocation_proof_ref(
                tenant_id=tenant_id,
                operation_id=operation_id,
                worker_instance_id=worker_instance_id,
                ownership_epoch=ownership_epoch,
                revocation_id=revocation_id,
            )
            record = WorkerExecutionOwnershipRevocationV1(
                tenant_id=tenant_id,
                operation_id=operation_id,
                worker_instance_id=worker_instance_id,
                ownership_epoch=ownership_epoch,
                revocation_id=revocation_id,
                revoked_at=_utc_now(),
                proof_ref=proof,
            )
            self._records[key] = record
            return record


def require_worker_execution_revocation_proof(
    port: WorkerExecutionOwnershipRevocationPort,
    *,
    tenant_id: str,
    operation_id: str,
    worker_instance_id: str,
    ownership_epoch: int,
) -> str:
    proof = port.authoritative_revocation_proof_ref(
        tenant_id=tenant_id,
        operation_id=operation_id,
        worker_instance_id=worker_instance_id,
        ownership_epoch=ownership_epoch,
    )
    if proof is None:
        raise WorkerExecutionOwnershipNotRevokedError(
            "worker execution ownership epoch is not authoritatively revoked",
        )
    return proof


__all__ = [
    "InMemoryWorkerExecutionOwnershipRevocationStore",
    "assert_authoritative_worker_execution_revocation_allowed",
    "load_and_assert_authoritative_worker_execution_revocation_allowed",
    "require_worker_execution_revocation_proof",
]
