# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-memory worker execution ownership revocation (reference / tests)."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone

from intergrax.contracts.worker_execution_ownership_revocation import (
    WorkerExecutionOwnershipNotRevokedError,
    WorkerExecutionOwnershipRevocationPort,
    WorkerExecutionOwnershipRevocationV1,
    worker_execution_revocation_proof_ref,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class InMemoryWorkerExecutionOwnershipRevocationStore(
    WorkerExecutionOwnershipRevocationPort,
):
    """Process-local revocation ledger — not cross-store durability evidence."""

    def __init__(self) -> None:
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
        with self._lock:
            if key in self._records:
                return self._records[key]
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
    "require_worker_execution_revocation_proof",
]
