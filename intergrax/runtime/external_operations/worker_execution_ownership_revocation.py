# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Worker execution ownership revocation (reference + authoritative binding guards)."""

from __future__ import annotations

import contextlib
import contextvars
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationNotFoundError,
    ExternalOperationPhysicalState,
)
from intergrax.contracts.worker_execution_ownership_revocation import (
    WorkerExecutionDirectRevocationInvocationError,
    WorkerExecutionOwnershipNotRevokedError,
    WorkerExecutionOwnershipRevocationPort,
    WorkerExecutionOwnershipRevocationUnauthorizedError,
    WorkerExecutionOwnershipRevocationV1,
    worker_execution_revocation_proof_ref,
)
from intergrax.runtime.external_operations.certified_owner_loss_composition import (
    require_certified_owner_loss_composition,
)

if TYPE_CHECKING:
    from intergrax.contracts.external_operation_cancellation import ExternalOperationState
    from intergrax.runtime.external_operations.external_operation_state_store import (
        ExternalOperationStateStore,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


_revocation_write_sanctioned: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "worker_execution_revocation_write_sanctioned",
    default=False,
)


@contextlib.contextmanager
def _sanctioned_revocation_write():
    token = _revocation_write_sanctioned.set(True)
    try:
        yield
    finally:
        _revocation_write_sanctioned.reset(token)


def _require_sanctioned_revocation_write() -> None:
    if not _revocation_write_sanctioned.get():
        raise WorkerExecutionDirectRevocationInvocationError(
            "use WorkerExecutionOwnershipRevocationRecorder",
        )


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
        _require_sanctioned_revocation_write()
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


@dataclass(frozen=True, slots=True)
class WorkerExecutionOwnershipRevocationRecorder:
    """Sanctioned revocation initiator — derives worker/epoch from authoritative store."""

    authoritative_state_store: ExternalOperationStateStore
    revocation_port: WorkerExecutionOwnershipRevocationPort

    def record_revocation_for_bound_operation(
        self,
        *,
        tenant_id: str,
        operation_id: str,
    ) -> WorkerExecutionOwnershipRevocationV1:
        composition = require_certified_owner_loss_composition()
        if composition.worker_execution_ownership_revocation is not self.revocation_port:
            raise WorkerExecutionOwnershipRevocationUnauthorizedError(
                "revocation port not registered in composition",
            )
        if composition.worker_execution_revocation_recorder is not self:
            raise WorkerExecutionOwnershipRevocationUnauthorizedError(
                "revocation recorder not registered in composition",
            )
        record = self.authoritative_state_store.load(tenant_id, operation_id)
        if record is None:
            raise ExternalOperationNotFoundError(operation_id)
        if record.worker_instance_id is None or record.worker_ownership_epoch is None:
            raise WorkerExecutionOwnershipRevocationUnauthorizedError(
                "worker execution ownership is not bound on authoritative record",
            )
        assert_authoritative_worker_execution_revocation_allowed(
            record,
            tenant_id=tenant_id,
            operation_id=operation_id,
            worker_instance_id=record.worker_instance_id,
            ownership_epoch=record.worker_ownership_epoch,
        )
        with _sanctioned_revocation_write():
            return self.revocation_port.revoke_worker_execution_ownership(
                tenant_id=tenant_id,
                operation_id=operation_id,
                worker_instance_id=record.worker_instance_id,
                ownership_epoch=record.worker_ownership_epoch,
            )


def build_worker_execution_ownership_revocation_recorder(
    *,
    authoritative_state_store: ExternalOperationStateStore,
    revocation_port: WorkerExecutionOwnershipRevocationPort | None = None,
) -> WorkerExecutionOwnershipRevocationRecorder:
    composition = require_certified_owner_loss_composition()
    port = revocation_port or composition.worker_execution_ownership_revocation
    if port is None:
        raise WorkerExecutionOwnershipRevocationUnauthorizedError(
            "worker execution ownership revocation port is not configured",
        )
    return WorkerExecutionOwnershipRevocationRecorder(
        authoritative_state_store=authoritative_state_store,
        revocation_port=port,
    )


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
    "WorkerExecutionOwnershipRevocationRecorder",
    "assert_authoritative_worker_execution_revocation_allowed",
    "build_worker_execution_ownership_revocation_recorder",
    "load_and_assert_authoritative_worker_execution_revocation_allowed",
    "require_worker_execution_revocation_proof",
]
