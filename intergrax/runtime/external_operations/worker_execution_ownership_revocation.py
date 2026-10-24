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
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from intergrax.contracts.certified_owner_loss_evidence import CertifiedOwnerLossReason
from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationNotFoundError,
    ExternalOperationPhysicalState,
)
from intergrax.contracts.worker_execution_ownership_revocation import (
    CANONICAL_WORKER_EXECUTION_REVOCATION_SUPERVISION_AUTHORITY_ID,
    WorkerExecutionDirectRevocationInvocationError,
    WorkerExecutionOwnershipNotRevokedError,
    WorkerExecutionOwnershipRevocationAuthorizationRejectedError,
    WorkerExecutionOwnershipRevocationAuthorizationV1,
    WorkerExecutionOwnershipRevocationPort,
    WorkerExecutionOwnershipRevocationSupervisionNotConfiguredError,
    WorkerExecutionOwnershipRevocationSupervisionPort,
    WorkerExecutionOwnershipRevocationUnauthorizedError,
    WorkerExecutionOwnershipRevocationV1,
    compute_worker_execution_ownership_revocation_authorization_digest,
    worker_execution_revocation_proof_ref,
)
from intergrax.runtime.external_operations.worker_execution_ownership_revocation_trust import (
    require_worker_execution_ownership_revocation_trust,
    verify_worker_execution_ownership_revocation_authorization_mac,
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


_recorder_invocation_flag: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "worker_execution_recorder_revocation_invocation",
    default=False,
)


@contextlib.contextmanager
def _recorder_revocation_invocation():
    token = _recorder_invocation_flag.set(True)
    try:
        yield
    finally:
        _recorder_invocation_flag.reset(token)


def _require_recorder_revocation_invocation() -> None:
    if not _recorder_invocation_flag.get():
        raise WorkerExecutionDirectRevocationInvocationError(
            "use WorkerExecutionOwnershipRevocationRecorder",
        )


@runtime_checkable
class ColocatedWorkerExecutionOwnershipRevocationPort(
    WorkerExecutionOwnershipRevocationPort,
    Protocol,
):
    """Revocation port colocated with the authoritative external-operation store."""

    @property
    def authoritative_state_store(self) -> ExternalOperationStateStore:
        ...


def verify_worker_execution_ownership_revocation_authorization(
    authorization: WorkerExecutionOwnershipRevocationAuthorizationV1,
    *,
    tenant_id: str,
    operation_id: str,
    worker_instance_id: str,
    ownership_epoch: int,
    bound_store_revision: int,
) -> None:
    expected_digest = compute_worker_execution_ownership_revocation_authorization_digest(
        authorization,
    )
    if authorization.authorization_digest != expected_digest:
        raise WorkerExecutionOwnershipRevocationAuthorizationRejectedError(
            "revocation authorization digest mismatch",
        )
    if authorization.tenant_id != tenant_id:
        raise WorkerExecutionOwnershipRevocationAuthorizationRejectedError(
            "authorization tenant mismatch",
        )
    if authorization.operation_id != operation_id:
        raise WorkerExecutionOwnershipRevocationAuthorizationRejectedError(
            "authorization operation mismatch",
        )
    if authorization.worker_instance_id != worker_instance_id:
        raise WorkerExecutionOwnershipRevocationAuthorizationRejectedError(
            "authorization worker_instance_id mismatch",
        )
    if authorization.ownership_epoch != ownership_epoch:
        raise WorkerExecutionOwnershipRevocationAuthorizationRejectedError(
            "authorization ownership_epoch mismatch",
        )
    if authorization.bound_store_revision != bound_store_revision:
        raise WorkerExecutionOwnershipRevocationAuthorizationRejectedError(
            "stale revocation authorization for store revision",
        )
    if authorization.supervision_authority_id != (
        CANONICAL_WORKER_EXECUTION_REVOCATION_SUPERVISION_AUTHORITY_ID
    ):
        raise WorkerExecutionOwnershipRevocationAuthorizationRejectedError(
            "revocation supervision authority mismatch",
        )
    now = _utc_now()
    if authorization.valid_until < now:
        raise WorkerExecutionOwnershipRevocationAuthorizationRejectedError(
            "revocation authorization expired",
        )
    if authorization.authorized_at > now:
        raise WorkerExecutionOwnershipRevocationAuthorizationRejectedError(
            "revocation authorization not yet valid",
        )
    trust = require_worker_execution_ownership_revocation_trust()
    verify_worker_execution_ownership_revocation_authorization_mac(
        authorization,
        trust=trust,
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


@dataclass(frozen=True, slots=True)
class WorkerExecutionOwnershipRevocationAuthorizer:
    """Mint typed revocation decisions from supervision + authoritative store binding."""

    authoritative_state_store: ExternalOperationStateStore
    supervision: WorkerExecutionOwnershipRevocationSupervisionPort

    def authorize_revocation_for_bound_operation(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        loss_reason: CertifiedOwnerLossReason = CertifiedOwnerLossReason.WORKER_LOST,
    ) -> WorkerExecutionOwnershipRevocationAuthorizationV1:
        composition = require_certified_owner_loss_composition()
        if composition.worker_execution_revocation_authorizer is not self:
            raise WorkerExecutionOwnershipRevocationUnauthorizedError(
                "revocation authorizer not registered in composition",
            )
        if composition.worker_execution_revocation_supervision is not self.supervision:
            raise WorkerExecutionOwnershipRevocationUnauthorizedError(
                "revocation supervision port not registered in composition",
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
        return self.supervision.authorize_worker_execution_ownership_revocation(
            tenant_id=tenant_id,
            operation_id=operation_id,
            worker_instance_id=record.worker_instance_id,
            ownership_epoch=record.worker_ownership_epoch,
            bound_store_revision=record.revision,
            loss_reason=loss_reason,
        )


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

    @property
    def authoritative_state_store(self) -> ExternalOperationStateStore:
        return self._authoritative_state_store

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
        authorization: WorkerExecutionOwnershipRevocationAuthorizationV1,
    ) -> WorkerExecutionOwnershipRevocationV1:
        _require_recorder_revocation_invocation()
        key = (tenant_id, operation_id, worker_instance_id, ownership_epoch)
        with self._lock:
            existing = self._records.get(key)
            if existing is not None:
                return existing
            record = load_and_assert_authoritative_worker_execution_revocation_allowed(
                self._authoritative_state_store,
                tenant_id=tenant_id,
                operation_id=operation_id,
                worker_instance_id=worker_instance_id,
                ownership_epoch=ownership_epoch,
            )
            verify_worker_execution_ownership_revocation_authorization(
                authorization,
                tenant_id=tenant_id,
                operation_id=operation_id,
                worker_instance_id=worker_instance_id,
                ownership_epoch=ownership_epoch,
                bound_store_revision=record.revision,
            )
            revocation_id = uuid.uuid4().hex
            proof = worker_execution_revocation_proof_ref(
                tenant_id=tenant_id,
                operation_id=operation_id,
                worker_instance_id=worker_instance_id,
                ownership_epoch=ownership_epoch,
                revocation_id=revocation_id,
            )
            stored = WorkerExecutionOwnershipRevocationV1(
                tenant_id=tenant_id,
                operation_id=operation_id,
                worker_instance_id=worker_instance_id,
                ownership_epoch=ownership_epoch,
                revocation_id=revocation_id,
                revoked_at=_utc_now(),
                proof_ref=proof,
            )
            self._records[key] = stored
            return stored


@dataclass(frozen=True, slots=True)
class WorkerExecutionOwnershipRevocationRecorder:
    """Sanctioned revocation initiator — supervision decision + durable store apply."""

    authoritative_state_store: ExternalOperationStateStore
    revocation_port: WorkerExecutionOwnershipRevocationPort
    authorizer: WorkerExecutionOwnershipRevocationAuthorizer

    def record_revocation_for_bound_operation(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        loss_reason: CertifiedOwnerLossReason = CertifiedOwnerLossReason.WORKER_LOST,
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
        authorization = self.authorizer.authorize_revocation_for_bound_operation(
            tenant_id=tenant_id,
            operation_id=operation_id,
            loss_reason=loss_reason,
        )
        record = self.authoritative_state_store.load(tenant_id, operation_id)
        if record is None:
            raise ExternalOperationNotFoundError(operation_id)
        if record.worker_instance_id is None or record.worker_ownership_epoch is None:
            raise WorkerExecutionOwnershipRevocationUnauthorizedError(
                "worker execution ownership is not bound on authoritative record",
            )
        with _recorder_revocation_invocation():
            return self.revocation_port.revoke_worker_execution_ownership(
                tenant_id=tenant_id,
                operation_id=operation_id,
                worker_instance_id=record.worker_instance_id,
                ownership_epoch=record.worker_ownership_epoch,
                authorization=authorization,
            )


def build_worker_execution_ownership_revocation_recorder(
    *,
    authoritative_state_store: ExternalOperationStateStore,
    revocation_port: WorkerExecutionOwnershipRevocationPort | None = None,
    authorizer: WorkerExecutionOwnershipRevocationAuthorizer | None = None,
) -> WorkerExecutionOwnershipRevocationRecorder:
    composition = require_certified_owner_loss_composition()
    port = revocation_port or composition.worker_execution_ownership_revocation
    if port is None:
        raise WorkerExecutionOwnershipRevocationUnauthorizedError(
            "worker execution ownership revocation port is not configured",
        )
    resolved_authorizer = authorizer or composition.worker_execution_revocation_authorizer
    if resolved_authorizer is None:
        raise WorkerExecutionOwnershipRevocationSupervisionNotConfiguredError(
            "worker execution ownership revocation authorizer is not configured",
        )
    return WorkerExecutionOwnershipRevocationRecorder(
        authoritative_state_store=authoritative_state_store,
        revocation_port=port,
        authorizer=resolved_authorizer,
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


def resolve_colocated_authoritative_state_store(
    revocation_port: WorkerExecutionOwnershipRevocationPort,
    *,
    explicit: ExternalOperationStateStore | None,
) -> ExternalOperationStateStore:
    if explicit is not None:
        return explicit
    if isinstance(revocation_port, ColocatedWorkerExecutionOwnershipRevocationPort):
        return revocation_port.authoritative_state_store
    raise WorkerExecutionOwnershipRevocationUnauthorizedError(
        "authoritative external-operation state store is not configured for revocation",
    )


__all__ = [
    "ColocatedWorkerExecutionOwnershipRevocationPort",
    "InMemoryWorkerExecutionOwnershipRevocationStore",
    "WorkerExecutionOwnershipRevocationAuthorizer",
    "WorkerExecutionOwnershipRevocationRecorder",
    "assert_authoritative_worker_execution_revocation_allowed",
    "build_worker_execution_ownership_revocation_recorder",
    "load_and_assert_authoritative_worker_execution_revocation_allowed",
    "require_worker_execution_revocation_proof",
    "resolve_colocated_authoritative_state_store",
    "verify_worker_execution_ownership_revocation_authorization",
]
