# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Durable external operation state with revision CAS (W4-C + L3-R6 tenant scope)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from intergrax.contracts.certified_owner_loss_evidence import (
    CertifiedOwnerLossAuthorityBypassError,
    CertifiedOwnerLossEvidenceRejectedError,
    CertifiedOwnerLossIssuerNotConfiguredError,
    StagedCertifiedOwnerLossEvidenceV1,
)
from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationIntentState,
    ExternalOperationNotFoundError,
    ExternalOperationPhysicalState,
    ExternalOperationState,
)
from intergrax.runtime.external_operations.certified_owner_loss_composition import (
    current_certified_owner_loss_issuer_binding,
)


class StaleExternalOperationStateError(RuntimeError):
    """CAS conflict on external operation durable state."""

    def __init__(
        self,
        *,
        operation_id: str,
        expected_revision: int,
        actual_revision: int | None,
    ) -> None:
        self.operation_id = operation_id
        self.expected_revision = expected_revision
        self.actual_revision = actual_revision
        super().__init__(
            f"stale external operation state for {operation_id!r}: "
            f"expected revision {expected_revision}, actual {actual_revision!r}",
        )


class ExternalOperationLegacyTenantBindingError(RuntimeError):
    """Durable record lacks proven tenant binding — shared-store access forbidden."""


class ExternalOperationTenantScopeViolationError(RuntimeError):
    """Tenant-scoped store access does not match durable record tenant."""


@dataclass(frozen=True, slots=True)
class ExternalOperationStateUpdate:
    intent_state: ExternalOperationIntentState | None = None
    physical_state: ExternalOperationPhysicalState | None = None
    owner_token: str | None = None
    clear_owner_token: bool = False
    ownership_epoch_fence: int | None = None
    staged_certified_owner_loss: StagedCertifiedOwnerLossEvidenceV1 | None = None
    clear_staged_certified_owner_loss: bool = False
    consumed_certified_loss_evidence_ids: frozenset[str] | None = None


def _validate_tenant_id(tenant_id: str) -> str:
    if type(tenant_id) is not str or not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty str")
    return tenant_id


def _store_key(tenant_id: str, operation_id: str) -> tuple[str, str]:
    return (_validate_tenant_id(tenant_id), operation_id)


def _require_tenant_bound(record: ExternalOperationState) -> None:
    if type(record.tenant_id) is not str or not record.tenant_id.strip():
        raise ExternalOperationLegacyTenantBindingError(
            f"external operation {record.operation_id!r} lacks tenant binding",
        )


def _update_touches_certification_authority(
    update: ExternalOperationStateUpdate,
) -> bool:
    return (
        update.ownership_epoch_fence is not None
        or update.staged_certified_owner_loss is not None
        or update.clear_staged_certified_owner_loss
        or update.consumed_certified_loss_evidence_ids is not None
    )


@runtime_checkable
class ExternalOperationStateStore(Protocol):
    """Process-local or shared durable store for external operation lifecycle."""

    def load(self, tenant_id: str, operation_id: str) -> ExternalOperationState | None:
        ...

    def create_if_absent(
        self,
        tenant_id: str,
        operation_id: str,
        *,
        owner_token: str | None = None,
    ) -> ExternalOperationState:
        ...

    def compare_and_set(
        self,
        tenant_id: str,
        operation_id: str,
        *,
        expected_revision: int,
        update: ExternalOperationStateUpdate,
    ) -> ExternalOperationState:
        ...

    def list_running(self, tenant_id: str) -> tuple[ExternalOperationState, ...]:
        ...

    def stage_certified_owner_loss_evidence(
        self,
        tenant_id: str,
        operation_id: str,
        *,
        expected_revision: int,
        staged: StagedCertifiedOwnerLossEvidenceV1,
    ) -> ExternalOperationState:
        ...

    def apply_certified_owner_loss(
        self,
        tenant_id: str,
        operation_id: str,
        *,
        expected_revision: int,
        evidence_id: str,
    ) -> ExternalOperationState:
        ...


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class InMemoryExternalOperationStateStore:
    """Thread-safe in-memory store for tests and process-local reconciliation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[tuple[str, str], ExternalOperationState] = {}

    def _get_scoped(
        self,
        tenant_id: str,
        operation_id: str,
    ) -> ExternalOperationState | None:
        if type(operation_id) is not str or not operation_id:
            raise ValueError("operation_id must be a non-empty str")
        key = _store_key(tenant_id, operation_id)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return None
            _require_tenant_bound(record)
            if record.tenant_id != key[0]:
                return None
            return record

    def load(self, tenant_id: str, operation_id: str) -> ExternalOperationState | None:
        return self._get_scoped(tenant_id, operation_id)

    def create_if_absent(
        self,
        tenant_id: str,
        operation_id: str,
        *,
        owner_token: str | None = None,
    ) -> ExternalOperationState:
        tid = _validate_tenant_id(tenant_id)
        if type(operation_id) is not str or not operation_id:
            raise ValueError("operation_id must be a non-empty str")
        now = _utc_now()
        key = (tid, operation_id)
        with self._lock:
            existing = self._records.get(key)
            if existing is not None:
                _require_tenant_bound(existing)
                return existing
            record = ExternalOperationState(
                operation_id=operation_id,
                tenant_id=tid,
                intent_state=ExternalOperationIntentState.ACTIVE,
                physical_state=ExternalOperationPhysicalState.NOT_STARTED,
                created_at=now,
                updated_at=now,
                revision=1,
                owner_token=owner_token,
            )
            self._records[key] = record
            return record

    def compare_and_set(
        self,
        tenant_id: str,
        operation_id: str,
        *,
        expected_revision: int,
        update: ExternalOperationStateUpdate,
    ) -> ExternalOperationState:
        if _update_touches_certification_authority(update):
            raise CertifiedOwnerLossAuthorityBypassError(
                "certification authority fields require stage_certified_owner_loss_evidence "
                "or apply_certified_owner_loss",
            )
        return self._compare_and_set_locked(
            tenant_id,
            operation_id,
            expected_revision=expected_revision,
            update=update,
        )

    def _compare_and_set_locked(
        self,
        tenant_id: str,
        operation_id: str,
        *,
        expected_revision: int,
        update: ExternalOperationStateUpdate,
    ) -> ExternalOperationState:
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("expected_revision must be int >= 1")
        tid = _validate_tenant_id(tenant_id)
        now = _utc_now()
        key = (tid, operation_id)
        with self._lock:
            current = self._records.get(key)
            if current is None:
                raise ExternalOperationNotFoundError(operation_id)
            _require_tenant_bound(current)
            if current.tenant_id != tid:
                raise ExternalOperationTenantScopeViolationError(
                    f"tenant scope mismatch for {operation_id!r}",
                )
            if current.revision != expected_revision:
                raise StaleExternalOperationStateError(
                    operation_id=operation_id,
                    expected_revision=expected_revision,
                    actual_revision=current.revision,
                )
            intent = (
                update.intent_state
                if update.intent_state is not None
                else current.intent_state
            )
            physical = (
                update.physical_state
                if update.physical_state is not None
                else current.physical_state
            )
            owner = current.owner_token
            if update.clear_owner_token:
                owner = None
            elif update.owner_token is not None:
                owner = update.owner_token
            fence = (
                update.ownership_epoch_fence
                if update.ownership_epoch_fence is not None
                else current.ownership_epoch_fence
            )
            staged = current.staged_certified_owner_loss
            if update.clear_staged_certified_owner_loss:
                staged = None
            elif update.staged_certified_owner_loss is not None:
                staged = update.staged_certified_owner_loss
            consumed = current.consumed_certified_loss_evidence_ids
            if update.consumed_certified_loss_evidence_ids is not None:
                consumed = update.consumed_certified_loss_evidence_ids
            next_record = ExternalOperationState(
                operation_id=operation_id,
                tenant_id=current.tenant_id,
                intent_state=intent,
                physical_state=physical,
                created_at=current.created_at,
                updated_at=now,
                revision=current.revision + 1,
                owner_token=owner,
                ownership_epoch_fence=fence,
                staged_certified_owner_loss=staged,
                consumed_certified_loss_evidence_ids=consumed,
            )
            self._records[key] = next_record
            return next_record

    def list_running(self, tenant_id: str) -> tuple[ExternalOperationState, ...]:
        tid = _validate_tenant_id(tenant_id)
        with self._lock:
            return tuple(
                record
                for record in self._records.values()
                if record.tenant_id == tid
                and record.physical_state is ExternalOperationPhysicalState.RUNNING
            )

    def stage_certified_owner_loss_evidence(
        self,
        tenant_id: str,
        operation_id: str,
        *,
        expected_revision: int,
        staged: StagedCertifiedOwnerLossEvidenceV1,
    ) -> ExternalOperationState:
        binding = current_certified_owner_loss_issuer_binding()
        if staged.staged_by_issuer_id not in binding.allowed_issuer_ids:
            raise CertifiedOwnerLossIssuerNotConfiguredError(
                "certified owner-loss issuer binding is not configured",
            )
        current = self._get_scoped(tenant_id, operation_id)
        if current is None:
            raise ExternalOperationNotFoundError(operation_id)
        if current.physical_state is not ExternalOperationPhysicalState.RUNNING:
            raise CertifiedOwnerLossEvidenceRejectedError(
                "staging requires physical RUNNING",
            )
        evidence = staged.evidence
        if evidence.epoch.tenant_id != current.tenant_id:
            raise CertifiedOwnerLossEvidenceRejectedError("epoch tenant mismatch")
        if evidence.epoch.operation_id != current.operation_id:
            raise CertifiedOwnerLossEvidenceRejectedError("epoch operation mismatch")
        if evidence.epoch.store_revision != expected_revision:
            raise CertifiedOwnerLossEvidenceRejectedError("ownership epoch revision mismatch")
        if evidence.epoch.owner_token != (current.owner_token or ""):
            raise CertifiedOwnerLossEvidenceRejectedError("ownership epoch owner mismatch")
        if evidence.issuer_id not in binding.allowed_issuer_ids:
            raise CertifiedOwnerLossEvidenceRejectedError("issuer not allowlisted")
        occupied = current.staged_certified_owner_loss
        if occupied is not None and occupied.evidence.evidence_id != evidence.evidence_id:
            raise CertifiedOwnerLossEvidenceRejectedError("staged slot occupied")
        return self._compare_and_set_locked(
            tenant_id,
            operation_id,
            expected_revision=expected_revision,
            update=ExternalOperationStateUpdate(
                staged_certified_owner_loss=staged,
            ),
        )

    def apply_certified_owner_loss(
        self,
        tenant_id: str,
        operation_id: str,
        *,
        expected_revision: int,
        evidence_id: str,
    ) -> ExternalOperationState:
        binding = current_certified_owner_loss_issuer_binding()
        if not binding.allowed_issuer_ids:
            raise CertifiedOwnerLossIssuerNotConfiguredError(
                "certified owner-loss issuer binding is not configured",
            )
        current = self._get_scoped(tenant_id, operation_id)
        if current is None:
            raise ExternalOperationNotFoundError(operation_id)
        if (
            current.physical_state is ExternalOperationPhysicalState.UNKNOWN
            and evidence_id in current.consumed_certified_loss_evidence_ids
        ):
            return current
        staged_slot = current.staged_certified_owner_loss
        if staged_slot is None:
            raise CertifiedOwnerLossEvidenceRejectedError("no staged evidence")
        evidence = staged_slot.evidence
        if evidence.evidence_id != evidence_id:
            raise CertifiedOwnerLossEvidenceRejectedError("evidence_id mismatch")
        if evidence_id in current.consumed_certified_loss_evidence_ids:
            raise CertifiedOwnerLossEvidenceRejectedError("evidence already consumed")
        if current.physical_state is not ExternalOperationPhysicalState.RUNNING:
            raise CertifiedOwnerLossEvidenceRejectedError("apply requires physical RUNNING")
        if evidence.epoch.tenant_id != current.tenant_id:
            raise CertifiedOwnerLossEvidenceRejectedError("cross-tenant evidence")
        if evidence.epoch.owner_token != (current.owner_token or ""):
            raise CertifiedOwnerLossEvidenceRejectedError("owner-token mismatch")
        if evidence.epoch.store_revision + 1 != expected_revision:
            raise CertifiedOwnerLossEvidenceRejectedError(
                "apply revision must follow staged ownership epoch revision",
            )
        if evidence.issuer_id not in binding.allowed_issuer_ids:
            raise CertifiedOwnerLossEvidenceRejectedError("issuer not allowlisted")
        next_fence = (current.ownership_epoch_fence or 0) + 1
        if evidence.monotonic_fence < next_fence:
            raise CertifiedOwnerLossEvidenceRejectedError("monotonic fence regression")
        consumed = frozenset(
            set(current.consumed_certified_loss_evidence_ids) | {evidence_id},
        )
        return self._compare_and_set_locked(
            tenant_id,
            operation_id,
            expected_revision=expected_revision,
            update=ExternalOperationStateUpdate(
                physical_state=ExternalOperationPhysicalState.UNKNOWN,
                intent_state=ExternalOperationIntentState.TERMINATING,
                ownership_epoch_fence=evidence.monotonic_fence,
                clear_staged_certified_owner_loss=True,
                consumed_certified_loss_evidence_ids=consumed,
                clear_owner_token=True,
            ),
        )
