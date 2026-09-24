# © Artur Czarnecki. All rights reserved.

"""ConditionalDocumentStore-backed suspended operation store (UCA-6C-R6-R1)."""

from __future__ import annotations

import copy
import threading
from datetime import datetime

from intergrax.contracts.execution_continuation import PendingExecutionContinuation
from intergrax.contracts.execution.suspended_operation.claim import (
    SuspendedOperationAbandonReason,
    SuspendedOperationClaimResult,
    SuspendedOperationMutationResult,
)
from intergrax.contracts.execution.suspended_operation.authority_scope import (
    SuspendedOperationAuthorityScope,
)
from intergrax.contracts.agent_governance_hitl import LogicalInvocationFingerprint
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedExecutionOperationDescriptor,
)
from intergrax.contracts.execution.suspended_operation.persistence_conflict import (
    SuspendedOperationPersistenceConflictError,
)
from intergrax.contracts.execution.suspended_operation.store import (
    SuspendedExecutionOperationStore,
)
from intergrax.contracts.governed_continuation_correlation import (
    GovernedContinuationCorrelation,
)
from intergrax.contracts.execution_deadline.clock import UtcClockPort
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
)
from intergrax.integrations.contracts.document_store_process_durability import (
    document_store_survives_process_restart,
)
from intergrax.runtime.execution.deadline_authority.system_clocks import SystemUtcClock
from intergrax.runtime.execution.suspended_operation.store_engine import (
    SuspendedOperationBackingStore,
)

_DURABLE_SCHEMA_V1 = "suspended_operation_backing.v1"
_PARTITION = "execution.suspended_operations"
_ROW_KEY = "backing"


class DocumentStoreSuspendedExecutionOperationStore(SuspendedExecutionOperationStore):
    """Durable provider — descriptors survive process restart via document store."""

    def __init__(
        self,
        document_store: ConditionalDocumentStore,
        *,
        utc_clock: UtcClockPort | None = None,
    ) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "suspended operation persistence requires ConditionalDocumentStore",
            )
        self._document_store = document_store
        self._utc_clock = utc_clock if utc_clock is not None else SystemUtcClock()
        self._lock = threading.Lock()
        self._backing = SuspendedOperationBackingStore(utc_clock=self._utc_clock)
        self._load_from_document()

    @property
    def is_durable(self) -> bool:
        return document_store_survives_process_restart(self._document_store)

    def _load_from_document(self) -> None:
        record = self._document_store.get(_PARTITION, _ROW_KEY)
        self._backing = self._backing_from_record(record)

    def _backing_from_record(
        self,
        record: DocumentRecord | None,
    ) -> SuspendedOperationBackingStore:
        backing = SuspendedOperationBackingStore(utc_clock=self._utc_clock)
        if record is None:
            return backing
        payload = record.data.get("backing")
        if not isinstance(payload, dict):
            raise RuntimeError("corrupt suspended operation durable backing")
        if payload.get("schema_version") != _DURABLE_SCHEMA_V1:
            raise RuntimeError("unknown suspended operation durable schema")
        raw_records = payload.get("records")
        if not isinstance(raw_records, dict):
            raise RuntimeError("corrupt suspended operation durable records")
        restored: dict[str, SuspendedExecutionOperationDescriptor] = {}
        for key, value in raw_records.items():
            restored[str(key)] = SuspendedExecutionOperationDescriptor.model_validate(
                value
            )
        backing.replace_all(restored)
        return backing

    def _serialize_backing(
        self,
        backing: SuspendedOperationBackingStore,
    ) -> DocumentRecord:
        snapshot = {
            "schema_version": _DURABLE_SCHEMA_V1,
            "records": {
                key: descriptor.model_dump(mode="json")
                for key, descriptor in backing.snapshot().items()
            },
        }
        return DocumentRecord(
            partition_key=_PARTITION,
            row_key=_ROW_KEY,
            data={"backing": copy.deepcopy(snapshot)},
        )

    def _persist_snapshot(
        self,
        backing: SuspendedOperationBackingStore,
        *,
        expected_record: DocumentRecord | None,
    ) -> None:
        replacement = self._serialize_backing(backing)
        if expected_record is None:
            if not self._document_store.put_if_absent(replacement):
                self._load_from_document()
                raise SuspendedOperationPersistenceConflictError(
                    "suspended operation durable persist race",
                )
            return
        if not self._document_store.replace_if_match(
            expected=expected_record,
            replacement=replacement,
        ):
            self._load_from_document()
            raise SuspendedOperationPersistenceConflictError(
                "suspended operation durable persist stale",
            )

    def _mutate(self, operation):
        with self._lock:
            expected_record = self._document_store.get(_PARTITION, _ROW_KEY)
            backing = self._backing_from_record(expected_record)
            result = operation(backing)
            try:
                self._persist_snapshot(backing, expected_record=expected_record)
            except SuspendedOperationPersistenceConflictError:
                raise
            self._backing = backing
            return result

    def prepare(
        self,
        descriptor: SuspendedExecutionOperationDescriptor,
    ) -> SuspendedOperationMutationResult:
        return self._mutate(lambda backing: backing.prepare(descriptor))

    def block(
        self,
        *,
        suspended_operation_id: str,
        expected_materialization_revision: int,
        continuation: PendingExecutionContinuation,
        governed_correlation: GovernedContinuationCorrelation,
    ) -> SuspendedOperationMutationResult:
        return self._mutate(
            lambda backing: backing.block(
                suspended_operation_id=suspended_operation_id,
                expected_materialization_revision=expected_materialization_revision,
                continuation=continuation,
                governed_correlation=governed_correlation,
            ),
        )

    def load(
        self,
        suspended_operation_id: str,
    ) -> SuspendedExecutionOperationDescriptor | None:
        with self._lock:
            return self._backing.load(suspended_operation_id)

    def load_active_for_continuation(
        self,
        continuation_id: str,
    ) -> SuspendedExecutionOperationDescriptor | None:
        with self._lock:
            return self._backing.load_active_for_continuation(continuation_id)

    def load_materialized_for_continuation(
        self,
        continuation_id: str,
    ) -> SuspendedExecutionOperationDescriptor | None:
        with self._lock:
            return self._backing.load_materialized_for_continuation(continuation_id)

    def load_active_for_logical_invocation(
        self,
        logical_invocation_fingerprint: LogicalInvocationFingerprint,
    ) -> SuspendedExecutionOperationDescriptor | None:
        with self._lock:
            return self._backing.load_active_for_logical_invocation(
                logical_invocation_fingerprint,
            )

    def claim(
        self,
        *,
        suspended_operation_id: str,
        expected_materialization_revision: int,
        owner_id: str,
        lease_expires_at: datetime,
    ) -> SuspendedOperationClaimResult:
        return self._mutate(
            lambda backing: backing.claim(
                suspended_operation_id=suspended_operation_id,
                expected_materialization_revision=expected_materialization_revision,
                owner_id=owner_id,
                lease_expires_at=lease_expires_at,
            ),
        )

    def reclaim(
        self,
        *,
        suspended_operation_id: str,
        expected_materialization_revision: int,
        owner_id: str,
        lease_expires_at: datetime,
        expected_fence: int | None = None,
    ) -> SuspendedOperationMutationResult:
        return self._mutate(
            lambda backing: backing.reclaim(
                suspended_operation_id=suspended_operation_id,
                expected_materialization_revision=expected_materialization_revision,
                owner_id=owner_id,
                lease_expires_at=lease_expires_at,
                expected_fence=expected_fence,
            ),
        )

    def mark_consumed(
        self,
        *,
        suspended_operation_id: str,
        expected_materialization_revision: int,
        owner_id: str,
        fence: int,
        expected_pause_generation: int | None = None,
    ) -> SuspendedOperationMutationResult:
        return self._mutate(
            lambda backing: backing.mark_consumed(
                suspended_operation_id=suspended_operation_id,
                expected_materialization_revision=expected_materialization_revision,
                owner_id=owner_id,
                fence=fence,
                expected_pause_generation=expected_pause_generation,
            ),
        )

    def authority_reblock_from_claimed(
        self,
        *,
        suspended_operation_id: str,
        expected_materialization_revision: int,
        expected_pause_generation: int,
        expected_owner_id: str,
        expected_fence: int,
        next_pause_generation: int,
        next_continuation: PendingExecutionContinuation,
        next_governed_correlation: GovernedContinuationCorrelation,
        next_invocation_scope_id: str,
        next_authority_scope: SuspendedOperationAuthorityScope,
    ) -> SuspendedOperationMutationResult:
        return self._mutate(
            lambda backing: backing.authority_reblock_from_claimed(
                suspended_operation_id=suspended_operation_id,
                expected_materialization_revision=expected_materialization_revision,
                expected_pause_generation=expected_pause_generation,
                expected_owner_id=expected_owner_id,
                expected_fence=expected_fence,
                next_pause_generation=next_pause_generation,
                next_continuation=next_continuation,
                next_governed_correlation=next_governed_correlation,
                next_invocation_scope_id=next_invocation_scope_id,
                next_authority_scope=next_authority_scope,
            ),
        )

    def abandon(
        self,
        *,
        suspended_operation_id: str,
        expected_materialization_revision: int,
        reason: SuspendedOperationAbandonReason,
        owner_id: str | None = None,
        fence: int | None = None,
        expected_pause_generation: int | None = None,
    ) -> SuspendedOperationMutationResult:
        return self._mutate(
            lambda backing: backing.abandon(
                suspended_operation_id=suspended_operation_id,
                expected_materialization_revision=expected_materialization_revision,
                reason=reason,
                owner_id=owner_id,
                fence=fence,
                expected_pause_generation=expected_pause_generation,
            ),
        )


def reconnect_document_store_suspended_operation_store(
    document_store: ConditionalDocumentStore,
    *,
    utc_clock: UtcClockPort | None = None,
) -> DocumentStoreSuspendedExecutionOperationStore:
    """Process B: new store object over existing durable backing."""
    return DocumentStoreSuspendedExecutionOperationStore(
        document_store,
        utc_clock=utc_clock,
    )


__all__ = [
    "DocumentStoreSuspendedExecutionOperationStore",
    "reconnect_document_store_suspended_operation_store",
]
