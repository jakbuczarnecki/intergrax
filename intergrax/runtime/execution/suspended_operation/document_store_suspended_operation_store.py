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
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedExecutionOperationDescriptor,
)
from intergrax.contracts.execution.suspended_operation.store import (
    SuspendedExecutionOperationStore,
)
from intergrax.contracts.governed_continuation_correlation import (
    GovernedContinuationCorrelation,
)
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
)
from intergrax.integrations.contracts.document_store_process_durability import (
    document_store_survives_process_restart,
)
from intergrax.runtime.execution.suspended_operation.store_engine import (
    SuspendedOperationBackingStore,
)

_DURABLE_SCHEMA_V1 = "suspended_operation_backing.v1"
_PARTITION = "execution.suspended_operations"
_ROW_KEY = "backing"


class DocumentStoreSuspendedExecutionOperationStore(SuspendedExecutionOperationStore):
    """Durable provider — descriptors survive process restart via document store."""

    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "suspended operation persistence requires ConditionalDocumentStore",
            )
        self._document_store = document_store
        self._lock = threading.Lock()
        self._backing = SuspendedOperationBackingStore()
        self._load_from_document()

    @property
    def is_durable(self) -> bool:
        return document_store_survives_process_restart(self._document_store)

    def _load_from_document(self) -> None:
        record = self._document_store.get(_PARTITION, _ROW_KEY)
        if record is None:
            return
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
        self._backing.replace_all(restored)

    def _persist(self) -> None:
        snapshot = {
            "schema_version": _DURABLE_SCHEMA_V1,
            "records": {
                key: descriptor.model_dump(mode="json")
                for key, descriptor in self._backing.snapshot().items()
            },
        }
        replacement = DocumentRecord(
            partition_key=_PARTITION,
            row_key=_ROW_KEY,
            data={"backing": copy.deepcopy(snapshot)},
        )
        existing = self._document_store.get(_PARTITION, _ROW_KEY)
        if existing is None:
            if not self._document_store.put_if_absent(replacement):
                raise RuntimeError("suspended operation durable persist race")
            return
        if not self._document_store.replace_if_match(
            expected=existing,
            replacement=replacement,
        ):
            self._load_from_document()
            raise RuntimeError("suspended operation durable persist stale")

    def _mutate(self, operation):
        with self._lock:
            result = operation(self._backing)
            self._persist()
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
    ) -> SuspendedOperationMutationResult:
        return self._mutate(
            lambda backing: backing.mark_consumed(
                suspended_operation_id=suspended_operation_id,
                expected_materialization_revision=expected_materialization_revision,
                owner_id=owner_id,
                fence=fence,
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
    ) -> SuspendedOperationMutationResult:
        return self._mutate(
            lambda backing: backing.abandon(
                suspended_operation_id=suspended_operation_id,
                expected_materialization_revision=expected_materialization_revision,
                reason=reason,
                owner_id=owner_id,
                fence=fence,
            ),
        )


def reconnect_document_store_suspended_operation_store(
    document_store: ConditionalDocumentStore,
) -> DocumentStoreSuspendedExecutionOperationStore:
    """Process B: new store object over existing durable backing."""
    return DocumentStoreSuspendedExecutionOperationStore(document_store)


__all__ = [
    "DocumentStoreSuspendedExecutionOperationStore",
    "reconnect_document_store_suspended_operation_store",
]
