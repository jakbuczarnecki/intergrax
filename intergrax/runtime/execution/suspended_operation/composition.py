# © Artur Czarnecki. All rights reserved.

"""Composition boundary for suspended execution operations (UCA-6C-R6-R1)."""

from __future__ import annotations

from intergrax.contracts.execution.suspended_operation.codec import (
    SuspendedOperationCodecRegistry,
)
from intergrax.contracts.execution.suspended_operation.store import (
    SuspendedExecutionOperationStore,
)
from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
from intergrax.runtime.execution.suspended_operation.codec_registry import (
    DefaultSuspendedOperationCodecRegistry,
)
from intergrax.runtime.execution.suspended_operation.document_store_suspended_operation_store import (
    DocumentStoreSuspendedExecutionOperationStore,
)
from intergrax.runtime.execution.suspended_operation.in_memory_store import (
    InMemorySuspendedExecutionOperationStore,
)


class SuspendedOperationCompositionError(RuntimeError):
    """Fail closed when production suspended-operation wiring is invalid."""


def wire_suspended_execution_operation_store(
    *,
    document_store: ConditionalDocumentStore | None = None,
) -> SuspendedExecutionOperationStore:
    if document_store is not None:
        return DocumentStoreSuspendedExecutionOperationStore(document_store)
    return InMemorySuspendedExecutionOperationStore()


def wire_default_suspended_operation_codec_registry() -> (
    SuspendedOperationCodecRegistry
):
    return DefaultSuspendedOperationCodecRegistry()


def validate_suspended_operation_store_for_production(
    store: SuspendedExecutionOperationStore,
) -> None:
    if not store.is_durable:
        raise SuspendedOperationCompositionError(
            "production continuation path requires durable SuspendedExecutionOperationStore",
        )


__all__ = [
    "SuspendedOperationCompositionError",
    "validate_suspended_operation_store_for_production",
    "wire_default_suspended_operation_codec_registry",
    "wire_suspended_execution_operation_store",
]
