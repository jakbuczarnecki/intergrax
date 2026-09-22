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
from intergrax.integrations.contracts.document_store_process_durability import (
    document_store_survives_process_restart,
)
from intergrax.runtime.execution.suspended_operation.codec_registry import (
    DefaultSuspendedOperationCodecRegistry,
)
from intergrax.runtime.execution.suspended_operation.document_store_suspended_operation_store import (
    DocumentStoreSuspendedExecutionOperationStore,
)
from intergrax.runtime.execution.suspended_operation.in_memory_store import (
    InMemorySuspendedExecutionOperationStore,
)
from intergrax.runtime.execution.suspended_operation.reentry_coordinator import (
    ExecutionSuspendedWorkReentryCoordinator,
)
from intergrax.contracts.execution_continuation import ExecutionContinuationPort
from intergrax.runtime.nexus.tools.continuation_aware_catalog_tool_host import (
    ContinuationAwareCatalogToolHost,
)
from intergrax.runtime.nexus.tools.nexus_execution_bound_catalog_tool_invoker import (
    NexusExecutionBoundCatalogToolInvoker,
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


def validate_document_store_for_production_suspended_operations(
    document_store: ConditionalDocumentStore | None,
) -> None:
    if document_store is None:
        raise SuspendedOperationCompositionError(
            "production continuation path requires explicit durable document store",
        )
    if not document_store_survives_process_restart(document_store):
        raise SuspendedOperationCompositionError(
            "document store does not declare process-restart durability",
        )


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


def wire_execution_suspended_work_reentry_coordinator(
    *,
    store: SuspendedExecutionOperationStore,
    continuation_port: ExecutionContinuationPort,
    catalog_invoker: NexusExecutionBoundCatalogToolInvoker,
    catalog_host: ContinuationAwareCatalogToolHost,
    claim_owner_id: str,
) -> ExecutionSuspendedWorkReentryCoordinator:
    return ExecutionSuspendedWorkReentryCoordinator(
        store=store,
        continuation_port=continuation_port,
        catalog_host=catalog_host,
        catalog_invoker=catalog_invoker,
        codec_registry=wire_default_suspended_operation_codec_registry(),
        claim_owner_id=claim_owner_id,
    )


def validate_production_suspended_operation_wiring(
    *,
    document_store: ConditionalDocumentStore | None,
    store: SuspendedExecutionOperationStore,
) -> None:
    validate_document_store_for_production_suspended_operations(document_store)
    validate_suspended_operation_store_for_production(store)


__all__ = [
    "SuspendedOperationCompositionError",
    "validate_document_store_for_production_suspended_operations",
    "validate_production_suspended_operation_wiring",
    "validate_suspended_operation_store_for_production",
    "wire_default_suspended_operation_codec_registry",
    "wire_execution_suspended_work_reentry_coordinator",
    "wire_suspended_execution_operation_store",
]
