# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Composition resolver for execution lineage persistence (DG-001 R1)."""

from __future__ import annotations

from intergrax.contracts.execution_lineage import (
    ExecutionLineageConfigurationError,
    ExecutionLineagePersistence,
)
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.integrations.contracts.partition_atomic_document_store import (
    PartitionAtomicDocumentStore,
)
from intergrax.runtime.execution.lineage.document_store_persistence import (
    DocumentStoreExecutionLineagePersistence,
)


def resolve_execution_lineage_persistence(
    *,
    explicit_persistence: ExecutionLineagePersistence | None = None,
    document_store: DocumentStore | None = None,
) -> ExecutionLineagePersistence | None:
    """
    Select execution lineage persistence.

    Precedence: explicit persistence > PartitionAtomicDocumentStore adapter > None.
    """
    if explicit_persistence is not None:
        return explicit_persistence
    if document_store is None:
        return None
    if not isinstance(document_store, PartitionAtomicDocumentStore):
        return None
    return DocumentStoreExecutionLineagePersistence(document_store)


def require_compatible_execution_lineage_persistence(
    *,
    explicit_persistence: ExecutionLineagePersistence | None,
    document_store: DocumentStore | None,
) -> ExecutionLineagePersistence:
    if explicit_persistence is not None:
        return explicit_persistence
    if document_store is None:
        raise ExecutionLineageConfigurationError(
            "execution lineage persistence requires durable provider",
        )
    if not isinstance(document_store, PartitionAtomicDocumentStore):
        raise ExecutionLineageConfigurationError(
            "execution lineage persistence requires PartitionAtomicDocumentStore",
        )
    return DocumentStoreExecutionLineagePersistence(document_store)
