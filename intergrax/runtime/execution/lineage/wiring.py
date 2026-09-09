# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Composition resolver for execution lineage persistence (DG-001 R1)."""

from __future__ import annotations

from intergrax.contracts.execution_lineage import (
    ExecutionLineageConfigurationError,
    ExecutionLineagePersistence,
    ExecutionLineagePersistenceProvider,
)
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.integrations.contracts.partition_atomic_document_store import (
    PartitionAtomicDocumentStore,
)
from intergrax.runtime.execution.lineage.document_store_persistence import (
    DocumentStoreExecutionLineagePersistence,
)

_MISSING_SELECTED_PROVIDER_MSG = (
    "execution_lineage_persistence_provider={provider} requires the selected capability"
)


def resolve_execution_lineage_persistence(
    *,
    explicit_persistence: ExecutionLineagePersistence | None = None,
    document_store: DocumentStore | None = None,
    provider: ExecutionLineagePersistenceProvider | None = None,
) -> ExecutionLineagePersistence | None:
    """
    Select execution lineage persistence.

    Precedence: explicit persistence > configured provider + compatible document store > None.
    """
    if explicit_persistence is not None:
        return explicit_persistence
    if provider is None or document_store is None:
        return None
    if provider is not ExecutionLineagePersistenceProvider.DOCUMENT_STORE:
        raise ExecutionLineageConfigurationError(
            _MISSING_SELECTED_PROVIDER_MSG.format(provider=provider.value),
        )
    if not isinstance(document_store, PartitionAtomicDocumentStore):
        raise ExecutionLineageConfigurationError(
            "execution lineage persistence requires PartitionAtomicDocumentStore",
        )
    return DocumentStoreExecutionLineagePersistence(document_store)


def require_compatible_execution_lineage_persistence(
    *,
    explicit_persistence: ExecutionLineagePersistence | None,
    document_store: DocumentStore | None,
    provider: ExecutionLineagePersistenceProvider | None = None,
) -> ExecutionLineagePersistence:
    if explicit_persistence is not None:
        return explicit_persistence
    if provider is None or document_store is None:
        raise ExecutionLineageConfigurationError(
            "execution lineage persistence requires durable provider",
        )
    if not isinstance(document_store, PartitionAtomicDocumentStore):
        raise ExecutionLineageConfigurationError(
            "execution lineage persistence requires PartitionAtomicDocumentStore",
        )
    return DocumentStoreExecutionLineagePersistence(document_store)
