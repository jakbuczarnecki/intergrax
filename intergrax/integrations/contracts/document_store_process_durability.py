# © Artur Czarnecki. All rights reserved.

"""Process-restart durability capability for document store backends (UCA-6C-R6-R2)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.integrations.contracts.document_store import ConditionalDocumentStore


@runtime_checkable
class ProcessRestartDurableDocumentStore(ConditionalDocumentStore, Protocol):
    """Backing store whose persisted rows survive process termination."""

    @property
    def survives_process_restart(self) -> bool:
        """When True, a new client instance can reload the same logical dataset."""


def document_store_survives_process_restart(
    document_store: ConditionalDocumentStore,
) -> bool:
    if isinstance(document_store, ProcessRestartDurableDocumentStore):
        return bool(document_store.survives_process_restart)
    return False


__all__ = [
    "ProcessRestartDurableDocumentStore",
    "document_store_survives_process_restart",
]
