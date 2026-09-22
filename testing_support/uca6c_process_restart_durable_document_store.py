# © Artur Czarnecki. All rights reserved.

"""Test-only document store declaring process-restart durability (UCA-6C-R6-R2)."""

from __future__ import annotations

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store_process_durability import (
    ProcessRestartDurableDocumentStore,
)


class ProcessRestartQualificationDocumentStore(
    InMemoryDocumentStore,
    ProcessRestartDurableDocumentStore,
):
    """Lab backing: new store clients reload rows from the same partition API."""

    @property
    def survives_process_restart(self) -> bool:
        return True


__all__ = ["ProcessRestartQualificationDocumentStore"]
