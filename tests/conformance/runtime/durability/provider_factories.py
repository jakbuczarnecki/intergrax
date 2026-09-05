# © Artur Czarnecki. All rights reserved.

"""Typed durable provider factories for P0C-8 conformance matrix."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.runtime.background_execution.admission_wiring import (
    BackgroundExecutionAdmissionDependencies,
    wire_background_execution_admission_dependencies,
)
from intergrax.runtime.execution.execution_terminal import ExecutionTerminalService
from intergrax.runtime.execution.execution_terminal.persistence import (
    CheckpointStoreExecutionTerminalStore,
)
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore

from tests.unit.runtime.background_execution.reentry_admission_doubles import InMemoryKVStore


class DurableProviderKind(str, Enum):
    KV = "kv"
    DOCUMENT_STORE = "document_store"


@dataclass(frozen=True, slots=True)
class DurableAdmissionBacking:
    """Shared durable primitive; service instances are recreated per restart."""

    kind: DurableProviderKind
    kv_store: DistributedKVStore | None = None
    document_store: DocumentStore | None = None

    @classmethod
    def fresh_kv(cls) -> DurableAdmissionBacking:
        return cls(kind=DurableProviderKind.KV, kv_store=InMemoryKVStore())

    @classmethod
    def fresh_document_store(cls) -> DurableAdmissionBacking:
        return cls(
            kind=DurableProviderKind.DOCUMENT_STORE,
            document_store=InMemoryDocumentStore(),
        )


BACKGROUND_IDENTITY_PROVIDERS: tuple[DurableProviderKind, ...] = (
    DurableProviderKind.KV,
    DurableProviderKind.DOCUMENT_STORE,
)


def create_admission_dependencies(
    backing: DurableAdmissionBacking,
) -> BackgroundExecutionAdmissionDependencies:
    if backing.kind is DurableProviderKind.KV:
        if backing.kv_store is None:
            raise ValueError("KV backing requires kv_store")
        return wire_background_execution_admission_dependencies(kv_store=backing.kv_store)
    if backing.document_store is None:
        raise ValueError("DocumentStore backing requires document_store")
    return wire_background_execution_admission_dependencies(
        document_store=backing.document_store,
    )


def create_checkpoint_store(db_path: Path) -> SQLiteTaskCheckpointStore:
    return SQLiteTaskCheckpointStore(db_path=db_path)


def create_checkpoint_terminal_service(
    checkpoint_store: SQLiteTaskCheckpointStore,
) -> ExecutionTerminalService:
    return ExecutionTerminalService(
        CheckpointStoreExecutionTerminalStore(checkpoint_store),
    )


def provider_qualification_matrix() -> dict[str, dict[str, str]]:
    """Explicit provider capability matrix (must match running tests)."""
    yes = "yes"
    na = "n/a"
    return {
        "background identity": {
            "KV": yes,
            "DocumentStore": yes,
            "Checkpoint": na,
        },
        "attempt lifecycle": {
            "KV": yes,
            "DocumentStore": yes,
            "Checkpoint": na,
        },
        "terminal authority": {
            "KV": yes,
            "DocumentStore": yes,
            "Checkpoint": yes,
        },
        "checkpoint/recovery": {
            "KV": na,
            "DocumentStore": na,
            "Checkpoint": yes,
        },
    }
