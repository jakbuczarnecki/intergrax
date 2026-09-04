# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Composition helpers for background execution re-entry admission (P0C-7)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_terminal import ExecutionTerminalPersistenceCapability
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.runtime.background_execution.identity_persistence import (
    BackgroundExecutionIdentityPersistence,
    wire_background_execution_identity_persistence,
)
from intergrax.runtime.execution.attempt_lifecycle import (
    AttemptLifecycleService,
    wire_attempt_lifecycle_store,
)
from intergrax.runtime.execution.execution_terminal import (
    ExecutionTerminalService,
    wire_execution_terminal_store,
)


@dataclass(frozen=True, slots=True)
class BackgroundExecutionAdmissionDependencies:
    """Immutable dependency bundle for canonical background re-entry admission."""

    identity_persistence: BackgroundExecutionIdentityPersistence
    attempt_lifecycle: AttemptLifecycleService
    execution_terminal: ExecutionTerminalService


def wire_background_execution_admission_dependencies(
    *,
    kv_store: DistributedKVStore | None = None,
    document_store: DocumentStore | None = None,
    checkpoint_store: ExecutionTerminalPersistenceCapability | None = None,
) -> BackgroundExecutionAdmissionDependencies:
    """Platform composition boundary for background worker re-entry authorities."""
    if kv_store is not None and document_store is not None:
        raise ValueError(
            "wire_background_execution_admission_dependencies accepts kv_store or "
            "document_store, not both",
        )
    if kv_store is None and document_store is None:
        raise ValueError(
            "wire_background_execution_admission_dependencies requires kv_store or document_store",
        )
    return BackgroundExecutionAdmissionDependencies(
        identity_persistence=wire_background_execution_identity_persistence(
            kv_store=kv_store,
            document_store=document_store,
        ),
        attempt_lifecycle=AttemptLifecycleService(
            wire_attempt_lifecycle_store(
                kv_store=kv_store,
                document_store=document_store,
            ),
        ),
        execution_terminal=ExecutionTerminalService(
            wire_execution_terminal_store(checkpoint_store=checkpoint_store),
        ),
    )
