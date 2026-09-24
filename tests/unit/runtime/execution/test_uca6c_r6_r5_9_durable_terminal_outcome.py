# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.9 — durable execution terminal outcome store semantics."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution.execution_terminal_outcome_by_execution_id import (
    ExecutionTerminalOutcomeByExecutionIdDisposition,
)
from intergrax.contracts.execution_identity import mint_execution_id
from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.runtime.execution.document_store_execution_terminal_outcome_by_execution_id import (
    DocumentStoreExecutionTerminalOutcomeByExecutionIdStore,
)
from intergrax.runtime.execution.execution_terminal_outcome_by_execution_id import (
    InMemoryExecutionTerminalOutcomeByExecutionIdStore,
)


def test_document_store_outcome_restart_visibility() -> None:
    backend = InMemoryDocumentStore()
    execution_id = mint_execution_id()
    host_a = DocumentStoreExecutionTerminalOutcomeByExecutionIdStore(backend)
    host_a.record_terminal_disposition(
        execution_id,
        ExecutionTerminalOutcomeByExecutionIdDisposition.SUCCEEDED,
    )
    del host_a
    host_b = DocumentStoreExecutionTerminalOutcomeByExecutionIdStore(backend)
    assert (
        host_b.get_recorded_disposition(execution_id)
        is ExecutionTerminalOutcomeByExecutionIdDisposition.SUCCEEDED
    )


def test_concurrent_same_terminal_idempotent() -> None:
    backend = InMemoryDocumentStore()
    execution_id = mint_execution_id()
    host_a = DocumentStoreExecutionTerminalOutcomeByExecutionIdStore(backend)
    host_b = DocumentStoreExecutionTerminalOutcomeByExecutionIdStore(backend)
    host_a.record_terminal_disposition(
        execution_id,
        ExecutionTerminalOutcomeByExecutionIdDisposition.SUCCEEDED,
    )
    host_b.record_terminal_disposition(
        execution_id,
        ExecutionTerminalOutcomeByExecutionIdDisposition.SUCCEEDED,
    )
    assert (
        host_b.get_recorded_disposition(execution_id)
        is ExecutionTerminalOutcomeByExecutionIdDisposition.SUCCEEDED
    )


def test_concurrent_conflicting_terminal_fails_closed() -> None:
    backend = InMemoryDocumentStore()
    execution_id = mint_execution_id()
    host_a = DocumentStoreExecutionTerminalOutcomeByExecutionIdStore(backend)
    host_b = DocumentStoreExecutionTerminalOutcomeByExecutionIdStore(backend)
    host_a.record_terminal_disposition(
        execution_id,
        ExecutionTerminalOutcomeByExecutionIdDisposition.SUCCEEDED,
    )
    with pytest.raises(ValueError, match="conflict"):
        host_b.record_terminal_disposition(
            execution_id,
            ExecutionTerminalOutcomeByExecutionIdDisposition.FAILED,
        )


def test_in_memory_and_document_store_conflict_parity() -> None:
    execution_id = mint_execution_id()
    mem = InMemoryExecutionTerminalOutcomeByExecutionIdStore()
    mem.record_terminal_disposition(
        execution_id,
        ExecutionTerminalOutcomeByExecutionIdDisposition.SUCCEEDED,
    )
    with pytest.raises(ValueError, match="conflict"):
        mem.record_terminal_disposition(
            execution_id,
            ExecutionTerminalOutcomeByExecutionIdDisposition.FAILED,
        )
