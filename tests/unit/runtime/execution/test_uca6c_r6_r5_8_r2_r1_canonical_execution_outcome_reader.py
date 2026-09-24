# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.8-R2-R1 — EE terminal outcome reader by ExecutionId."""

from __future__ import annotations

import pytest

from intergrax.autonomous_work.recovery_orchestration_ports import (
    CanonicalExecutionTerminalDisposition,
)
from intergrax.contracts.execution.execution_terminal_outcome_by_execution_id import (
    ExecutionTerminalOutcomeByExecutionIdDisposition,
    ExecutionTerminalOutcomeByExecutionIdReadPort,
)
from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionDispatchDisposition,
)
from intergrax.contracts.execution_identity import ExecutionId, mint_execution_id
from intergrax.runtime.execution.execution_terminal_outcome_by_execution_id import (
    InMemoryExecutionTerminalOutcomeByExecutionIdStore,
    build_canonical_execution_outcome_reader,
    record_delegate_terminal_disposition,
)

pytestmark = pytest.mark.unit


class _ReadOnlyTerminalOutcomePort:
    """Fake read port without write authority."""

    def __init__(self) -> None:
        self._records: dict[str, ExecutionTerminalOutcomeByExecutionIdDisposition] = {}

    def seed(
        self,
        execution_id: ExecutionId,
        disposition: ExecutionTerminalOutcomeByExecutionIdDisposition,
    ) -> None:
        self._records[str(execution_id)] = disposition

    def get_recorded_disposition(
        self,
        execution_id: ExecutionId,
    ) -> ExecutionTerminalOutcomeByExecutionIdDisposition | None:
        return self._records.get(str(execution_id))


def test_reader_uses_read_port_without_write_method() -> None:
    read_port: ExecutionTerminalOutcomeByExecutionIdReadPort = (
        _ReadOnlyTerminalOutcomePort()
    )
    reader = build_canonical_execution_outcome_reader(read_port)
    execution_id = mint_execution_id()
    assert (
        reader.get_terminal_outcome(execution_id).disposition
        is CanonicalExecutionTerminalDisposition.IN_PROGRESS
    )
    assert not hasattr(read_port, "record_terminal_disposition")


def test_reader_in_progress_until_terminal_recorded() -> None:
    store = InMemoryExecutionTerminalOutcomeByExecutionIdStore()
    reader = build_canonical_execution_outcome_reader(store)
    execution_id = mint_execution_id()
    assert (
        reader.get_terminal_outcome(execution_id).disposition
        is CanonicalExecutionTerminalDisposition.IN_PROGRESS
    )
    store.record_terminal_disposition(
        execution_id,
        ExecutionTerminalOutcomeByExecutionIdDisposition.SUCCEEDED,
    )
    assert (
        reader.get_terminal_outcome(execution_id).disposition
        is CanonicalExecutionTerminalDisposition.SUCCEEDED
    )


def test_reader_unknown_execution_fail_closed_in_progress() -> None:
    store = InMemoryExecutionTerminalOutcomeByExecutionIdStore()
    reader = build_canonical_execution_outcome_reader(store)
    other = mint_execution_id()
    assert (
        reader.get_terminal_outcome(other).disposition
        is CanonicalExecutionTerminalDisposition.IN_PROGRESS
    )


def test_reader_maps_failed_terminal() -> None:
    read_port = _ReadOnlyTerminalOutcomePort()
    execution_id = mint_execution_id()
    read_port.seed(
        execution_id, ExecutionTerminalOutcomeByExecutionIdDisposition.FAILED
    )
    reader = build_canonical_execution_outcome_reader(read_port)
    assert (
        reader.get_terminal_outcome(execution_id).disposition
        is CanonicalExecutionTerminalDisposition.FAILED
    )


def test_terminal_outcome_conflict_fail_closed() -> None:
    store = InMemoryExecutionTerminalOutcomeByExecutionIdStore()
    execution_id = mint_execution_id()
    store.record_terminal_disposition(
        execution_id,
        ExecutionTerminalOutcomeByExecutionIdDisposition.SUCCEEDED,
    )
    with pytest.raises(ValueError, match="terminal outcome conflict"):
        store.record_terminal_disposition(
            execution_id,
            ExecutionTerminalOutcomeByExecutionIdDisposition.FAILED,
        )


def test_delegate_dispatched_does_not_terminalize_pause() -> None:
    store = InMemoryExecutionTerminalOutcomeByExecutionIdStore()
    execution_id = mint_execution_id()
    record_delegate_terminal_disposition(
        store,
        execution_id=execution_id,
        disposition=QualifiedCapabilityExecutionDispatchDisposition.DISPATCHED,
    )
    reader = build_canonical_execution_outcome_reader(store)
    assert (
        reader.get_terminal_outcome(execution_id).disposition
        is CanonicalExecutionTerminalDisposition.IN_PROGRESS
    )


def test_delegate_failed_terminalizes() -> None:
    store = InMemoryExecutionTerminalOutcomeByExecutionIdStore()
    execution_id = mint_execution_id()
    record_delegate_terminal_disposition(
        store,
        execution_id=execution_id,
        disposition=QualifiedCapabilityExecutionDispatchDisposition.FAILED,
    )
    reader = build_canonical_execution_outcome_reader(store)
    assert (
        reader.get_terminal_outcome(execution_id).disposition
        is CanonicalExecutionTerminalDisposition.FAILED
    )
