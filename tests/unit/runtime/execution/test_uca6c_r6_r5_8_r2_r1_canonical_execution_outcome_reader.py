# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.8-R2-R1 — EE terminal outcome reader by ExecutionId."""

from __future__ import annotations

import pytest

from intergrax.autonomous_work.recovery_orchestration_ports import (
    CanonicalExecutionTerminalDisposition,
)
from intergrax.contracts.execution.execution_terminal_outcome_by_execution_id import (
    ExecutionTerminalOutcomeByExecutionIdDisposition,
)
from intergrax.contracts.execution_identity import mint_execution_id
from intergrax.runtime.execution.execution_terminal_outcome_by_execution_id import (
    InMemoryExecutionTerminalOutcomeByExecutionIdStore,
    build_canonical_execution_outcome_reader,
)

pytestmark = pytest.mark.unit


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
