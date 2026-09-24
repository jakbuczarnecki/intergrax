# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Durable terminal execution outcome keyed by canonical ExecutionId (EE read contract)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_identity import ExecutionId


class ExecutionTerminalOutcomeByExecutionIdDisposition(StrEnum):
    """Terminal dispositions recorded for one root execution identity."""

    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


@runtime_checkable
class ExecutionTerminalOutcomeByExecutionIdReadPort(Protocol):
    """Read-side port for worker recovery and other consumers — EE is source of truth."""

    def get_recorded_disposition(
        self,
        execution_id: ExecutionId,
    ) -> ExecutionTerminalOutcomeByExecutionIdDisposition | None: ...


class ExecutionTerminalOutcomeByExecutionIdStore(
    ExecutionTerminalOutcomeByExecutionIdReadPort,
    ABC,
):
    """Write authority for terminal outcomes — owned by Execution Engine lifecycle."""

    @abstractmethod
    def record_terminal_disposition(
        self,
        execution_id: ExecutionId,
        disposition: ExecutionTerminalOutcomeByExecutionIdDisposition,
    ) -> None: ...


__all__ = [
    "ExecutionTerminalOutcomeByExecutionIdDisposition",
    "ExecutionTerminalOutcomeByExecutionIdReadPort",
    "ExecutionTerminalOutcomeByExecutionIdStore",
]
