# © Artur Czarnecki. All rights reserved.

"""Production durability requirements for terminal execution authority (P0C-5 / P0C-6)."""

from __future__ import annotations

from intergrax.contracts.execution_terminal import ExecutionTerminalError, ExecutionTerminalStore
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence

DURABLE_EXECUTION_TERMINAL_REQUIRED_MSG = (
    "durable execution terminal store required for terminal execution continuity"
)


def validate_durable_execution_terminal_for_composition(
    *,
    production_mode: bool,
    checkpoint_store: TaskCheckpointPersistence | None,
    store: ExecutionTerminalStore,
) -> None:
    """Fail closed when production long-running resume relies on non-durable terminal authority."""
    if not production_mode:
        return
    if checkpoint_store is None:
        return
    if not store.is_durable:
        raise ExecutionTerminalError(DURABLE_EXECUTION_TERMINAL_REQUIRED_MSG)
