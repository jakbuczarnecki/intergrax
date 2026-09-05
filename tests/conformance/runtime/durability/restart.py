# © Artur Czarnecki. All rights reserved.

"""P0C-8 process restart semantics for durability conformance."""

from __future__ import annotations

from pathlib import Path

from intergrax.runtime.background_execution.admission_wiring import (
    BackgroundExecutionAdmissionDependencies,
)
from intergrax.runtime.execution.execution_terminal import ExecutionTerminalService
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore

from tests.conformance.runtime.durability.provider_factories import (
    DurableAdmissionBacking,
    create_admission_dependencies,
    create_checkpoint_store,
    create_checkpoint_terminal_service,
)

# In P0C-8, *restart* means:
#     same durable backing primitive
#     + new adapter instance
#     + new service instance
# NOT reusing the previous in-process service object.


def fresh_admission_composition(
    backing: DurableAdmissionBacking,
) -> BackgroundExecutionAdmissionDependencies:
    """Return a new admission dependency bundle bound to the same durable backing."""
    return create_admission_dependencies(backing)


def fresh_checkpoint_composition(
    db_path: Path,
) -> tuple[SQLiteTaskCheckpointStore, ExecutionTerminalService]:
    """Return fresh checkpoint store + terminal service adapters on the same DB file."""
    store = create_checkpoint_store(db_path)
    terminal = create_checkpoint_terminal_service(store)
    return store, terminal
