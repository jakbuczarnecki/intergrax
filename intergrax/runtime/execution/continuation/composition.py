# © Artur Czarnecki. All rights reserved.

"""GR-5-R2 — Execution Engine composition for canonical continuation lifecycle."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_continuation import ExecutionContinuationPort
from intergrax.contracts.execution_continuation_state_store import ExecutionContinuationStateStore
from intergrax.runtime.execution.continuation.persistence import wire_execution_continuation_state_store
from intergrax.runtime.execution.continuation.lifecycle_driver import (
    ExecutionContinuationLifecycleDriver,
)
from intergrax.runtime.execution.continuation.service import (
    ExecutionContinuationService,
    execution_continuation_port,
)
from intergrax.runtime.task.execution_continuation_projection import (
    wire_task_execution_continuation_projection_sink,
)


@dataclass(frozen=True, slots=True)
class ExecutionEngineContinuationDependencies:
    """Immutable continuation capability bundle for Execution Engine composition roots."""

    continuation: ExecutionContinuationPort
    continuation_service: ExecutionContinuationService
    lifecycle_driver: ExecutionContinuationLifecycleDriver


def wire_execution_continuation_port(
    *,
    state_store: ExecutionContinuationStateStore | None = None,
) -> ExecutionContinuationPort:
    """Dedicated continuation composition; ``state_store=None`` uses lab in-memory default.

    Production hosts must pass an explicit **durable** store
    (``store.is_durable is True``; see ``validate_execution_continuation_for_composition``).
    Silent ``None`` / in-memory is lab/test only.
    """
    store = wire_execution_continuation_state_store(state_store=state_store)
    service = ExecutionContinuationService(store)
    return execution_continuation_port(service)


def wire_execution_engine_continuation_dependencies(
    *,
    state_store: ExecutionContinuationStateStore | None = None,
) -> ExecutionEngineContinuationDependencies:
    """Canonical Execution Engine composition boundary for GR-5 continuation."""
    store = wire_execution_continuation_state_store(state_store=state_store)
    service = ExecutionContinuationService(store)
    driver = ExecutionContinuationLifecycleDriver(service)
    return ExecutionEngineContinuationDependencies(
        continuation=execution_continuation_port(service),
        continuation_service=service,
        lifecycle_driver=driver,
    )


def reconnect_execution_engine_continuation_dependencies(
    *,
    state_store: ExecutionContinuationStateStore,
) -> ExecutionEngineContinuationDependencies:
    """Process B composition: new service/runtime objects over an existing durable store."""
    return wire_execution_engine_continuation_dependencies(state_store=state_store)


__all__ = [
    "ExecutionEngineContinuationDependencies",
    "reconnect_execution_engine_continuation_dependencies",
    "wire_execution_continuation_port",
    "wire_execution_engine_continuation_dependencies",
    "wire_task_execution_continuation_projection_sink",
]
