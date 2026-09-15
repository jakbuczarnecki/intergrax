# © Artur Czarnecki. All rights reserved.

"""GR-5-R2 — Execution Engine composition for canonical continuation lifecycle."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_continuation import ExecutionContinuationPort
from intergrax.contracts.execution_continuation_state_store import ExecutionContinuationStateStore
from intergrax.runtime.execution.continuation.persistence import wire_execution_continuation_state_store
from intergrax.runtime.execution.continuation.service import (
    ExecutionContinuationService,
    execution_continuation_port,
)


@dataclass(frozen=True, slots=True)
class ExecutionEngineContinuationDependencies:
    """Immutable continuation capability bundle for Execution Engine composition roots."""

    continuation: ExecutionContinuationPort
    continuation_service: ExecutionContinuationService


def wire_execution_continuation_port(
    *,
    state_store: ExecutionContinuationStateStore | None = None,
) -> ExecutionContinuationPort:
    """Resolve continuation port; explicit ``state_store=None`` uses in-memory default."""
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
    return ExecutionEngineContinuationDependencies(
        continuation=execution_continuation_port(service),
        continuation_service=service,
    )


__all__ = [
    "ExecutionEngineContinuationDependencies",
    "wire_execution_continuation_port",
    "wire_execution_engine_continuation_dependencies",
]
