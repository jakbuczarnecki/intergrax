# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime propagation for active execution lineage (DG-001 R1)."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass

from intergrax.contracts.execution_identity import ExecutionId, validate_execution_id
from intergrax.contracts.execution_lineage import (
    ExecutionLineageAttemptScope,
    ExecutionLineagePersistence,
)


@dataclass(frozen=True, slots=True)
class ActiveExecutionLineageState:
    """Runtime-only lineage propagation carrier — not authority."""

    persistence: ExecutionLineagePersistence
    scope: ExecutionLineageAttemptScope
    segment_root_execution_id: ExecutionId


@dataclass(frozen=True, slots=True)
class AttemptLineageDegradationState:
    """Runtime-only monotonic degradation fast path."""

    degraded: bool


_active_execution_lineage: ContextVar[ActiveExecutionLineageState | None] = ContextVar(
    "active_execution_lineage",
    default=None,
)
_attempt_lineage_degradation: ContextVar[AttemptLineageDegradationState | None] = (
    ContextVar(
        "attempt_lineage_degradation",
        default=None,
    )
)


def bind_active_execution_lineage(state: ActiveExecutionLineageState) -> Token:
    validated_root = validate_execution_id(state.segment_root_execution_id)
    return _active_execution_lineage.set(
        ActiveExecutionLineageState(
            persistence=state.persistence,
            scope=state.scope,
            segment_root_execution_id=validated_root,
        ),
    )


def peek_active_execution_lineage() -> ActiveExecutionLineageState | None:
    return _active_execution_lineage.get()


def require_active_execution_lineage() -> ActiveExecutionLineageState:
    state = peek_active_execution_lineage()
    if state is None:
        raise RuntimeError("active execution lineage required")
    return state


def reset_active_execution_lineage(token: Token) -> None:
    _active_execution_lineage.reset(token)


def bind_attempt_lineage_degradation(degraded: bool) -> Token:
    current = peek_attempt_lineage_degradation()
    if current is not None and current.degraded:
        degraded = True
    return _attempt_lineage_degradation.set(
        AttemptLineageDegradationState(degraded=degraded)
    )


def peek_attempt_lineage_degradation() -> AttemptLineageDegradationState | None:
    return _attempt_lineage_degradation.get()


def reset_attempt_lineage_degradation(token: Token) -> None:
    _attempt_lineage_degradation.reset(token)
