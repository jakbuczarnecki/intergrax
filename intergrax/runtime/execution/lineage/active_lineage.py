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
    non_durable_execution_ids: frozenset[ExecutionId] = frozenset()


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
    validated_non_durable = frozenset(
        validate_execution_id(execution_id)
        for execution_id in state.non_durable_execution_ids
    )
    return _active_execution_lineage.set(
        ActiveExecutionLineageState(
            persistence=state.persistence,
            scope=state.scope,
            segment_root_execution_id=validated_root,
            non_durable_execution_ids=validated_non_durable,
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


def mark_execution_lineage_non_durable(execution_id: ExecutionId) -> None:
    """Monotonic runtime mark: execution admission was not durably persisted."""
    current = peek_active_execution_lineage()
    if current is None:
        return
    validated = validate_execution_id(execution_id)
    if validated in current.non_durable_execution_ids:
        return
    _active_execution_lineage.set(
        ActiveExecutionLineageState(
            persistence=current.persistence,
            scope=current.scope,
            segment_root_execution_id=current.segment_root_execution_id,
            non_durable_execution_ids=current.non_durable_execution_ids
            | frozenset({validated}),
        ),
    )


def bind_attempt_lineage_degradation(degraded: bool) -> Token:
    current = peek_attempt_lineage_degradation()
    if current is not None and current.degraded:
        degraded = True
    return _attempt_lineage_degradation.set(
        AttemptLineageDegradationState(degraded=degraded)
    )


def peek_attempt_lineage_degradation() -> AttemptLineageDegradationState | None:
    return _attempt_lineage_degradation.get()


def mark_attempt_lineage_degraded() -> None:
    """Monotonic runtime degradation mark — does not return a reset token."""
    current = peek_attempt_lineage_degradation()
    if current is not None and current.degraded:
        return
    _attempt_lineage_degradation.set(AttemptLineageDegradationState(degraded=True))


def reset_attempt_lineage_degradation(token: Token) -> None:
    _attempt_lineage_degradation.reset(token)
