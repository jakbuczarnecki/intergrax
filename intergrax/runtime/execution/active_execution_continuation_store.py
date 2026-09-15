# © Artur Czarnecki. All rights reserved.

"""Execution-scoped continuation state store binding (GR-5-R2-R2)."""

from __future__ import annotations

from contextvars import ContextVar, Token

from intergrax.contracts.execution_continuation_state_store import ExecutionContinuationStateStore

_active_execution_continuation_state_store: ContextVar[
    ExecutionContinuationStateStore | None
] = ContextVar(
    "active_execution_continuation_state_store",
    default=None,
)


def bind_active_execution_continuation_state_store(
    store: ExecutionContinuationStateStore,
) -> Token[ExecutionContinuationStateStore | None]:
    return _active_execution_continuation_state_store.set(store)


def reset_active_execution_continuation_state_store(
    token: Token[ExecutionContinuationStateStore | None],
) -> None:
    _active_execution_continuation_state_store.reset(token)


def peek_active_execution_continuation_state_store() -> ExecutionContinuationStateStore | None:
    return _active_execution_continuation_state_store.get()


__all__ = [
    "bind_active_execution_continuation_state_store",
    "peek_active_execution_continuation_state_store",
    "reset_active_execution_continuation_state_store",
]
