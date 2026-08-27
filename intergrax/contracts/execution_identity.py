# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical execution identity carriers (OBSERVABILITY §5, TRACE-1A/1B)."""

from __future__ import annotations

import re
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import NewType
from uuid import uuid4

TaskId = NewType("TaskId", str)
RunId = NewType("RunId", str)
AttemptId = NewType("AttemptId", str)
ExecutionId = NewType("ExecutionId", str)
EventId = NewType("EventId", str)

_CANONICAL_SUFFIX = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True, slots=True)
class ActiveExecutionIdentityState:
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId | None = None
    parent_execution_id: ExecutionId | None = None


_active_execution_identity: ContextVar[ActiveExecutionIdentityState | None] = ContextVar(
    "active_execution_identity",
    default=None,
)


def _validate_canonical_id(value: object, prefix: str, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be str, got {type(value).__name__}")
    if not value or not value.strip():
        raise ValueError(f"{label} must be non-empty and not whitespace-only")
    if value != value.strip():
        raise ValueError(f"{label} must not contain leading or trailing whitespace")
    if not value.startswith(prefix):
        raise ValueError(f"{label} must start with {prefix!r}")
    suffix = value[len(prefix):]
    if not _CANONICAL_SUFFIX.fullmatch(suffix):
        raise ValueError(f"{label} suffix must match [0-9a-f]{{32}}")
    return value


def validate_task_id(value: object) -> TaskId:
    return TaskId(_validate_canonical_id(value, "task_", "TaskId"))


def validate_run_id(value: object) -> RunId:
    return RunId(_validate_canonical_id(value, "run_", "RunId"))


def validate_attempt_id(value: object) -> AttemptId:
    return AttemptId(_validate_canonical_id(value, "attempt_", "AttemptId"))


def validate_execution_id(value: object) -> ExecutionId:
    return ExecutionId(_validate_canonical_id(value, "exec_", "ExecutionId"))


def validate_event_id(value: object) -> EventId:
    return EventId(_validate_canonical_id(value, "evt_", "EventId"))


def mint_task_id() -> TaskId:
    return TaskId(f"task_{uuid4().hex}")


def mint_run_id() -> RunId:
    return RunId(f"run_{uuid4().hex}")


def mint_attempt_id() -> AttemptId:
    return AttemptId(f"attempt_{uuid4().hex}")


def mint_execution_id() -> ExecutionId:
    return ExecutionId(f"exec_{uuid4().hex}")


def mint_event_id() -> EventId:
    return EventId(f"evt_{uuid4().hex}")


def bind_active_execution_identity(
    *,
    run_id: RunId,
    attempt_id: AttemptId,
    execution_id: ExecutionId | None = None,
    parent_execution_id: ExecutionId | None = None,
) -> Token:
    validated_run_id = validate_run_id(run_id)
    validated_attempt_id = validate_attempt_id(attempt_id)
    validated_execution_id = (
        validate_execution_id(execution_id) if execution_id is not None else None
    )
    validated_parent_execution_id = (
        validate_execution_id(parent_execution_id)
        if parent_execution_id is not None
        else None
    )
    state = ActiveExecutionIdentityState(
        run_id=validated_run_id,
        attempt_id=validated_attempt_id,
        execution_id=validated_execution_id,
        parent_execution_id=validated_parent_execution_id,
    )
    return _active_execution_identity.set(state)


def reset_active_execution_identity(token: Token) -> None:
    _active_execution_identity.reset(token)


def peek_active_execution_identity() -> tuple[RunId, AttemptId] | None:
    state = _active_execution_identity.get()
    if state is None:
        return None
    return (state.run_id, state.attempt_id)


def peek_active_execution_id() -> ExecutionId | None:
    state = _active_execution_identity.get()
    if state is None:
        return None
    return state.execution_id


def peek_active_parent_execution_id() -> ExecutionId | None:
    state = _active_execution_identity.get()
    if state is None:
        return None
    return state.parent_execution_id


def require_active_execution_identity() -> tuple[RunId, AttemptId]:
    bound = peek_active_execution_identity()
    if bound is None:
        raise RuntimeError("active execution identity required")
    return bound


def require_active_execution_id() -> ExecutionId:
    execution_id = peek_active_execution_id()
    if execution_id is None:
        raise RuntimeError("active ExecutionId required")
    return execution_id


def transition_active_execution_identity() -> AttemptId:
    state = _active_execution_identity.get()
    if state is None:
        raise RuntimeError("active execution identity required")
    new_attempt_id = mint_attempt_id()
    _active_execution_identity.set(
        ActiveExecutionIdentityState(
            run_id=state.run_id,
            attempt_id=new_attempt_id,
            execution_id=None,
            parent_execution_id=None,
        ),
    )
    return new_attempt_id


class ActiveExecutionIdentity:
    """Stateless facade; canonical active identity lives in ContextVar."""

    __slots__ = ()

    def bind(
        self,
        *,
        run_id: RunId,
        attempt_id: AttemptId,
        execution_id: ExecutionId | None = None,
        parent_execution_id: ExecutionId | None = None,
    ) -> Token:
        return bind_active_execution_identity(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            parent_execution_id=parent_execution_id,
        )

    def reset(self, token: Token) -> None:
        reset_active_execution_identity(token)

    @property
    def run_id(self) -> RunId | None:
        state = _active_execution_identity.get()
        return state.run_id if state is not None else None

    @property
    def attempt_id(self) -> AttemptId | None:
        state = _active_execution_identity.get()
        return state.attempt_id if state is not None else None

    @property
    def execution_id(self) -> ExecutionId | None:
        return peek_active_execution_id()

    @property
    def parent_execution_id(self) -> ExecutionId | None:
        return peek_active_parent_execution_id()

    def require(self) -> tuple[RunId, AttemptId]:
        return require_active_execution_identity()

    def require_execution_id(self) -> ExecutionId:
        return require_active_execution_id()

    def transition_retry(self) -> AttemptId:
        return transition_active_execution_identity()

    def clear(self) -> None:
        raise RuntimeError(
            "ActiveExecutionIdentity.clear() is forbidden; reset the bind token instead",
        )
