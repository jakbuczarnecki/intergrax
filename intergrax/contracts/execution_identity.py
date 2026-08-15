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
EventId = NewType("EventId", str)

_active_execution_identity: ContextVar[tuple[RunId, AttemptId] | None] = ContextVar(
    "active_execution_identity",
    default=None,
)

_CANONICAL_SUFFIX = re.compile(r"^[0-9a-f]{32}$")


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


def validate_event_id(value: object) -> EventId:
    return EventId(_validate_canonical_id(value, "evt_", "EventId"))


def mint_task_id() -> TaskId:
    return TaskId(f"task_{uuid4().hex}")


def mint_run_id() -> RunId:
    return RunId(f"run_{uuid4().hex}")


def mint_attempt_id() -> AttemptId:
    return AttemptId(f"attempt_{uuid4().hex}")


def mint_event_id() -> EventId:
    return EventId(f"evt_{uuid4().hex}")


def bind_active_execution_identity(*, run_id: RunId, attempt_id: AttemptId) -> Token:
    validated_run_id = validate_run_id(run_id)
    validated_attempt_id = validate_attempt_id(attempt_id)
    return _active_execution_identity.set((validated_run_id, validated_attempt_id))


def reset_active_execution_identity(token: Token) -> None:
    _active_execution_identity.reset(token)


def require_active_execution_identity() -> tuple[RunId, AttemptId]:
    bound = _active_execution_identity.get()
    if bound is None:
        raise RuntimeError("active execution identity required")
    return bound


@dataclass
class ActiveExecutionIdentity:
    """Mutable run/attempt binding for a single Nexus execution scope."""

    run_id: RunId | None = None
    attempt_id: AttemptId | None = None
    _context_token: Token | None = None

    def bind(self, *, run_id: RunId, attempt_id: AttemptId) -> None:
        self.run_id = validate_run_id(run_id)
        self.attempt_id = validate_attempt_id(attempt_id)
        if self._context_token is not None:
            reset_active_execution_identity(self._context_token)
        self._context_token = bind_active_execution_identity(
            run_id=self.run_id,
            attempt_id=self.attempt_id,
        )

    def clear(self) -> None:
        if self._context_token is not None:
            reset_active_execution_identity(self._context_token)
            self._context_token = None
        self.run_id = None
        self.attempt_id = None

    def require(self) -> tuple[RunId, AttemptId]:
        if self.run_id is None or self.attempt_id is None:
            raise RuntimeError("active execution identity required")
        return self.run_id, self.attempt_id

    def transition_retry(self) -> AttemptId:
        run_id, _ = self.require()
        new_attempt_id = mint_attempt_id()
        self.bind(run_id=run_id, attempt_id=new_attempt_id)
        return new_attempt_id
