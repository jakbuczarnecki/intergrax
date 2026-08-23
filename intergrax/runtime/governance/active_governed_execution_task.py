# © Artur Czarnecki. All rights reserved.

"""Trusted runtime carrier for the live Nexus ``Task`` during governed agent execution."""

from __future__ import annotations

from contextvars import ContextVar, Token

from intergrax.runtime.task.task import Task

_active_governed_execution_task: ContextVar[Task | None] = ContextVar(
    "active_governed_execution_task",
    default=None,
)


def bind_governed_execution_task(task: Task) -> Token:
    return _active_governed_execution_task.set(task)


def reset_governed_execution_task(token: Token) -> None:
    _active_governed_execution_task.reset(token)


def peek_governed_execution_task() -> Task | None:
    return _active_governed_execution_task.get()


def current_governed_execution_task() -> Task | None:
    return peek_governed_execution_task()


class ActiveGovernedExecutionTask:
    """Stateless facade; canonical governed task lives in ContextVar."""

    __slots__ = ()

    def bind(self, task: Task) -> Token:
        return bind_governed_execution_task(task)

    def reset(self, token: Token) -> None:
        reset_governed_execution_task(token)

    @property
    def current(self) -> Task | None:
        return peek_governed_execution_task()

    def require(self) -> Task:
        task = peek_governed_execution_task()
        if task is None:
            raise RuntimeError("active governed execution task required")
        return task
