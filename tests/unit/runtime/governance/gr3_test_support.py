# © Artur Czarnecki. All rights reserved.

"""Shared GR-3 test wiring — active execution bind + injectable task scope."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from intergrax.contracts.active_execution_task_scope import ActiveExecutionTaskScopePort
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.runtime.governance.canonical_inner_execution_guard import (
    DefaultCanonicalInnerExecutionGuard,
)


class StaticActiveTaskScope(ActiveExecutionTaskScopePort):
    __slots__ = ("_task_id",)

    def __init__(self, task_id: TaskId) -> None:
        self._task_id = task_id

    def resolve_current_task_scope(
        self,
        *,
        run_id: RunId,
        attempt_id: AttemptId,
        execution_id: ExecutionId,
    ) -> TaskId:
        del run_id, attempt_id, execution_id
        return self._task_id


def default_gr3_identity_bundle() -> tuple[TaskId, RunId, AttemptId, ExecutionId]:
    return (mint_task_id(), mint_run_id(), mint_attempt_id(), mint_execution_id())


def default_gr3_inner_guard(
    task_id: TaskId,
) -> DefaultCanonicalInnerExecutionGuard:
    return DefaultCanonicalInnerExecutionGuard(task_scope=StaticActiveTaskScope(task_id))


@contextmanager
def bound_gr3_active_execution(
    *,
    run_id: RunId,
    attempt_id: AttemptId,
    execution_id: ExecutionId,
) -> Iterator[None]:
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    try:
        yield
    finally:
        reset_active_execution_identity(token)
