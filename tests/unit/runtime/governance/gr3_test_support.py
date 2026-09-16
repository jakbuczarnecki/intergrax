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
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
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


def minimal_meaningful_side_effect_request_for_tests(
    *,
    action: str = "op",
    kinds: tuple[MeaningfulSideEffectKind, ...] = (MeaningfulSideEffectKind.MUTATION,),
    side_effect_scope_id: str = "scope-test",
    task_id: TaskId | None = None,
    run_id: RunId | None = None,
    attempt_id: AttemptId | None = None,
    execution_id: ExecutionId | None = None,
    principal_id: str | None = "principal-test",
    tenant_id: str | None = "tenant-test",
    resource: str | None = "resource-test",
    side_effect_scope_digest: str | None = None,
) -> MeaningfulSideEffectRequest:
    """Canonical ``MeaningfulSideEffectRequest`` for governance qualification tests."""
    resolved_task_id = task_id or mint_task_id()
    resolved_run_id = run_id or mint_run_id()
    resolved_attempt_id = attempt_id or mint_attempt_id()
    resolved_execution_id = execution_id or mint_execution_id()
    return MeaningfulSideEffectRequest(
        action=action,
        kinds=kinds,
        side_effect_scope_id=side_effect_scope_id,
        side_effect_scope_digest=side_effect_scope_digest,
        task_id=resolved_task_id,
        run_id=resolved_run_id,
        attempt_id=resolved_attempt_id,
        execution_id=resolved_execution_id,
        principal_id=principal_id,
        tenant_id=tenant_id,
        resource=resource,
    )


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
