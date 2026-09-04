# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime adapter for delegated subtask child execution (AC-4 Phase 8)."""

from __future__ import annotations

from typing import Generic, TypeVar

from intergrax.agent_distribution.delegated_subtasks import (
    ChildExecutionPort,
    DelegatedChildExecutionOptions,
    DelegatedSubtaskDelegate,
)
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.nexus.budget.budget_models import RunBudget

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


class ChildExecutionRunnerPort(Generic[RequestT, ResultT]):
    """Bridge :class:`ChildExecutionRunner` to :class:`ChildExecutionPort`."""

    __slots__ = ("_runner",)

    def __init__(self, runner: ChildExecutionRunner[RequestT, ResultT]) -> None:
        self._runner = runner

    async def execute_child(
        self,
        *,
        request: RequestT,
        delegate: DelegatedSubtaskDelegate[RequestT, ResultT],
        options: DelegatedChildExecutionOptions | None = None,
    ) -> ResultT:
        requested_permission_scopes = (
            None if options is None else options.requested_permission_scopes
        )
        requested_budget = None
        if options is not None and options.requested_budget is not None:
            if isinstance(options.requested_budget, RunBudget):
                requested_budget = options.requested_budget
            else:
                raise TypeError(
                    "requested_budget must be RunBudget for ChildExecutionRunnerPort",
                )
        return await self._runner.execute(
            request=request,
            delegate=delegate,
            requested_permission_scopes=requested_permission_scopes,
            requested_budget=requested_budget,
        )


def as_child_execution_port(
    runner: ChildExecutionRunner[RequestT, ResultT],
) -> ChildExecutionPort[RequestT, ResultT]:
    return ChildExecutionRunnerPort(runner)


__all__ = ["ChildExecutionRunnerPort", "as_child_execution_port"]
