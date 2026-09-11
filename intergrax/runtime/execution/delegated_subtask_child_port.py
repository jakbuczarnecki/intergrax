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
from intergrax.runtime.execution.execution_work_port import (
    DelegatedSpecialistChildWorkEnvelope,
    DelegatedSubtaskChildExecutionWorkPort,
)
from intergrax.runtime.execution.request import ExecutionRequest
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


class DelegatedSubtaskChildExecutionWorkPortAdapter(Generic[RequestT, ResultT]):
    """Translate :class:`ChildExecutionPort` invocations to canonical child work port."""

    __slots__ = ("_work_port",)

    def __init__(
        self,
        work_port: DelegatedSubtaskChildExecutionWorkPort[RequestT, ResultT],
    ) -> None:
        self._work_port = work_port

    async def execute_child(
        self,
        *,
        request: RequestT,
        delegate: DelegatedSubtaskDelegate[RequestT, ResultT],
        options: DelegatedChildExecutionOptions | None = None,
    ) -> ResultT:
        envelope = DelegatedSpecialistChildWorkEnvelope(
            domain_request=request,
            specialist=delegate,
            options=options,
        )
        work_request = ExecutionRequest(
            input=envelope,
            output_type=None,
        )
        return await self._work_port.execute(work_request)


def child_execution_port_from_work_port(
    work_port: DelegatedSubtaskChildExecutionWorkPort[RequestT, ResultT],
) -> ChildExecutionPort[RequestT, ResultT]:
    """Build domain-facing child port over canonical :class:`ExecutionWorkPort` wiring."""
    return DelegatedSubtaskChildExecutionWorkPortAdapter(work_port)


__all__ = [
    "ChildExecutionRunnerPort",
    "DelegatedSubtaskChildExecutionWorkPortAdapter",
    "as_child_execution_port",
    "child_execution_port_from_work_port",
]
