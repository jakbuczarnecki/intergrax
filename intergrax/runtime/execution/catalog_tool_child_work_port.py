# © Artur Czarnecki. All rights reserved.

"""Canonical child work owner for in-process catalog tool invoke (ChildExecutionRunner)."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.child_execution_context_inheritance import (
    ChildExecutionContextInheritancePort,
)
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvokeRequest,
    ExecutionBoundCatalogToolInvoker,
)
from intergrax.contracts.execution_request import ExecutionRequest
from intergrax.contracts.tool_execution_outcome import ToolExecutionResult
from intergrax.runtime.execution.boundary import ExecutionAdmissionHook
from intergrax.runtime.execution.budget import (
    ExecutionBudgetLedger,
    create_execution_budget_ledger,
)
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.governance.active_governed_execution_task import (
    bind_governed_execution_task,
    reset_governed_execution_task,
)
from intergrax.runtime.task.task import Task


@dataclass(frozen=True, slots=True)
class CatalogToolChildInvokeWork:
    """Child-bound catalog invoke payload admitted only under active parent execution."""

    invoke_request: ExecutionBoundCatalogToolInvokeRequest
    governed_task: Task
    catalog_invoker: ExecutionBoundCatalogToolInvoker
    tool_id: str
    step_id: str
    side_effects: bool
    caller: RequestIdentity
    resolved_agent_id: str


class _CatalogToolChildDelegate:
    """Invokes the execution-bound catalog gateway under child execution identity."""

    async def execute(
        self,
        request: ExecutionRequest[CatalogToolChildInvokeWork, ToolExecutionResult[BaseModel]],
    ) -> ToolExecutionResult[BaseModel]:
        work = request.input
        token = bind_governed_execution_task(work.governed_task)
        try:
            return work.catalog_invoker.invoke(work.invoke_request)
        finally:
            reset_governed_execution_task(token)


class CatalogToolChildExecutionWorkPort:
    """
    Routes catalog child work through :class:`ChildExecutionRunner` (sole child id mint).
    """

    __slots__ = ("_child_runner", "_delegate")

    def __init__(
        self,
        *,
        ledger: ExecutionBudgetLedger | None = None,
        child_context_inheritance: ChildExecutionContextInheritancePort | None = None,
    ) -> None:
        self._delegate = _CatalogToolChildDelegate()
        self._child_runner = ChildExecutionRunner[
            ExecutionRequest[CatalogToolChildInvokeWork, ToolExecutionResult[BaseModel]],
            ToolExecutionResult[BaseModel],
        ](
            ledger=ledger,
            child_context_inheritance=child_context_inheritance,
        )

    async def execute(
        self,
        *,
        work: CatalogToolChildInvokeWork,
        admission_hooks: tuple[
            ExecutionAdmissionHook[
                ExecutionRequest[CatalogToolChildInvokeWork, ToolExecutionResult[BaseModel]]
            ],
            ...,
        ] = (),
    ) -> ToolExecutionResult[BaseModel]:
        request = ExecutionRequest(input=work)
        return await self._child_runner.execute(
            request=request,
            delegate=self._delegate,
            admission_hooks=admission_hooks,
        )


def catalog_tool_child_execution_work_port(
    *,
    ledger: ExecutionBudgetLedger | None = None,
    child_context_inheritance: ChildExecutionContextInheritancePort | None = None,
) -> CatalogToolChildExecutionWorkPort:
    resolved_ledger = ledger if ledger is not None else create_execution_budget_ledger(None)
    return CatalogToolChildExecutionWorkPort(
        ledger=resolved_ledger,
        child_context_inheritance=child_context_inheritance,
    )


__all__ = [
    "CatalogToolChildInvokeWork",
    "CatalogToolChildExecutionWorkPort",
    "catalog_tool_child_execution_work_port",
]
