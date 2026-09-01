# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neutral child execution work submission seam (DS-NEXUS-01)."""

from __future__ import annotations

from typing import Generic, Protocol, TypeVar

from intergrax.runtime.execution.boundary import ExecutionDelegate
from intergrax.runtime.execution.budget.ledger import ExecutionBudgetLedger
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.execution.request import ExecutionRequest

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
ResultT = TypeVar("ResultT", covariant=True)


class ExecutionWorkPort(Protocol[InputT, OutputT, ResultT]):
    """Typed neutral port for submitting canonical Execution work from active Execution."""

    async def execute(
        self,
        request: ExecutionRequest[InputT, OutputT],
    ) -> ResultT:
        ...


class ChildExecutionWorkPort(Generic[InputT, OutputT, ResultT]):
    """
    Submit canonical Execution work as a child Execution under the active parent.

    Physical strategy routing is supplied by the wired Execution delegate at
    composition root — typically :class:`StrategyExecutionRouter`.
    """

    __slots__ = ("_child_runner", "_delegate")

    def __init__(
        self,
        delegate: ExecutionDelegate[ExecutionRequest[InputT, OutputT], ResultT],
        *,
        ledger: ExecutionBudgetLedger | None = None,
    ) -> None:
        self._child_runner = ChildExecutionRunner[
            ExecutionRequest[InputT, OutputT],
            ResultT,
        ](ledger=ledger)
        self._delegate = delegate

    async def execute(
        self,
        request: ExecutionRequest[InputT, OutputT],
    ) -> ResultT:
        return await self._child_runner.execute(
            request=request,
            delegate=self._delegate,
        )


def child_execution_work_port(
    delegate: ExecutionDelegate[ExecutionRequest[InputT, OutputT], ResultT],
    *,
    ledger: ExecutionBudgetLedger | None = None,
) -> ChildExecutionWorkPort[InputT, OutputT, ResultT]:
    """Build a child-work port backed by canonical child execution lineage."""
    return ChildExecutionWorkPort(delegate, ledger=ledger)
