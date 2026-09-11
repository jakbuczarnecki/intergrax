# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neutral child execution work submission seam (DS-NEXUS-01)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from intergrax.agent_distribution.delegated_subtasks import (
    DelegatedChildExecutionOptions,
    DelegatedSubtaskDelegate,
)
from intergrax.runtime.execution.boundary import ExecutionDelegate
from intergrax.runtime.execution.budget.ledger import ExecutionBudgetLedger
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.execution.request import ExecutionRequest
from intergrax.runtime.nexus.budget.budget_models import RunBudget

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
ResultT = TypeVar("ResultT", covariant=True)
DelegatedRequestT = TypeVar("DelegatedRequestT")
DelegatedResultT = TypeVar("DelegatedResultT")


@dataclass(frozen=True, slots=True)
class DelegatedSpecialistChildWorkEnvelope(Generic[DelegatedRequestT, DelegatedResultT]):
    """Typed child-work payload for delegated specialist execution (AC-4 / U4)."""

    domain_request: DelegatedRequestT
    specialist: DelegatedSubtaskDelegate[DelegatedRequestT, DelegatedResultT]
    options: DelegatedChildExecutionOptions | None = None


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


class _DelegatedSpecialistEnvelopeDelegate(Generic[DelegatedRequestT, DelegatedResultT]):
    """Routes envelope-bearing execution requests to the per-invocation specialist."""

    async def execute(
        self,
        request: ExecutionRequest[
            DelegatedSpecialistChildWorkEnvelope[DelegatedRequestT, DelegatedResultT],
            DelegatedResultT,
        ],
    ) -> DelegatedResultT:
        envelope = request.input
        return await envelope.specialist.execute(envelope.domain_request)


class DelegatedSubtaskChildExecutionWorkPort(Generic[DelegatedRequestT, DelegatedResultT]):
    """
    Canonical child work owner for delegated specialist invocations (U4 / EP-15).

    Implements :class:`ExecutionWorkPort` with envelope-shaped requests so
    :class:`ChildExecutionRunner` remains an internal runtime detail.
    """

    __slots__ = ("_child_runner", "_delegate")

    def __init__(self, *, ledger: ExecutionBudgetLedger | None = None) -> None:
        self._delegate = _DelegatedSpecialistEnvelopeDelegate[
            DelegatedRequestT,
            DelegatedResultT,
        ]()
        self._child_runner = ChildExecutionRunner[
            ExecutionRequest[
                DelegatedSpecialistChildWorkEnvelope[DelegatedRequestT, DelegatedResultT],
                DelegatedResultT,
            ],
            DelegatedResultT,
        ](ledger=ledger)

    async def execute(
        self,
        request: ExecutionRequest[
            DelegatedSpecialistChildWorkEnvelope[DelegatedRequestT, DelegatedResultT],
            DelegatedResultT,
        ],
    ) -> DelegatedResultT:
        envelope = request.input
        options = envelope.options
        requested_permission_scopes = (
            None if options is None else options.requested_permission_scopes
        )
        requested_budget: RunBudget | None = None
        if options is not None and options.requested_budget is not None:
            budget = options.requested_budget
            if type(budget) is not RunBudget:
                raise TypeError(
                    "requested_budget must be RunBudget for DelegatedSubtaskChildExecutionWorkPort",
                )
            requested_budget = budget
        return await self._child_runner.execute(
            request=request,
            delegate=self._delegate,
            requested_permission_scopes=requested_permission_scopes,
            requested_budget=requested_budget,
        )


def delegated_subtask_child_execution_work_port(
    *,
    ledger: ExecutionBudgetLedger | None = None,
) -> DelegatedSubtaskChildExecutionWorkPort[DelegatedRequestT, DelegatedResultT]:
    """Build canonical delegated-subtask child work port at composition root."""
    return DelegatedSubtaskChildExecutionWorkPort(ledger=ledger)
