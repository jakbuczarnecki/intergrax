# © Artur Czarnecki. All rights reserved.

"""Composition-root wiring for canonical delegated-subtask child execution (U4 / EP-15)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.agent_distribution.delegated_subtasks import ChildExecutionPort
from intergrax.runtime.execution.budget import (
    ExecutionBudgetLedger,
    create_execution_budget_ledger,
)
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.execution.delegated_subtask_child_port import as_child_execution_port
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.nexus_loop import NexusLoop


@dataclass(frozen=True, slots=True)
class ProductionDelegatedSubtaskChildExecutionPort:
    """
    Process-lifetime child execution binding for delegated subtasks.

    Uses the same :class:`ChildExecutionRunner` admission plane as
    :class:`intergrax.runtime.execution.execution_work_port.ChildExecutionWorkPort`.
    """

    _runner: ChildExecutionRunner

    def port[RequestT, ResultT](self) -> ChildExecutionPort[RequestT, ResultT]:
        return as_child_execution_port(self._runner)


def build_production_delegated_subtask_child_execution_port(
    *,
    nexus_loop: NexusLoop | None = None,
    run_budget: RunBudget | None = None,
    ledger: ExecutionBudgetLedger | None = None,
) -> ProductionDelegatedSubtaskChildExecutionPort:
    """
    Build canonical ``ChildExecutionPort`` at the production composition root.

    When ``nexus_loop`` is supplied, budget limits align with the active Nexus run budget
    (same composition seam as ``build_compensation_side_effect_execution``).
    """
    resolved_budget = run_budget
    if nexus_loop is not None:
        resolved_budget = nexus_loop.run_budget
    resolved_ledger = (
        ledger
        if ledger is not None
        else create_execution_budget_ledger(resolved_budget)
    )
    runner = ChildExecutionRunner(ledger=resolved_ledger)
    return ProductionDelegatedSubtaskChildExecutionPort(_runner=runner)


__all__ = [
    "ProductionDelegatedSubtaskChildExecutionPort",
    "build_production_delegated_subtask_child_execution_port",
]
