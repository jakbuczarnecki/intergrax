# © Artur Czarnecki. All rights reserved.

"""Composition-root wiring for canonical delegated-subtask child execution (U4 / EP-15)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.agent_distribution.delegated_subtasks import ChildExecutionPort
from intergrax.runtime.execution.budget import (
    ExecutionBudgetLedger,
    create_execution_budget_ledger,
)
from intergrax.runtime.execution.delegated_subtask_child_port import (
    child_execution_port_from_work_port,
)
from intergrax.runtime.execution.execution_work_port import (
    DelegatedSubtaskChildExecutionWorkPort,
    delegated_subtask_child_execution_work_port,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget


@dataclass(frozen=True, slots=True)
class ProductionDelegatedSubtaskChildExecutionPort:
    """
    Process-lifetime child execution binding for delegated subtasks.

    Owns a canonical :class:`DelegatedSubtaskChildExecutionWorkPort` — not
    :class:`intergrax.runtime.execution.child.ChildExecutionRunner`.
    """

    _work_port: DelegatedSubtaskChildExecutionWorkPort

    def port[RequestT, ResultT](self) -> ChildExecutionPort[RequestT, ResultT]:
        return child_execution_port_from_work_port(self._work_port)


def build_production_delegated_subtask_child_execution_port(
    *,
    run_budget: RunBudget | None = None,
    ledger: ExecutionBudgetLedger | None = None,
) -> ProductionDelegatedSubtaskChildExecutionPort:
    """
    Build canonical ``ChildExecutionPort`` at the production composition root.

    ``run_budget`` aligns ledger limits with the active Nexus run budget when supplied
    by the composition owner (budget alignment only — not Nexus child scheduling).
    """
    resolved_ledger = (
        ledger if ledger is not None else create_execution_budget_ledger(run_budget)
    )
    work_port = delegated_subtask_child_execution_work_port(ledger=resolved_ledger)
    return ProductionDelegatedSubtaskChildExecutionPort(_work_port=work_port)


__all__ = [
    "ProductionDelegatedSubtaskChildExecutionPort",
    "build_production_delegated_subtask_child_execution_port",
]
