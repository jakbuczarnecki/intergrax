# © Artur Czarnecki. All rights reserved.

"""Composition-root wiring for in-process catalog tool child execution (OD-CHILD-IMPL)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.catalog_tool_child_execution import CatalogToolChildExecutionPort
from intergrax.runtime.execution.catalog_tool_child_work_port import CatalogToolChildInvokeWork
from intergrax.contracts.child_execution_context_inheritance import (
    ChildExecutionContextInheritancePort,
)
from intergrax.contracts.tool_execution_outcome import ToolExecutionResult
from intergrax.runtime.execution.budget import (
    ExecutionBudgetLedger,
    create_execution_budget_ledger,
)
from intergrax.runtime.execution.catalog_tool_child_work_port import (
    CatalogToolChildExecutionWorkPort,
    catalog_tool_child_execution_work_port,
)
from intergrax.contracts.run_budget import RunBudget
from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class ProductionCatalogToolChildExecutionPort:
    """Process-lifetime binding for catalog child work — owns work port, not ChildExecutionRunner."""

    _work_port: CatalogToolChildExecutionWorkPort

    async def execute_catalog_child_invoke(
        self,
        work: CatalogToolChildInvokeWork,
        *,
        admission_hooks: tuple[object, ...] = (),
    ) -> ToolExecutionResult[BaseModel]:
        return await self._work_port.execute(
            work=work,
            admission_hooks=admission_hooks,  # type: ignore[arg-type]
        )


def build_production_catalog_tool_child_execution_port(
    *,
    run_budget: RunBudget | None = None,
    ledger: ExecutionBudgetLedger | None = None,
    child_context_inheritance: ChildExecutionContextInheritancePort | None = None,
) -> CatalogToolChildExecutionPort:
    resolved_ledger = (
        ledger if ledger is not None else create_execution_budget_ledger(run_budget)
    )
    work_port = catalog_tool_child_execution_work_port(
        ledger=resolved_ledger,
        child_context_inheritance=child_context_inheritance,
    )
    return ProductionCatalogToolChildExecutionPort(_work_port=work_port)


__all__ = [
    "ProductionCatalogToolChildExecutionPort",
    "build_production_catalog_tool_child_execution_port",
]
