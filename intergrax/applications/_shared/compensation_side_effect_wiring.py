# © Artur Czarnecki. All rights reserved.

"""Tier-3 wiring for admitted compensation side-effect execution (U2)."""

from __future__ import annotations

from intergrax.agents.persistence.compensation_tool_invoke_session import (
    bound_compensation_tool_invoke_session,
)
from intergrax.contracts.compensation_side_effect_execution import (
    CompensationSideEffectExecutionPort,
)
from intergrax.contracts.execution_bound_declarative_tool_invocation import (
    ExecutionBoundDeclarativeToolInvoker,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_lineage import ExecutionLineagePersistence
from intergrax.runtime.execution.budget.ledger import ExecutionBudgetLedgerFactory
from intergrax.runtime.execution.compensation_side_effect import (
    build_runtime_compensation_side_effect_execution,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget


def build_compensation_side_effect_execution(
    invoker: ExecutionBoundDeclarativeToolInvoker,
    *,
    authority: ParentExecutionAuthority | None = None,
    ledger_factory: ExecutionBudgetLedgerFactory | None = None,
    run_budget: RunBudget | None = None,
    execution_lineage_persistence: ExecutionLineagePersistence | None = None,
) -> CompensationSideEffectExecutionPort:
    """Compose canonical compensation admission from execution stack inputs + catalog invoker."""
    resolved_authority = authority or ParentExecutionAuthority.unrestricted_root()
    return build_runtime_compensation_side_effect_execution(
        tool_session=bound_compensation_tool_invoke_session(invoker),
        authority=resolved_authority,
        ledger_factory=ledger_factory,
        run_budget=run_budget,
        execution_lineage_persistence=execution_lineage_persistence,
    )
