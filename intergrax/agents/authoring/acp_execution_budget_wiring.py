# © Artur Czarnecki. All rights reserved.

"""ACP composition wiring for execution budget ledger factory (UE-8B1)."""

from __future__ import annotations

from intergrax.agents.authoring.acp_session_host import ACPSessionHostContext
from intergrax.runtime.execution.budget import create_execution_budget_ledger_factory
from intergrax.runtime.execution.budget.ledger import ExecutionBudgetLedgerFactory

_DEFAULT_ACP_EXECUTION_BUDGET_LEDGER_FACTORY: ExecutionBudgetLedgerFactory = (
    create_execution_budget_ledger_factory(None)
)


def default_acp_execution_budget_ledger_factory() -> ExecutionBudgetLedgerFactory:
    """Default per-Run ledger factory for direct ACP entry without Tier-3 host slices."""
    return _DEFAULT_ACP_EXECUTION_BUDGET_LEDGER_FACTORY


def resolve_acp_execution_budget_ledger_factory(
    host: ACPSessionHostContext | None,
) -> ExecutionBudgetLedgerFactory:
    """Resolve configured factory from host context or ACP composition default."""
    if host is not None and host.execution_budget_ledger_factory is not None:
        return host.execution_budget_ledger_factory
    return default_acp_execution_budget_ledger_factory()
