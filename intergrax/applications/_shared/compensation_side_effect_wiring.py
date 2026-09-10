# © Artur Czarnecki. All rights reserved.

"""Tier-3 wiring for admitted compensation side-effect execution (U2)."""

from __future__ import annotations

from intergrax.agents.persistence.compensation_tool_invoke_session import (
    bound_compensation_tool_invoke_session,
)
from intergrax.agents.persistence.declarative_tool_executor import DeclarativeToolInvoker
from intergrax.contracts.compensation_side_effect_execution import (
    CompensationSideEffectExecutionPort,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.runtime.execution.compensation_side_effect import (
    build_runtime_compensation_side_effect_execution,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop


def build_compensation_side_effect_execution(
    nexus_loop: NexusLoop,
    invoker: DeclarativeToolInvoker,
    *,
    authority: ParentExecutionAuthority | None = None,
) -> CompensationSideEffectExecutionPort:
    """Compose canonical compensation admission from Nexus execution stack + catalog invoker."""
    resolved_authority = authority or ParentExecutionAuthority.unrestricted_root()
    return build_runtime_compensation_side_effect_execution(
        tool_session=bound_compensation_tool_invoke_session(invoker),
        authority=resolved_authority,
        ledger_factory=nexus_loop.execution_budget_ledger_factory,
        run_budget=nexus_loop.run_budget,
        execution_lineage_persistence=nexus_loop.execution_lineage_persistence,
    )
