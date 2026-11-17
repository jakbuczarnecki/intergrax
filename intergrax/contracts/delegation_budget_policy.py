# © Artur Czarnecki. All rights reserved.

"""Neutral delegation budget policy applied to delegated subtask contracts."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.subtask_contract import SubtaskContract


@dataclass(frozen=True, slots=True)
class DelegationBudgetPolicy:
    max_llm_calls: int | None
    max_tool_calls: int | None
    max_delegation_depth: int
    enforcement_enabled: bool


def apply_delegation_budget_to_subtask(
    contract: SubtaskContract,
    policy: DelegationBudgetPolicy,
) -> SubtaskContract:
    """Attach delegation budgets when enforcement is enabled."""
    if not policy.enforcement_enabled:
        return contract
    return contract.model_copy(
        update={
            "max_llm_calls": policy.max_llm_calls,
            "max_tool_calls": policy.max_tool_calls,
        },
    )


__all__ = ["DelegationBudgetPolicy", "apply_delegation_budget_to_subtask"]
