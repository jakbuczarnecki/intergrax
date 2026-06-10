# © Artur Czarnecki. All rights reserved.

"""Delegation budget enforcement wiring for product graph specs (AUDIT-IDEAL-10.2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.subtask_contract import SubtaskContract


@dataclass(frozen=True, slots=True)
class DelegationBudgetPolicy:
    max_llm_calls: int | None
    max_tool_calls: int | None
    max_delegation_depth: int
    enforcement_enabled: bool


def resolve_delegation_budget_policy(
    env: ApplicationEnvironmentProfile,
) -> DelegationBudgetPolicy:
    """Mirror host cost + orchestration limits onto delegation paths."""
    is_reference = env.application_profile in (ApplicationProfile.PRODUCT, ApplicationProfile.LAB)
    cost = env.cost_profile
    orch = env.orchestration_profile
    enforcement_enabled = is_reference and cost.budget_enforcement_enabled
    return DelegationBudgetPolicy(
        max_llm_calls=cost.max_llm_calls,
        max_tool_calls=cost.max_tool_calls,
        max_delegation_depth=orch.max_delegation_depth,
        enforcement_enabled=enforcement_enabled,
    )


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
