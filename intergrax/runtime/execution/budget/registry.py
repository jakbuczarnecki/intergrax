# © Artur Czarnecki. All rights reserved.

"""Entry-point registry and resolver for ``ExecutionBudgetAllocationPolicy`` plugins (UE-8B1)."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import TYPE_CHECKING

from intergrax.runtime.execution.budget.policy import (
    DefaultSharedPoolBudgetPolicy,
    ExecutionBudgetAllocationPolicy,
)

if TYPE_CHECKING:
    from intergrax.runtime.nexus.config import RuntimeConfig

_ENTRY_POINT_GROUP = "intergrax.execution_budget_allocation_policies"


class ExecutionBudgetAllocationPolicyConfigurationError(RuntimeError):
    """Raised when an explicitly configured budget allocation policy cannot be loaded."""


def load_execution_budget_allocation_policy(
    policy_id: str,
) -> ExecutionBudgetAllocationPolicy | None:
    """Load a policy by entry-point name from ``intergrax.execution_budget_allocation_policies``."""
    try:
        eps = entry_points(group=_ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover — Python 3.11
        eps = entry_points().select(group=_ENTRY_POINT_GROUP)

    for ep in eps:
        if ep.name != policy_id:
            continue
        loaded = ep.load()
        if isinstance(loaded, type):
            instance: ExecutionBudgetAllocationPolicy = loaded()
        else:
            instance = loaded
        if not isinstance(instance, ExecutionBudgetAllocationPolicy):
            raise TypeError(
                f"execution budget allocation entry point {ep.name!r} must return "
                "ExecutionBudgetAllocationPolicy"
            )
        return instance
    return None


def list_execution_budget_allocation_policy_ids() -> tuple[str, ...]:
    """Return registered entry-point policy ids (sorted)."""
    try:
        eps = entry_points(group=_ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover — Python 3.11
        eps = entry_points().select(group=_ENTRY_POINT_GROUP)
    return tuple(sorted(ep.name for ep in eps))


def resolve_execution_budget_allocation_policy(
    *,
    policy_override: ExecutionBudgetAllocationPolicy | None = None,
    entry_point_policy_id: str | None = None,
) -> ExecutionBudgetAllocationPolicy:
    """
    Resolve budget allocation policy from explicit instance, entry-point id, or default.

    Explicit ``entry_point_policy_id`` fails closed when the plugin is missing or invalid.
    """
    if policy_override is not None:
        if not isinstance(policy_override, ExecutionBudgetAllocationPolicy):
            raise TypeError("policy_override must satisfy ExecutionBudgetAllocationPolicy")
        return policy_override
    if entry_point_policy_id:
        loaded = load_execution_budget_allocation_policy(entry_point_policy_id)
        if loaded is None:
            raise ExecutionBudgetAllocationPolicyConfigurationError(
                f"execution budget allocation policy entry point "
                f"{entry_point_policy_id!r} not found"
            )
        return loaded
    return DefaultSharedPoolBudgetPolicy()


def resolve_execution_budget_allocation_policy_from_runtime_config(
    config: RuntimeConfig,
) -> ExecutionBudgetAllocationPolicy:
    """Resolve budget allocation policy from ``RuntimeConfig`` execution budget fields."""
    return resolve_execution_budget_allocation_policy(
        policy_override=config.execution_budget_allocation_policy,
        entry_point_policy_id=config.execution_budget_allocation_policy_id,
    )
