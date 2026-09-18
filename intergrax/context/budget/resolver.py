# © Artur Czarnecki. All rights reserved.

"""Single entry for authoritative global budget resolution (CE-02)."""

from __future__ import annotations

from intergrax.context.budget.contracts import (
    ContextBudgetResolveInput,
    ModelContextCapabilitySnapshot,
    ResolvedModelContextBudget,
)
from intergrax.context.budget.default_model_budget_policy import DefaultContextModelBudgetPolicy
from intergrax.context.budget.model_budget_policy import ContextModelBudgetPolicy
from intergrax.context.contracts import ContextAssemblyRequest, ContextBudgetSnapshot


def resolve_authoritative_model_budget(
    *,
    capability: ModelContextCapabilitySnapshot,
    request: ContextAssemblyRequest,
    policy: ContextModelBudgetPolicy | None = None,
    mandatory_reserve_tokens: int = 0,
) -> ResolvedModelContextBudget:
    """Canonical global budget owner — one resolution per assembly."""
    active = policy or DefaultContextModelBudgetPolicy()
    reserve = max(0, mandatory_reserve_tokens)
    resolved = active.resolve_budget(
        ContextBudgetResolveInput(
            capability=capability,
            request_budget=request.budget_policy,
            mandatory_reserve_tokens=reserve,
        ),
    )
    if reserve > resolved.available_input_tokens:
        from intergrax.context.budget.contracts import ContextBudgetUnsatisfiableError

        raise ContextBudgetUnsatisfiableError(
            detail="mandatory_reserve_exceeds_available_input",
            mandatory_tokens=reserve,
            available_tokens=resolved.available_input_tokens,
        )
    return resolved


def global_allocatable_tokens(resolved: ResolvedModelContextBudget) -> int:
    """Tokens available for ranked fragment allocation (policy pipeline)."""
    return resolved.allocatable_tokens
