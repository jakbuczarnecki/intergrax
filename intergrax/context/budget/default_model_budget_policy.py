# © Artur Czarnecki. All rights reserved.

"""Default deterministic model budget policy (CE-02)."""

from __future__ import annotations

from intergrax.context.budget.contracts import ContextBudgetResolveInput, ResolvedModelContextBudget
from intergrax.context.budget.model_budget_policy import ContextModelBudgetPolicy


class DefaultContextModelBudgetPolicy:
    """Caps allocatable tokens by capability and optional request snapshot."""

    @property
    def policy_id(self) -> str:
        return "default_context_model_budget_policy"

    @property
    def policy_version(self) -> str:
        return "1"

    def resolve_budget(self, inputs: ContextBudgetResolveInput) -> ResolvedModelContextBudget:
        capability = inputs.capability
        available = capability.available_input_tokens
        request_cap = inputs.request_budget.max_tokens_estimate
        effective = min(available, request_cap) if request_cap > 0 else available
        mandatory_reserve = max(0, inputs.mandatory_reserve_tokens)
        allocatable = max(0, effective - mandatory_reserve)
        return ResolvedModelContextBudget(
            model_context_window=capability.model_context_window,
            reserved_output_tokens=capability.reserved_output_tokens,
            platform_margin_tokens=capability.platform_margin_tokens,
            available_input_tokens=effective,
            mandatory_reserve_tokens=mandatory_reserve,
            allocatable_tokens=allocatable,
            request_cap_tokens=request_cap,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
        )
