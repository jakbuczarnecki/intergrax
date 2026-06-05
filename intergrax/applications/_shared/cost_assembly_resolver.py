# © Artur Czarnecki. All rights reserved.

"""Cost governance assembly validation for Tier-3 hosts (Phase COST-2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from intergrax.applications._shared.cost_wiring import ApplicationCostWiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


@dataclass(frozen=True, slots=True)
class CostAssemblyValidationResult:
    """Outcome of cost assembly validation."""

    valid: bool
    errors: tuple[str, ...] = ()


class CostAssemblyError(ValueError):
    """Raised when cost assembly validation fails."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors: tuple[str, ...] = tuple(errors)
        message = "; ".join(self.errors)
        super().__init__(message)


def _profile_has_explicit_limits(profile: ApplicationEnvironmentProfile) -> bool:
    cost = profile.cost_profile
    return any(
        (
            cost.max_total_tokens,
            cost.max_llm_calls,
            cost.max_tool_calls,
            cost.max_planner_iterations,
        )
    )


def validate_cost_wiring(
    wiring: ApplicationCostWiring,
    env: ApplicationEnvironmentProfile,
) -> CostAssemblyValidationResult:
    """Validate cost artifacts match environment profile requirements."""
    errors: list[str] = []
    profile = env.cost_profile
    options = wiring.options

    if options.budget_enforcement_enabled != profile.budget_enforcement_enabled:
        errors.append("budget_enforcement_enabled mismatch between wiring and cost_profile")

    if profile.budget_enforcement_enabled and wiring.budget_policy is None:
        errors.append("budget_enforcement_enabled requires budget_policy")

    if not profile.budget_enforcement_enabled and wiring.budget_policy is not None:
        errors.append("budget_enforcement_enabled=False requires no budget_policy")

    if profile.budget_enforcement_enabled:
        has_limits = wiring.run_budget is not None or _profile_has_explicit_limits(env)
        has_context_budget = env.context_profile.budget_policy is not None
        if not has_limits and not has_context_budget:
            errors.append(
                "budget_enforcement_enabled requires explicit cost limits or context budget_policy",
            )

    if wiring.run_budget is not None and profile.max_total_tokens is not None:
        if wiring.run_budget.max_total_tokens != profile.max_total_tokens:
            errors.append("run_budget.max_total_tokens must match cost_profile.max_total_tokens")

    if (
        wiring.domain_fragments.get("cost_governance", {}).get("quota_degrade_threshold_ratio")
        != profile.quota_degrade_threshold_ratio
    ):
        errors.append("cost_governance domain fragment must match cost_profile quota threshold")

    return CostAssemblyValidationResult(valid=not errors, errors=tuple(errors))


def assert_cost_assembly_valid(
    wiring: ApplicationCostWiring,
    env: ApplicationEnvironmentProfile,
) -> None:
    """Raise :class:`CostAssemblyError` when cost validation fails."""
    result = validate_cost_wiring(wiring, env)
    if not result.valid:
        raise CostAssemblyError(result.errors)
