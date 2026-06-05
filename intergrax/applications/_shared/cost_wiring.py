# © Artur Czarnecki. All rights reserved.

"""Tier-3 cost governance wiring (Phase COST-1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.applications._shared.cost_runtime_bridge import (
    CostWiringOptions,
    build_budget_policy_from_cost_profile,
    build_run_budget_from_cost_profile,
    resolve_cost_wiring_options,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    CostProfile,
)
from intergrax.runtime.nexus.budget.budget_models import BudgetPolicy, RunBudget


@dataclass(frozen=True, slots=True)
class ApplicationCostWiring:
    """Resolved cost governance artifacts for a Tier-3 host."""

    profile: CostProfile
    options: CostWiringOptions
    budget_policy: BudgetPolicy | None
    run_budget: RunBudget | None
    domain_fragments: dict[str, Any]


def wire_application_cost(env: ApplicationEnvironmentProfile) -> ApplicationCostWiring:
    """Materialize budget policy and run budget from environment profile."""
    profile = env.cost_profile
    options = resolve_cost_wiring_options(profile)
    return ApplicationCostWiring(
        profile=profile,
        options=options,
        budget_policy=build_budget_policy_from_cost_profile(profile),
        run_budget=build_run_budget_from_cost_profile(profile),
        domain_fragments={
            "cost_governance": {
                "quota_degrade_threshold_ratio": profile.quota_degrade_threshold_ratio,
            },
        },
    )
