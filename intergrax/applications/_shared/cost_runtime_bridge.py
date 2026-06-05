# © Artur Czarnecki. All rights reserved.

"""Map ApplicationEnvironmentProfile cost fields to Nexus budget config (Phase COST-1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    CostProfile,
)
from intergrax.runtime.nexus.budget.budget_models import (
    BudgetEnforcementMode,
    BudgetPolicy,
    RunBudget,
)
from intergrax.runtime.nexus.config import RuntimeConfig


@dataclass(frozen=True, slots=True)
class CostWiringOptions:
    """Resolved cost wiring flags for Tier-3 hosts."""

    budget_enforcement_enabled: bool
    enforcement_mode: Literal["abort", "hitl"]
    max_total_tokens: int | None
    max_llm_calls: int | None
    max_tool_calls: int | None
    max_planner_iterations: int | None
    quota_degrade_threshold_ratio: float


def resolve_cost_wiring_options(profile: CostProfile) -> CostWiringOptions:
    """Translate ``CostProfile`` into host wiring flags."""
    return CostWiringOptions(
        budget_enforcement_enabled=profile.budget_enforcement_enabled,
        enforcement_mode=profile.enforcement_mode,
        max_total_tokens=profile.max_total_tokens,
        max_llm_calls=profile.max_llm_calls,
        max_tool_calls=profile.max_tool_calls,
        max_planner_iterations=profile.max_planner_iterations,
        quota_degrade_threshold_ratio=profile.quota_degrade_threshold_ratio,
    )


def _enforcement_mode(mode: Literal["abort", "hitl"]) -> BudgetEnforcementMode:
    if mode == "hitl":
        return BudgetEnforcementMode.HITL
    return BudgetEnforcementMode.ABORT


def build_budget_policy_from_cost_profile(profile: CostProfile) -> BudgetPolicy | None:
    """Build Nexus ``BudgetPolicy`` when enforcement is enabled."""
    if not profile.budget_enforcement_enabled:
        return None
    return BudgetPolicy(enforcement_mode=_enforcement_mode(profile.enforcement_mode))


def build_run_budget_from_cost_profile(profile: CostProfile) -> RunBudget | None:
    """Build Nexus ``RunBudget`` from explicit cost profile limits."""
    if not any(
        (
            profile.max_total_tokens,
            profile.max_llm_calls,
            profile.max_tool_calls,
            profile.max_planner_iterations,
        )
    ):
        return None
    return RunBudget(
        max_total_tokens=profile.max_total_tokens,
        max_llm_calls=profile.max_llm_calls,
        max_tool_calls=profile.max_tool_calls,
        max_planner_iterations=profile.max_planner_iterations,
    )


def merge_run_budget(existing: RunBudget | None, profile: CostProfile) -> RunBudget | None:
    """Overlay explicit cost limits onto an existing run budget."""
    explicit = build_run_budget_from_cost_profile(profile)
    if explicit is None:
        return existing
    if existing is None:
        return explicit
    return RunBudget(
        max_input_tokens=existing.max_input_tokens,
        max_output_tokens=existing.max_output_tokens,
        max_total_tokens=explicit.max_total_tokens or existing.max_total_tokens,
        max_llm_calls=explicit.max_llm_calls or existing.max_llm_calls,
        max_tool_calls=explicit.max_tool_calls or existing.max_tool_calls,
        max_rag_invocations=existing.max_rag_invocations,
        max_websearch_invocations=existing.max_websearch_invocations,
        max_wall_time_seconds=existing.max_wall_time_seconds,
        max_planner_iterations=explicit.max_planner_iterations or existing.max_planner_iterations,
        max_replans=existing.max_replans,
    )


def apply_cost_profile_to_runtime_config(
    config: RuntimeConfig,
    profile: CostProfile,
) -> RuntimeConfig:
    """Apply cost governance posture to runtime config."""
    budget_policy = build_budget_policy_from_cost_profile(profile)
    if budget_policy is not None:
        config.budget_policy = budget_policy
    elif not profile.budget_enforcement_enabled:
        config.budget_policy = None

    config.run_budget = merge_run_budget(config.run_budget, profile)
    return config


def apply_cost_profiles_from_environment(
    config: RuntimeConfig,
    env: ApplicationEnvironmentProfile,
) -> RuntimeConfig:
    """Apply environment-declared cost profile after context budget derivation."""
    return apply_cost_profile_to_runtime_config(config, env.cost_profile)
