# © Artur Czarnecki. All rights reserved.

"""Resolve merged token budget limits for agent sessions (§25.5.1 · ACP-TOK-1c)."""

from __future__ import annotations

from typing import Literal

from intergrax.contracts.agent_run_binding import AgentRunBinding
from intergrax.contracts.runtime_environment import RuntimeEnvironmentProfile
from intergrax.contracts.acp_token_metering import tokens_remaining as _tokens_remaining
from intergrax.contracts.agent_budget import BudgetLimitEnforcement, ResolvedBudgetLimits
from intergrax.contracts.agent_run import AgentExecutionOptions


def resolve_budget_limits(
    *,
    app_profile: RuntimeEnvironmentProfile | None,
    binding: AgentRunBinding | None,
    execution_options: AgentExecutionOptions | None,
) -> ResolvedBudgetLimits:
    """Materialize read-only limits during ``merge_environment``."""
    warn_ratio = 0.80
    agent_limit: int | None = None
    env_limit: int | None = None
    agent_enforcement = BudgetLimitEnforcement.ADVISORY
    env_enforcement = BudgetLimitEnforcement.ADVISORY
    limit_source: Literal["none", "binding", "environment", "request", "merged"] = "none"

    if app_profile is not None:
        env_limit = app_profile.cost_profile.max_total_tokens
        reaction = app_profile.cost_profile.budget_reaction
        if reaction is not None:
            warn_ratio = reaction.warn_threshold_ratio
        if env_limit is not None:
            env_enforcement = BudgetLimitEnforcement.HARD
            limit_source = "environment"

    if binding is not None and binding.budget_slice is not None:
        budget_slice = binding.budget_slice
        if budget_slice.max_total_tokens is not None:
            agent_limit = budget_slice.max_total_tokens
            agent_enforcement = budget_slice.enforcement
            limit_source = "binding"
        if budget_slice.warn_threshold_ratio is not None:
            warn_ratio = budget_slice.warn_threshold_ratio

    if execution_options is not None and execution_options.max_total_tokens is not None:
        agent_limit = execution_options.max_total_tokens
        agent_enforcement = BudgetLimitEnforcement.HARD
        limit_source = "request"

    return ResolvedBudgetLimits(
        agent_tokens_limit=agent_limit,
        agent_tokens_remaining=_tokens_remaining(agent_limit),
        agent_enforcement=(
            agent_enforcement if agent_limit is not None else BudgetLimitEnforcement.ADVISORY
        ),
        environment_tokens_limit=env_limit,
        environment_tokens_remaining=_tokens_remaining(env_limit),
        environment_enforcement=(
            env_enforcement if env_limit is not None else BudgetLimitEnforcement.ADVISORY
        ),
        warn_threshold_ratio=warn_ratio,
        limit_source=limit_source,
    )
