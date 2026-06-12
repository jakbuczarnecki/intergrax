# © Artur Czarnecki. All rights reserved.

"""Harness bridge for ACP token metering (§25.4 · ACP-TOK-1)."""

from __future__ import annotations

from typing import Any

from intergrax.agents.authoring.state_merge import (
    ACP_STATE_KEY,
    extract_acp_state_blob,
    wrap_acp_state_blob,
)
from intergrax.contracts.acp_state import AcpBudgetState, AcpInvocationUsageView
from intergrax.contracts.acp_token_metering import (
    build_invocation_usage_view,
    increment_budget_from_llm_calls,
    increment_token_usage,
    load_environment_usage,
    persist_environment_usage,
    seed_budget_limits_on_state,
)
from intergrax.contracts.agent_budget import ResolvedBudgetLimits
from intergrax.contracts.agent_run_trace import LlmCallRecord


def apply_llm_metering_after_step(
    *,
    state_root: dict[str, Any],
    step_metadata: dict[str, Any],
    llm_calls: list[LlmCallRecord],
    limits: ResolvedBudgetLimits,
) -> tuple[dict[str, Any], AcpInvocationUsageView]:
    """Increment agent + environment rollups after LLM calls are drained."""
    acp_blob = extract_acp_state_blob(state_root)
    budget_raw = acp_blob.get("budget")
    if isinstance(budget_raw, dict):
        budget = AcpBudgetState.model_validate(budget_raw)
    elif limits.agent_tokens_limit is not None:
        budget = AcpBudgetState(
            tokens_limit=limits.agent_tokens_limit,
            tokens_remaining=limits.agent_tokens_remaining,
        )
    else:
        seeded = seed_budget_limits_on_state(acp_blob, limits)
        budget = AcpBudgetState.model_validate(seeded.get("budget", {}))

    if budget.tokens_limit is None and limits.agent_tokens_limit is not None:
        budget = budget.model_copy(
            update={
                "tokens_limit": limits.agent_tokens_limit,
                "tokens_remaining": limits.agent_tokens_remaining,
            }
        )

    updated_budget = (
        increment_budget_from_llm_calls(budget, llm_calls) if llm_calls else budget
    )
    updated_acp = dict(acp_blob)
    updated_acp["budget"] = updated_budget.model_dump(mode="json")
    updated_root = (
        wrap_acp_state_blob(updated_acp) if ACP_STATE_KEY in state_root else updated_acp
    )

    env_usage = load_environment_usage(step_metadata)
    if llm_calls:
        env_usage = increment_token_usage(
            env_usage,
            llm_calls,
            tokens_limit=limits.environment_tokens_limit,
        )
        persist_environment_usage(step_metadata, env_usage)
    else:
        env_usage = env_usage.model_copy(
            update={
                "tokens_limit": limits.environment_tokens_limit,
            }
        )

    view = build_invocation_usage_view(
        agent_budget=updated_budget,
        environment_usage=env_usage,
        limits=limits,
    )
    return updated_root, view


def seed_state_root_budget_limits(
    state_root: dict[str, Any],
    limits: ResolvedBudgetLimits,
) -> dict[str, Any]:
    acp_blob = extract_acp_state_blob(state_root)
    updated_acp = seed_budget_limits_on_state(acp_blob, limits)
    if ACP_STATE_KEY in state_root:
        return wrap_acp_state_blob(updated_acp)
    return updated_acp


def initial_invocation_usage(
    state_root: dict[str, Any],
    step_metadata: dict[str, Any],
    limits: ResolvedBudgetLimits,
) -> AcpInvocationUsageView:
    acp_blob = extract_acp_state_blob(state_root)
    budget_raw = acp_blob.get("budget")
    budget = (
        AcpBudgetState.model_validate(budget_raw)
        if isinstance(budget_raw, dict)
        else AcpBudgetState(
            tokens_limit=limits.agent_tokens_limit,
            tokens_remaining=limits.agent_tokens_remaining,
        )
    )
    env_usage = load_environment_usage(step_metadata)
    return build_invocation_usage_view(
        agent_budget=budget,
        environment_usage=env_usage,
        limits=limits,
    )
