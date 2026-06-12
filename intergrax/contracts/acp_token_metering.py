# © Artur Czarnecki. All rights reserved.

"""ACP token metering rollups (architecture §25.4 · ACP-TOK-1)."""

from __future__ import annotations

from typing import Any

from intergrax.contracts.acp_state import (
    ACP_USAGE_KEY,
    AcpBudgetState,
    AcpInvocationUsageView,
    AcpTokenUsage,
)
from intergrax.contracts.agent_budget import ResolvedBudgetLimits
from intergrax.contracts.agent_run_trace import LlmCallRecord

ACP_USAGE_SCHEMA_VERSION = "acp.usage.v1"


def seed_budget_limits_on_state(
    acp_state: dict[str, Any],
    limits: ResolvedBudgetLimits,
) -> dict[str, Any]:
    """Attach resolved agent limits to ``acp.state.v1.budget`` before the first step."""
    budget_raw = acp_state.get("budget")
    budget = (
        AcpBudgetState.model_validate(budget_raw)
        if isinstance(budget_raw, dict)
        else AcpBudgetState()
    )
    budget = budget.model_copy(
        update={
            "tokens_limit": limits.agent_tokens_limit,
            "tokens_remaining": limits.agent_tokens_remaining,
        }
    )
    merged = dict(acp_state)
    merged["budget"] = budget.model_dump(mode="json")
    return merged


def load_environment_usage(metadata: dict[str, Any]) -> AcpTokenUsage:
    raw = metadata.get(ACP_USAGE_KEY)
    if isinstance(raw, dict):
        env_raw = raw.get("environment", raw)
        if isinstance(env_raw, dict):
            return AcpTokenUsage.model_validate(env_raw)
    return AcpTokenUsage()


def persist_environment_usage(metadata: dict[str, Any], usage: AcpTokenUsage) -> None:
    metadata[ACP_USAGE_KEY] = {
        "schema_version": ACP_USAGE_SCHEMA_VERSION,
        "environment": usage.model_dump(mode="json"),
    }


def increment_token_usage(
    usage: AcpTokenUsage,
    llm_calls: list[LlmCallRecord],
    *,
    tokens_limit: int | None = None,
) -> AcpTokenUsage:
    tokens_in = usage.tokens_in
    tokens_out = usage.tokens_out
    llm_call_count = usage.llm_calls
    for call in llm_calls:
        tokens_in += call.tokens_in
        tokens_out += call.tokens_out
        llm_call_count += 1
    tokens_total = tokens_in + tokens_out
    limit = tokens_limit if tokens_limit is not None else usage.tokens_limit
    return usage.model_copy(
        update={
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "tokens_total": tokens_total,
            "llm_calls": llm_call_count,
            "tokens_limit": limit,
            "tokens_remaining": tokens_remaining(limit, tokens_total),
        }
    )


def increment_budget_from_llm_calls(
    budget: AcpBudgetState,
    llm_calls: list[LlmCallRecord],
) -> AcpBudgetState:
    usage = increment_token_usage(
        AcpTokenUsage(
            tokens_in=budget.tokens_in,
            tokens_out=budget.tokens_out,
            tokens_total=budget.tokens_total,
            tokens_limit=budget.tokens_limit,
            tokens_remaining=budget.tokens_remaining,
            llm_calls=budget.llm_calls,
            cost_usd=budget.cost_usd,
        ),
        llm_calls,
        tokens_limit=budget.tokens_limit,
    )
    return budget.model_copy(
        update={
            "tokens_in": usage.tokens_in,
            "tokens_out": usage.tokens_out,
            "tokens_total": usage.tokens_total,
            "tokens_limit": usage.tokens_limit,
            "tokens_remaining": usage.tokens_remaining,
            "llm_calls": usage.llm_calls,
        }
    )


def build_invocation_usage_view(
    *,
    agent_budget: AcpBudgetState,
    environment_usage: AcpTokenUsage,
    limits: ResolvedBudgetLimits,
) -> AcpInvocationUsageView:
    agent = AcpTokenUsage(
        tokens_in=agent_budget.tokens_in,
        tokens_out=agent_budget.tokens_out,
        tokens_total=agent_budget.tokens_total,
        tokens_limit=agent_budget.tokens_limit,
        tokens_remaining=agent_budget.tokens_remaining,
        llm_calls=agent_budget.llm_calls,
        cost_usd=agent_budget.cost_usd,
    )
    environment = environment_usage.model_copy(
        update={
            "tokens_limit": limits.environment_tokens_limit,
            "tokens_remaining": tokens_remaining(
                limits.environment_tokens_limit,
                environment_usage.tokens_total,
            ),
        }
    )
    return AcpInvocationUsageView(agent=agent, environment=environment)


def tokens_remaining(limit: int | None, tokens_total: int = 0) -> int | None:
    if limit is None:
        return None
    return max(0, limit - tokens_total)
