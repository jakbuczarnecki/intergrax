# © Artur Czarnecki. All rights reserved.

"""Hard/advisory token budget enforcement helpers (§25.5 · ACP-TOK-2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.acp_state import AcpInvocationUsageView, AcpTokenUsage
from intergrax.contracts.agent_budget import BudgetLimitEnforcement, ResolvedBudgetLimits
from intergrax.contracts.agent_run_enums import AgentRunErrorCode, TerminalReason
from typing import Any


class BudgetScope(StrEnum):
    AGENT = "agent"
    ENVIRONMENT = "environment"


@dataclass(frozen=True, slots=True)
class HardBudgetViolation:
    """Raised when a hard token cap blocks the next LLM invocation."""

    scope: BudgetScope
    tokens_total: int
    tokens_limit: int
    limit_source: str


def _scope_blocked(
    usage: AcpTokenUsage,
    *,
    tokens_limit: int | None,
    enforcement: BudgetLimitEnforcement,
    pending_tokens: int,
) -> bool:
    if tokens_limit is None or enforcement != BudgetLimitEnforcement.HARD:
        return False
    projected = usage.tokens_total + pending_tokens
    return projected >= tokens_limit


def evaluate_hard_budget_violation(
    invocation_usage: AcpInvocationUsageView | None,
    limits: ResolvedBudgetLimits,
    *,
    pending_agent_tokens: int = 0,
    pending_environment_tokens: int | None = None,
) -> HardBudgetViolation | None:
    """Return a violation when the next LLM call must be blocked under hard enforcement."""
    if invocation_usage is None:
        return None
    env_pending = (
        pending_environment_tokens
        if pending_environment_tokens is not None
        else pending_agent_tokens
    )
    if _scope_blocked(
        invocation_usage.agent,
        tokens_limit=limits.agent_tokens_limit,
        enforcement=limits.agent_enforcement,
        pending_tokens=pending_agent_tokens,
    ):
        assert limits.agent_tokens_limit is not None
        return HardBudgetViolation(
            scope=BudgetScope.AGENT,
            tokens_total=invocation_usage.agent.tokens_total + pending_agent_tokens,
            tokens_limit=limits.agent_tokens_limit,
            limit_source=limits.limit_source,
        )
    if _scope_blocked(
        invocation_usage.environment,
        tokens_limit=limits.environment_tokens_limit,
        enforcement=limits.environment_enforcement,
        pending_tokens=env_pending,
    ):
        assert limits.environment_tokens_limit is not None
        return HardBudgetViolation(
            scope=BudgetScope.ENVIRONMENT,
            tokens_total=invocation_usage.environment.tokens_total + env_pending,
            tokens_limit=limits.environment_tokens_limit,
            limit_source=limits.limit_source,
        )
    return None


def is_budget_exceeded_outcome(outcome: Any) -> bool:
    if getattr(outcome, "terminal_reason", None) == TerminalReason.BUDGET_EXCEEDED:
        return True
    errors = getattr(outcome, "errors", None) or []
    return any(
        getattr(error, "code", None) == AgentRunErrorCode.BUDGET_EXCEEDED for error in errors
    )
