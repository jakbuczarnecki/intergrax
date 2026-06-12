# © Artur Czarnecki. All rights reserved.

"""Budget enforcement bridge for ACP session loop (§25.5 · ACP-TOK-2)."""

from __future__ import annotations

from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.contracts.acp_budget_enforcement import (
    HardBudgetViolation,
    evaluate_hard_budget_violation,
)
from intergrax.contracts.agent_run import AgentRunError
from intergrax.contracts.agent_run_enums import AgentRunErrorCode, TerminalReason
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.agent_budget import ResolvedBudgetLimits


class AcpBudgetExceededError(Exception):
    """Raised by budget-aware LLM router when a hard cap blocks ``complete()``."""

    def __init__(self, violation: HardBudgetViolation) -> None:
        self.violation = violation
        super().__init__(
            f"hard budget exceeded ({violation.scope.value}): "
            f"{violation.tokens_total}/{violation.tokens_limit}"
        )


def budget_exceeded_outcome(violation: HardBudgetViolation) -> StepOutcome:
    return StepOutcome.fail(
        [
            AgentRunError(
                code=AgentRunErrorCode.BUDGET_EXCEEDED,
                message=(
                    f"{violation.scope.value} token budget exceeded: "
                    f"{violation.tokens_total}/{violation.tokens_limit}"
                ),
                details={
                    "scope": violation.scope.value,
                    "tokens_total": violation.tokens_total,
                    "tokens_limit": violation.tokens_limit,
                    "limit_source": violation.limit_source,
                },
            )
        ],
        terminal_reason=TerminalReason.BUDGET_EXCEEDED,
    )


def check_step_boundary_budget(
    step_ctx: AgentStepContext,
    limits: ResolvedBudgetLimits,
) -> HardBudgetViolation | None:
    """Pre-step check before ``on_next_step`` when prior usage already at hard cap."""
    return evaluate_hard_budget_violation(step_ctx.invocation_usage, limits)
