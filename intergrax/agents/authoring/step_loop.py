# © Artur Czarnecki. All rights reserved.

"""Agent step loop runtime glue (architecture §32.4 · §38 · ACP-STEP-2)."""

from __future__ import annotations

import inspect
from typing import Protocol, runtime_checkable

from intergrax.agents.acp_budget_enforcement_bridge import (
    AcpBudgetExceededError,
    check_step_boundary_budget,
)
from intergrax.agents.acp_budget_reactions import handle_hard_budget_violation
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.step_execution import StepExecutionRecord
from intergrax.runtime.kernel.step_kernel import HarnessKernel, StepKernelContext


@runtime_checkable
class AgentWithNextStep(Protocol):
    async def on_next_step(self, step_ctx: AgentStepContext) -> StepOutcome: ...


class AgentRuntime:
    """
    Glue between domain hook and harness kernel.

    Invariant: no policy, trace, budget, or state-merge logic here — only delegation.
    """

    @staticmethod
    async def advance_step(
        agent: AgentWithNextStep,
        step_ctx: AgentStepContext,
        kernel_ctx: StepKernelContext,
    ) -> tuple[StepOutcome, StepExecutionRecord]:
        violation = check_step_boundary_budget(
            step_ctx,
            kernel_ctx.resolved_budget_limits,
        )
        if violation is not None:
            outcome = await handle_hard_budget_violation(violation, step_ctx, kernel_ctx)
            record = await HarnessKernel.execute_step(outcome, step_ctx, kernel_ctx)
            return outcome, record
        try:
            outcome = await agent.on_next_step(step_ctx)
        except AcpBudgetExceededError as exc:
            outcome = await handle_hard_budget_violation(
                exc.violation,
                step_ctx,
                kernel_ctx,
            )
            record = await HarnessKernel.execute_step(outcome, step_ctx, kernel_ctx)
            return outcome, record
        record = await HarnessKernel.execute_step(outcome, step_ctx, kernel_ctx)
        return outcome, record

    @staticmethod
    async def execute_next_step(
        agent: AgentWithNextStep,
        step_ctx: AgentStepContext,
        kernel_ctx: StepKernelContext,
    ) -> tuple[StepOutcome, StepExecutionRecord]:
        """Deprecated alias — use :meth:`advance_step`."""
        return await AgentRuntime.advance_step(agent, step_ctx, kernel_ctx)


def advance_step_is_glue_only() -> bool:
    """Static guard used in tests — advance_step must not import policy/trace sinks."""
    source = inspect.getsource(AgentRuntime.advance_step)
    forbidden = (
        "PolicyEngine",
        "RuntimePolicyEngine",
        "merge_session_state",
        "RuntimeEvent(",
        "TraceWriter",
    )
    return not any(token in source for token in forbidden)
