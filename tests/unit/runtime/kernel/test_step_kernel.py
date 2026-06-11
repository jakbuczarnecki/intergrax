# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.contracts.agent_run_enums import (
    AgentRunErrorCode,
    SideEffectMode,
    TerminalReason,
)
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.kernel.step_kernel import HarnessKernel, StepKernelContext
from intergrax.runtime.policy.policy_engine import PolicyEngine


@pytest.mark.unit
@pytest.mark.gate
async def test_kernel_policy_pre_deny() -> None:
    step_ctx = AgentStepContext(
        step_index=0,
        metadata={"policy_pre_deny": True},
    )
    kernel_ctx = StepKernelContext(
        agent_id="demo",
        run_id="run-1",
        policy_engine=PolicyEngine(),
    )
    outcome = StepOutcome.continue_with({"phase": "plan"})
    record = await HarnessKernel.execute_step(outcome, step_ctx, kernel_ctx)
    assert record.error_code == AgentRunErrorCode.POLICY_DENIED
    assert record.policy_pre is not None
    assert record.policy_pre.action == PolicyAction.DENY
    assert len(kernel_ctx.events) >= 1


@pytest.mark.unit
@pytest.mark.gate
async def test_kernel_budget_exceeded() -> None:
    step_ctx = AgentStepContext(step_index=2)
    kernel_ctx = StepKernelContext(agent_id="demo", run_id="run-1", max_steps=2)
    outcome = StepOutcome.continue_with({"phase": "execute"})
    record = await HarnessKernel.execute_step(outcome, step_ctx, kernel_ctx)
    assert record.budget_exceeded is True
    assert record.error_code == AgentRunErrorCode.MAX_STEPS_EXCEEDED


@pytest.mark.unit
@pytest.mark.gate
async def test_kernel_merges_state_and_appends_trace() -> None:
    step_ctx = AgentStepContext(step_index=0)
    kernel_ctx = StepKernelContext(
        agent_id="demo",
        run_id="run-trace",
        policy_engine=PolicyEngine(),
        state_root={"acp.state.v1": {"schema_version": "acp.state.v1", "_version": 0}},
    )
    outcome = StepOutcome.continue_with({"phase": "execute"})
    record = await HarnessKernel.execute_step(outcome, step_ctx, kernel_ctx)
    assert record.outcome_applied is True
    assert record.state_version == 1
    assert kernel_ctx.run_trace.steps
    assert step_ctx.state_snapshot["acp.state.v1"]["phase"] == "execute"


@pytest.mark.unit
@pytest.mark.gate
async def test_kernel_rejects_mixed_side_effect_mode() -> None:
    step_ctx = AgentStepContext(step_index=0, side_effect_mode=SideEffectMode.IMMEDIATE)
    kernel_ctx = StepKernelContext(
        agent_id="demo",
        side_effect_mode=SideEffectMode.IMMEDIATE,
    )
    outcome = StepOutcome.continue_with(
        {},
        diagnostics=None,
    )
    outcome = outcome.model_copy(update={"requested_actions": [{"tool_id": "demo.tool"}]})
    record = await HarnessKernel.execute_step(outcome, step_ctx, kernel_ctx)
    assert record.side_effect_mode_violation is True
    assert record.error_code == AgentRunErrorCode.VALIDATION_FAILED


@pytest.mark.unit
@pytest.mark.gate
async def test_kernel_policy_post_denies_empty_terminal_output() -> None:
    step_ctx = AgentStepContext(step_index=0)
    kernel_ctx = StepKernelContext(
        agent_id="demo",
        policy_engine=PolicyEngine(),
    )
    outcome = StepOutcome.complete("", terminal_reason=TerminalReason.GOAL_MET)
    record = await HarnessKernel.execute_step(outcome, step_ctx, kernel_ctx)
    assert record.error_code == AgentRunErrorCode.POLICY_DENIED
    assert record.policy_post is not None
    assert record.policy_post.action == PolicyAction.DENY
