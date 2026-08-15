# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.authoring.step_loop import AgentRuntime
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.runtime.kernel.step_kernel import StepKernelContext
from intergrax.runtime.policy.policy_engine import PolicyEngine


class _ThreeStepAgent:
    async def on_next_step(self, step_ctx: AgentStepContext) -> StepOutcome:
        if step_ctx.step_index >= 2:
            return StepOutcome.complete(
                output={"steps_completed": step_ctx.step_index + 1},
            )
        return StepOutcome.continue_with(
            state_delta={"phase": f"step-{step_ctx.step_index + 1}"},
        )


@pytest.mark.unit
@pytest.mark.gate
async def test_three_step_session_loop() -> None:
    agent = _ThreeStepAgent()
    run_id = mint_run_id()
    task_id = mint_task_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_id)
    try:
        step_ctx = AgentStepContext(
            step_index=0,
            run_id=run_id,
            task_id=task_id,
            agent_id="counter",
        )
        kernel_ctx = StepKernelContext(
            agent_id="counter",
            run_id=run_id,
            task_id=task_id,
            policy_engine=PolicyEngine(),
            state_root={"acp.state.v1": {"schema_version": "acp.state.v1", "_version": 0}},
        )

        last_outcome = StepOutcome.continue_with({})
        for _ in range(5):
            outcome, record = await AgentRuntime.advance_step(agent, step_ctx, kernel_ctx)
            last_outcome = outcome
            assert record.outcome_applied or record.error_code is not None
            if outcome.is_terminal:
                break
            step_ctx = step_ctx.model_copy(update={"step_index": step_ctx.step_index + 1})

        assert last_outcome.is_terminal is True
        assert last_outcome.output == {"steps_completed": 3}
        assert kernel_ctx.run_trace.steps
        blob = kernel_ctx.state_root["acp.state.v1"]
        assert blob.get("phase") == "step-2"
    finally:
        reset_active_execution_identity(token)
