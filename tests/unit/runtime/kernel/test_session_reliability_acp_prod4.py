# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.applications._shared.reliability_runtime_bridge import ReliabilityWiringOptions
from intergrax.contracts.agent_run import AgentRunError
from intergrax.contracts.agent_run_enums import AgentRunErrorCode
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.resilience_policy import default_resilience_policy
from intergrax.runtime.kernel.session_reliability import AgentSessionReliability
from intergrax.runtime.kernel.step_kernel import HarnessKernel, StepKernelContext
from intergrax.runtime.policy.policy_engine import PolicyEngine


def _reliability(*, threshold: int = 2, interval: int = 2) -> AgentSessionReliability:
    return AgentSessionReliability.from_wiring_options(
        ReliabilityWiringOptions(
            idempotency_enabled=True,
            circuit_breaker_failure_threshold=threshold,
            checkpoint_interval_steps=interval,
            long_running_scheduler_enabled=False,
            resilience_policy=default_resilience_policy(),
            default_autonomy_level="ask",
            tenant_autonomy_ceiling=None,
        )
    )


@pytest.mark.unit
@pytest.mark.gate
def test_reliability_checkpoint_interval() -> None:
    reliability = _reliability(interval=2)
    assert reliability.should_checkpoint(0) is False
    assert reliability.should_checkpoint(1) is True
    assert reliability.should_checkpoint(3) is True


@pytest.mark.unit
@pytest.mark.gate
async def test_kernel_opens_circuit_after_retriable_failures() -> None:
    reliability = _reliability(threshold=2)
    step_ctx = AgentStepContext(step_index=0)
    kernel_ctx = StepKernelContext(
        agent_id="demo",
        run_id="run-rel",
        policy_engine=PolicyEngine(),
        reliability=reliability,
    )
    fail_outcome = StepOutcome.fail(
        [AgentRunError(code=AgentRunErrorCode.TOOL_FAILED, message="transient")],
        is_terminal=False,
    )
    await HarnessKernel.execute_step(fail_outcome, step_ctx, kernel_ctx)
    await HarnessKernel.execute_step(fail_outcome, step_ctx, kernel_ctx)
    assert reliability.circuit_open is True

    ok_outcome = StepOutcome.continue_with({"phase": "retry"})
    blocked = await HarnessKernel.execute_step(ok_outcome, step_ctx, kernel_ctx)
    assert blocked.error_code == AgentRunErrorCode.INTERNAL_ERROR
    assert blocked.step_record is not None
    assert blocked.step_record.diagnostics.get("circuit_breaker") == "open"
