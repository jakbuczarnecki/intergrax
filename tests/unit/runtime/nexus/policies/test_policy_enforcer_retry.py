# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.policies.policy_enforcer import TransientOperationError
from intergrax.runtime.nexus.policies.runtime_policies import ExecutionKind, RuntimePolicies, RetryPolicy
from intergrax.runtime.nexus.runtime_steps.contract import RuntimeStepRunner
from tests._support.builder import build_runtime_state_for_tests


class _FlakyStep:
    def __init__(self) -> None:
        self.calls = 0

    def execution_kind(self) -> ExecutionKind:
        return ExecutionKind.LLM

    async def run(self, state) -> None:
        self.calls += 1
        if self.calls < 3:
            raise TransientOperationError("temporary failure")


@pytest.mark.asyncio
async def test_runtime_step_runner_retries_transient_step() -> None:
    state = build_runtime_state_for_tests(run_id="test-run-step-retry")

    # Inject deterministic retry policy
    state.context.config.runtime_policies = RuntimePolicies(
        retry=RetryPolicy(max_attempts=3, backoff_seconds=0.0)
    )

    step = _FlakyStep()

    await RuntimeStepRunner.execute_pipeline([step], state)

    assert step.calls == 3

    policy_events = [e for e in state.trace_events if e.component.name == "POLICY"]
    assert len(policy_events) >= 3


class _FailingStep:
    def execution_kind(self) -> ExecutionKind:
        return ExecutionKind.LLM

    async def run(self, state) -> None:
        raise ValueError("logical error")


@pytest.mark.asyncio
async def test_runtime_step_runner_does_not_retry_non_retryable_error() -> None:
    state = build_runtime_state_for_tests(run_id="test-run-no-retry")

    state.context.config.runtime_policies = RuntimePolicies(
        retry=RetryPolicy(max_attempts=3, backoff_seconds=0.0)
    )

    step = _FailingStep()

    with pytest.raises(ValueError):
        await RuntimeStepRunner.execute_pipeline([step], state)
