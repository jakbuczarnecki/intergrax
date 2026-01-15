# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.engine.runtime import RuntimeEngine
from intergrax.runtime.nexus.policies.policy_enforcer import TransientOperationError
from intergrax.runtime.nexus.policies.runtime_policies import ExecutionKind, RuntimePolicies, RetryPolicy
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest, StopReason
from intergrax.runtime.nexus.runtime_steps.contract import RuntimeStepRunner
from tests._support.builder import build_runtime_config_deterministic, build_engine_harness


class _AlwaysFailStep:
    def execution_kind(self) -> ExecutionKind:
        return ExecutionKind.LLM

    async def run(self, state) -> None:
        raise TransientOperationError("always failing")


class _FailingPipeline:
    async def run(self, *, state) -> object:
        # Enforce policy at step level; this should abort to HITL after retry
        await RuntimeStepRunner.execute_pipeline([_AlwaysFailStep()], state)
        return object()  # unreachable


@pytest.mark.asyncio
async def test_runtime_engine_returns_needs_user_input_on_policy_abort(monkeypatch) -> None:
    cfg = build_runtime_config_deterministic()
    cfg.runtime_policies = RuntimePolicies(
        retry=RetryPolicy(max_attempts=1, backoff_seconds=0.0)
    )

    harness = build_engine_harness(cfg=cfg)
    engine: RuntimeEngine = harness.engine

    # Monkeypatch PipelineFactory to return our failing pipeline
    from intergrax.runtime.nexus.pipelines import pipeline_factory as _pf
    monkeypatch.setattr(_pf.PipelineFactory, "build_pipeline", lambda *, state: _FailingPipeline())

    request = RuntimeRequest(
        user_id="test-user",
        session_id="test-session",
        message="test",
    )

    answer = await engine.run(request)

    assert answer.stop_reason == StopReason.NEEDS_USER_INPUT
