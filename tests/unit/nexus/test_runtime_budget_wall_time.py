# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import asyncio
import pytest

from intergrax.runtime.nexus.budget.budget_models import RunBudget, BudgetPolicy, BudgetEnforcementMode
from intergrax.runtime.nexus.pipelines.pipeline_factory import PipelineFactory
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest, StopReason, RuntimeAnswer
from tests._support.builder import build_runtime_config_deterministic, build_engine_harness


pytestmark = pytest.mark.unit


class _SlowPipeline:
    async def run(self, *, state) -> RuntimeAnswer:
        await asyncio.sleep(0.05)
        return RuntimeAnswer(
            run_id=state.run_id,
            answer="ok",
            stop_reason=StopReason.COMPLETED,
        )


@pytest.mark.asyncio
async def test_runtime_aborts_on_max_wall_time_exceeded(monkeypatch) -> None:
    cfg = build_runtime_config_deterministic(llm_text="OK")
    cfg.run_budget = RunBudget(max_wall_time_seconds=0.01)
    cfg.budget_policy = BudgetPolicy(enforcement_mode=BudgetEnforcementMode.ABORT)
    cfg.validate()

    harness = build_engine_harness(cfg=cfg)

    pipeline = _SlowPipeline()

    def _fake_build_pipeline(*, state):
        return pipeline

    monkeypatch.setattr(PipelineFactory, "build_pipeline", _fake_build_pipeline)

    request = RuntimeRequest(
        user_id="test-user",
        session_id="test-session",
        message="hello",
    )

    answer = await harness.engine.run(request)

    assert answer.stop_reason == StopReason.NEEDS_USER_INPUT
    steps = [evt.step for evt in answer.trace_events]
    assert "budget_exceeded" in steps
