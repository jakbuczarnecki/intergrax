# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.budget.budget_models import RunBudget, BudgetPolicy, BudgetEnforcementMode
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.pipelines.pipeline_factory import PipelineFactory
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest, StopReason, RuntimeAnswer
from intergrax.runtime.nexus.tracing.trace_models import ToolCallTrace
from tests._support.builder import build_runtime_config_deterministic, build_engine_harness


pytestmark = pytest.mark.unit


class _ToolOncePipeline:
    async def run(self, *, state: RuntimeState) -> RuntimeAnswer:
        state.tool_traces.append(
            ToolCallTrace(
                tool_name="fake_tool",
                arguments={"q": "hello"},
                output_preview="ok",
                success=True,
                error_message=None,
                raw_trace={"kind": "fake", "note": "deterministic test trace"},
            )
        )
        return RuntimeAnswer(
            run_id=state.run_id,
            answer="ok",
            stop_reason=StopReason.COMPLETED,
        )


@pytest.mark.asyncio
async def test_runtime_aborts_on_max_tool_calls_exceeded(monkeypatch) -> None:
    cfg = build_runtime_config_deterministic(llm_text="OK")
    cfg.run_budget = RunBudget(max_tool_calls=0)
    cfg.budget_policy = BudgetPolicy(enforcement_mode=BudgetEnforcementMode.ABORT)
    cfg.validate()

    harness = build_engine_harness(cfg=cfg)

    pipeline = _ToolOncePipeline()

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

    budget_events = [e for e in answer.trace_events if e.step == "budget_exceeded"]
    assert len(budget_events) >= 1
    payload = budget_events[-1].payload
    assert payload is not None
    assert payload.to_dict()["budget_name"] == "max_tool_calls"
