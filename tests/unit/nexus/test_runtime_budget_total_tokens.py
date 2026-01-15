# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.budget.budget_models import RunBudget, BudgetPolicy, BudgetEnforcementMode
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest, StopReason
from tests._support.builder import build_runtime_config_deterministic, build_engine_harness


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_runtime_aborts_on_max_total_tokens_exceeded() -> None:
    cfg = build_runtime_config_deterministic(llm_text="OK")  # output_tokens=2 in FakeLLMAdapter
    cfg.run_budget = RunBudget(max_total_tokens=0)
    cfg.budget_policy = BudgetPolicy(enforcement_mode=BudgetEnforcementMode.ABORT)
    cfg.validate()

    harness = build_engine_harness(cfg=cfg)

    request = RuntimeRequest(
        user_id="test-user",
        session_id="test-session",
        message="hello",
    )

    answer = await harness.engine.run(request)

    assert answer.stop_reason == StopReason.NEEDS_USER_INPUT
    steps = [evt.step for evt in answer.trace_events]
    assert "budget_exceeded" in steps
