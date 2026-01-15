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
async def test_runtime_aborts_on_max_llm_calls_exceeded() -> None:
    # Build deterministic config
    cfg = build_runtime_config_deterministic(llm_text="OK")

    # Enable budget: allow 0 calls, but pipeline will do 1
    cfg.run_budget = RunBudget(max_llm_calls=0)
    cfg.budget_policy = BudgetPolicy(enforcement_mode=BudgetEnforcementMode.ABORT)

    # Build engine using standard harness
    harness = build_engine_harness(cfg=cfg)
    engine = harness.engine

    request = RuntimeRequest(
        user_id="test-user",
        session_id="test-session",
        message="hello",
    )

    answer = await engine.run(request)

    # Budget exceeded should trigger policy abort => NEEDS_USER_INPUT
    assert answer.stop_reason == StopReason.NEEDS_USER_INPUT

    # Trace must contain budget_exceeded
    steps = [evt.step for evt in answer.trace_events]
    assert "budget_exceeded" in steps
    assert "hitl_escalation" in steps
