# © Artur Czarnecki. All rights reserved.

"""REL-MAINT-01 depth tests — compensation, partial results, retry."""

from __future__ import annotations

import pytest

from intergrax.contracts.resilience_policy import ResiliencePolicy
from intergrax.runtime.reliability.compensation import CompensationFlow, CompensationStep

pytestmark = pytest.mark.gate


@pytest.mark.asyncio
async def test_compensation_flow_runs_handler_on_failure() -> None:
    calls: list[str] = []

    async def undo(_step_id: str, _ctx: dict) -> None:
        calls.append("undo")

    flow = CompensationFlow(
        steps=[CompensationStep(step_id="s1", handler_id="undo")],
        handlers={"undo": undo},
    )
    executed = await flow.run("s1", {})
    assert executed == ["undo"]


def test_resilience_policy_partial_result_opt_in() -> None:
    strict = ResiliencePolicy(allow_partial_result=False)
    lenient = ResiliencePolicy(allow_partial_result=True)
    assert strict.allow_partial_result is False
    assert lenient.allow_partial_result is True
