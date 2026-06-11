# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.authoring.llm_router import StepLLMRouter
from intergrax.contracts.agent_run_trace import GatewayCallStatus


@pytest.mark.unit
@pytest.mark.gate
async def test_step_llm_router_records_call() -> None:
    router = StepLLMRouter(
        allowed_models=("balanced", "frontier"),
        default_model="balanced",
    )
    result = await router.complete("hello world", model_hint="frontier")
    assert result.model_id == "frontier"
    assert result.call_record.status == GatewayCallStatus.SUCCEEDED
    pending = router.drain_pending_calls()
    assert len(pending) == 1
    assert pending[0].tokens_in > 0


@pytest.mark.unit
@pytest.mark.gate
def test_step_llm_router_unknown_hint_falls_back_to_default() -> None:
    router = StepLLMRouter(allowed_models=("balanced",), default_model="balanced")
    assert router.resolve_model("unknown-model") == "balanced"
