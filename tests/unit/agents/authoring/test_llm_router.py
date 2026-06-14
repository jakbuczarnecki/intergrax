# © Artur Czarnecki. All rights reserved.

import pytest
from unittest.mock import MagicMock

from intergrax.agents.authoring.llm_router import StepLLMRouter
from intergrax.contracts.agent_run_trace import GatewayCallStatus
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider


@pytest.mark.unit
@pytest.mark.gate
async def test_step_llm_router_uses_llm_adapter_port() -> None:
    from intergrax.agents.authoring.llm_router import StepLLMRouter
    from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
    from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage

    adapter = MagicMock()
    adapter.provider = LLMProvider.OPENAI
    adapter.generate_messages.return_value = LLMAdapterResponse(
        content="adapter-response",
        usage=LLMTokenUsage(input_tokens=4, output_tokens=6),
        model="gpt-4o",
        provider="openai",
    )
    router = StepLLMRouter(
        allowed_models=("gpt-4o",),
        default_model="gpt-4o",
        llm_adapter=adapter,
    )
    result = await router.complete("hello")
    assert result.text == "adapter-response"
    assert result.tokens_in == 4
    assert result.tokens_out == 6


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
