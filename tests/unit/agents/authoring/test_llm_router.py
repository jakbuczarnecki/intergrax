# © Artur Czarnecki. All rights reserved.

import pytest
from unittest.mock import MagicMock, patch

from intergrax.agents.authoring.llm_router import StepLLMRouter
from intergrax.contracts.agent_run_trace import GatewayCallStatus
from intergrax.llm.messages import (
    ChatMessage,
    compute_model_facing_messages_hash,
    replace_final_user_message,
    StructuredModelInputRequiredError,
)
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


def _router_model_input_messages() -> tuple[ChatMessage, ...]:
    return tuple(
        [
            ChatMessage(role="system", content="[context:task_message:t1] objective"),
            ChatMessage(role="user", content="history user", entry_id="hist-user"),
            ChatMessage(
                role="assistant",
                content="assistant reply",
                entry_id="hist-assistant",
                tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "search", "arguments": "{}"}}],
            ),
            ChatMessage(role="tool", content="tool result", entry_id="hist-tool", tool_call_id="call-1"),
            ChatMessage(role="user", content="final user", entry_id="final-user"),
        ]
    )


@pytest.mark.unit
@pytest.mark.gate
async def test_step_llm_router_exact_send_with_model_envelope() -> None:
    from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
    from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage

    captured_messages: list[list[ChatMessage]] = []
    adapter = MagicMock()
    adapter.provider = LLMProvider.OPENAI

    def _generate_messages(messages, **kwargs):
        _ = kwargs
        captured_messages.append(list(messages))
        return LLMAdapterResponse(
            content="adapter-response",
            usage=LLMTokenUsage(input_tokens=4, output_tokens=6),
            model="gpt-4o",
            provider="openai",
        )

    adapter.generate_messages.side_effect = _generate_messages
    model_messages = _router_model_input_messages()
    with patch(
        "intergrax.runtime.nexus.context.compile_service.compile_prompt_text",
        side_effect=AssertionError("compile_prompt_text must not run on envelope path"),
    ):
        router = StepLLMRouter(
            allowed_models=("gpt-4o",),
            default_model="gpt-4o",
            llm_adapter=adapter,
            model_input_messages=model_messages,
        )
        result = await router.complete("agent-specific prompt")
    assert len(captured_messages) == 1
    sent = captured_messages[0]
    assert sent[0].content == model_messages[0].content
    assert sent[1].entry_id == "hist-user"
    assert sent[2].tool_calls == model_messages[2].tool_calls
    assert sent[3].tool_call_id == "call-1"
    assert sent[-1].content == "agent-specific prompt"
    assert sent[-1].entry_id == "final-user"
    expected_hash = compute_model_facing_messages_hash(
        replace_final_user_message(model_messages, "agent-specific prompt")
    )
    assert compute_model_facing_messages_hash(tuple(sent)) == expected_hash
    assert result.text == "adapter-response"
    assert adapter.generate_messages.call_count == 1


@pytest.mark.unit
@pytest.mark.gate
async def test_step_llm_router_text_only_port_rejects_structured_envelope() -> None:
    class _TextOnlyPort:
        async def complete(self, prompt: str, *, model_id: str, provider: str) -> tuple[str, int, int]:
            _ = prompt, model_id, provider
            raise AssertionError("text-only port must not be called")

    router = StepLLMRouter(
        allowed_models=("balanced",),
        default_model="balanced",
        llm_port=_TextOnlyPort(),
        model_input_messages=_router_model_input_messages(),
    )
    with pytest.raises(StructuredModelInputRequiredError):
        await router.complete("agent-specific prompt")
