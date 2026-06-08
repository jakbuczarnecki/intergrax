# © Artur Czarnecki. All rights reserved.

"""MEM-DEPTH-1.6: long session fixture completes without overflow."""

from __future__ import annotations

import pytest

from intergrax.applications.contracts.environment_profile import ContextDecisionProfile
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_compiler import ContextCompiler
from intergrax.runtime.nexus.context.context_preflight import verify_context_preflight
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse

pytestmark = pytest.mark.gate


class _SmallWindowAdapter(LLMAdapter):
    provider = "fake"
    model = "fake-small"

    def __init__(self, window: int) -> None:
        super().__init__()
        self._window = window

    @property
    def context_window_tokens(self) -> int:
        return self._window

    def generate_messages(self, messages, **kwargs) -> LLMAdapterResponse:
        _ = messages, kwargs
        return LLMAdapterResponse(content="ok")


def test_long_session_compiles_without_overflow() -> None:
    adapter = _SmallWindowAdapter(window=2048)
    config = RuntimeConfig(
        llm_adapter=adapter,
        context_decision_profile=ContextDecisionProfile().model_dump(mode="json"),
    )
    messages: list[ChatMessage] = [ChatMessage(role="system", content="System")]
    for turn in range(200):
        messages.append(ChatMessage(role="user", content=f"user turn {turn} " + "word " * 20))
        messages.append(ChatMessage(role="assistant", content=f"assistant {turn} " + "reply " * 20))
    messages.append(ChatMessage(role="system", content="RAG CONTEXT:\n" + "evidence " * 500))
    messages.append(ChatMessage(role="user", content="final question"))

    compiler = ContextCompiler()
    result = compiler.compile(messages, config, max_output_tokens=256)
    verify_context_preflight(result.messages, adapter, max_output_tokens=256)
    assert result.degradation_steps
    assert result.total_tokens <= result.budget_tokens
