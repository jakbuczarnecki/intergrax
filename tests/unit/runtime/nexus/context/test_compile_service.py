# © Artur Czarnecki. All rights reserved.

"""CE-3.9: ContextCompiler hot-path service."""

from __future__ import annotations

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.compile_service import compile_chat_messages, compile_prompt_text

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _SmallWindowAdapter(LLMAdapter):
    provider = "fake"
    model = "fake-small"

    def __init__(self, window: int = 1024) -> None:
        super().__init__()
        self._window = window

    @property
    def context_window_tokens(self) -> int:
        return self._window

    def generate_messages(self, messages, **kwargs) -> LLMAdapterResponse:
        _ = messages, kwargs
        return LLMAdapterResponse(content="ok")


def test_compile_prompt_text_runs_preflight() -> None:
    adapter = _SmallWindowAdapter(window=1024)
    config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
    text = compile_prompt_text("hello world", config, max_output_tokens=128)
    assert isinstance(text, str)


def test_compile_chat_messages_never_overflow_fixture() -> None:
    adapter = _SmallWindowAdapter(window=256)
    config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
    long_content = "word " * 400
    messages = [ChatMessage(role="user", content=long_content)]
    result = compile_chat_messages(messages, config, max_output_tokens=32)
    assert result.total_tokens <= result.budget_tokens
