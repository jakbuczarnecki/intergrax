# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm_adapters.providers.openai_compat_providers import (
    GroqChatAdapter,
    LlamaCppChatAdapter,
    VllmChatAdapter,
)


def test_groq_adapter_mocked_chat() -> None:
    client = MagicMock()
    usage = MagicMock(prompt_tokens=5, completion_tokens=3)
    msg = MagicMock(content="ok", tool_calls=None)
    choice = MagicMock(message=msg, finish_reason="stop")
    res = MagicMock(usage=usage, choices=[choice])
    client.chat.completions.create.return_value = res

    with patch.dict("os.environ", {"GROQ_API_KEY": "test"}, clear=False):
        adapter = GroqChatAdapter(client=client, model="llama-3.3-70b-versatile")

    response = adapter.generate_messages([ChatMessage(role="user", content="hi")], run_id="g1")
    assert response.content == "ok"
    assert adapter.supports_tools() is True


def test_vllm_adapter_mocked_chat() -> None:
    client = MagicMock()
    usage = MagicMock(prompt_tokens=2, completion_tokens=1)
    msg = MagicMock(content="local", tool_calls=None)
    choice = MagicMock(message=msg, finish_reason="stop")
    res = MagicMock(usage=usage, choices=[choice])
    client.chat.completions.create.return_value = res

    adapter = VllmChatAdapter(client=client, model="meta-llama/Llama-3.1-8B-Instruct")
    response = adapter.generate_messages([ChatMessage(role="user", content="ping")], run_id="v1")
    assert response.content == "local"


def test_llama_cpp_adapter_mocked_chat() -> None:
    client = MagicMock()
    usage = MagicMock(prompt_tokens=2, completion_tokens=1)
    msg = MagicMock(content="cpu-local", tool_calls=None)
    choice = MagicMock(message=msg, finish_reason="stop")
    res = MagicMock(usage=usage, choices=[choice])
    client.chat.completions.create.return_value = res

    adapter = LlamaCppChatAdapter(client=client, model="default")
    response = adapter.generate_messages([ChatMessage(role="user", content="ping")], run_id="lc1")
    assert response.content == "cpu-local"


def test_registry_lazy_groq() -> None:
    snapshot = dict(LLMAdapterRegistry._factories)
    try:
        LLMAdapterRegistry._factories.clear()
        with patch.dict("os.environ", {"GROQ_API_KEY": "k"}, clear=False):
            adapter = LLMAdapterRegistry.create(LLMProvider.GROQ, client=MagicMock(), model="m")
        assert isinstance(adapter, GroqChatAdapter)
    finally:
        LLMAdapterRegistry._factories = snapshot
