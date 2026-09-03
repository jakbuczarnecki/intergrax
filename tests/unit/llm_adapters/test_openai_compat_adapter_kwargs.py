# © Artur Czarnecki. All rights reserved.

"""Constructor/request kwargs boundary for OpenAI Chat Completions-compatible adapters."""

from __future__ import annotations

from typing import Any, Type
from unittest.mock import MagicMock, patch

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.providers.openai_compat_providers import (
    DeepSeekChatAdapter,
    FireworksChatAdapter,
    GroqChatAdapter,
    OpenRouterChatAdapter,
    TogetherChatAdapter,
    XaiChatAdapter,
)
from intergrax.llm_adapters.registry.profile import LLMProfile

pytestmark = pytest.mark.unit

_TEST_API_KEY = "test-secret"
_TEST_MODEL = "test-model"
_TEST_BASE_URL = "https://example.test/v1"
_TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]

_COMPAT_ADAPTERS: list[tuple[LLMProvider, Type[Any], str]] = [
    (LLMProvider.GROQ, GroqChatAdapter, "GROQ_API_KEY"),
    (LLMProvider.TOGETHER, TogetherChatAdapter, "TOGETHER_API_KEY"),
    (LLMProvider.FIREWORKS, FireworksChatAdapter, "FIREWORKS_API_KEY"),
    (LLMProvider.OPENROUTER, OpenRouterChatAdapter, "OPENROUTER_API_KEY"),
    (LLMProvider.DEEPSEEK, DeepSeekChatAdapter, "DEEPSEEK_API_KEY"),
    (LLMProvider.XAI, XaiChatAdapter, "XAI_API_KEY"),
]

_CONSTRUCTOR_LEAK_KEYS = frozenset(
    {"api_key", "base_url", "client", "organization", "project", "calls_per_minute"}
)


def _mock_chat_response(*, content: str = "ok") -> MagicMock:
    usage = MagicMock(prompt_tokens=3, completion_tokens=2)
    msg = MagicMock(content=content, tool_calls=None)
    choice = MagicMock(message=msg, finish_reason="stop")
    return MagicMock(usage=usage, choices=[choice])


@pytest.mark.parametrize("provider,adapter_cls,env_key", _COMPAT_ADAPTERS)
def test_profile_passes_api_key_to_openai_client_constructor(
    provider: LLMProvider,
    adapter_cls: Type[Any],
    env_key: str,
) -> None:
    profile = LLMProfile(
        provider=provider,
        model=_TEST_MODEL,
        options={"base_url": _TEST_BASE_URL, "temperature": 0.1},
    )
    with patch("intergrax.llm_adapters.providers.openai_compat_factory.OpenAI") as openai_cls:
        client_instance = MagicMock()
        openai_cls.return_value = client_instance
        adapter = profile.create_adapter(secrets={"api_key": _TEST_API_KEY})
        openai_cls.assert_called_once_with(
            api_key=_TEST_API_KEY,
            base_url=_TEST_BASE_URL,
        )
        assert adapter.model == _TEST_MODEL


@pytest.mark.parametrize("provider,adapter_cls,env_key", _COMPAT_ADAPTERS)
def test_generate_with_tools_does_not_leak_constructor_kwargs(
    provider: LLMProvider,
    adapter_cls: Type[Any],
    env_key: str,
) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_chat_response(content="tool-ok")

    adapter = adapter_cls(
        client=client,
        model=_TEST_MODEL,
        api_key=_TEST_API_KEY,
        base_url=_TEST_BASE_URL,
        temperature=0.2,
        calls_per_minute=50,
    )
    adapter.generate_with_tools(
        [ChatMessage(role="user", content="hello")],
        _TOOLS,
        run_id="r1",
    )

    create_kwargs = client.chat.completions.create.call_args.kwargs
    assert create_kwargs["model"] == _TEST_MODEL
    assert create_kwargs["tools"] == _TOOLS
    assert create_kwargs.get("temperature") == 0.2
    assert not _CONSTRUCTOR_LEAK_KEYS.intersection(create_kwargs)


@pytest.mark.parametrize("provider,adapter_cls,env_key", _COMPAT_ADAPTERS)
def test_generate_structured_does_not_leak_constructor_kwargs(
    provider: LLMProvider,
    adapter_cls: Type[Any],
    env_key: str,
) -> None:
    from pydantic import BaseModel, ConfigDict

    client = MagicMock()
    tool_call_msg = MagicMock(content='{"value": 7}', tool_calls=None)
    choice = MagicMock(message=tool_call_msg, finish_reason="stop")
    client.chat.completions.create.return_value = MagicMock(
        usage=MagicMock(prompt_tokens=1, completion_tokens=1),
        choices=[choice],
    )

    adapter = adapter_cls(
        client=client,
        model=_TEST_MODEL,
        api_key=_TEST_API_KEY,
        base_url=_TEST_BASE_URL,
    )

    class Payload(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: int

    adapter.generate_structured(
        [ChatMessage(role="user", content="json")],
        Payload,
        run_id="r1",
    )

    create_kwargs = client.chat.completions.create.call_args.kwargs
    assert create_kwargs["model"] == _TEST_MODEL
    assert "response_format" in create_kwargs
    assert not _CONSTRUCTOR_LEAK_KEYS.intersection(create_kwargs)
