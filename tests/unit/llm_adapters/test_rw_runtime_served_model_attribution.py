# © Artur Czarnecki. All rights reserved.

"""T-IGX-051-D NT-RW-RUN-05 (NT-ROW-107, PDC-D3-R005): served-model attribution.

Before the change the adapter response carried only the requested model and call accounting recorded the
requested model, also when the provider reported that another model served the call (an alias resolved to a
dated version, a gateway routing). The response now carries ``served_model`` when the SDK reports it, and the
accounting of the call records it; an SDK that reports nothing leaves ``served_model`` empty and the requested
model in the accounting. Fake SDK payloads only (no network, no credential).
"""

from __future__ import annotations

import dataclasses
import io
import json
import sys
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEventKind

pytestmark = pytest.mark.unit

_USER = [ChatMessage(role="user", content="hello")]
_SERVED = "served-model-2026-01-01"


@pytest.fixture(autouse=True)
def _isolate_provider_modules() -> Iterator[None]:
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if name.startswith("intergrax.llm_adapters.providers.") and name.endswith("_adapter"):
            sys.modules.pop(name, None)


def _spy_accounting(adapter: Any) -> list[str]:
    recorded: list[str] = []
    original = adapter.usage.end_call

    def _end_call(call: Any, **kwargs: Any) -> None:
        recorded.append(str(call.model))
        original(call, **kwargs)

    adapter.usage.end_call = _end_call
    return recorded


def _completion(model: str | None) -> Any:
    message = SimpleNamespace(content="x", tool_calls=None, refusal=None)
    return SimpleNamespace(
        id="r1",
        model=model,
        usage=None,
        system_fingerprint=None,
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
    )


def _openai_chat() -> tuple[Any, str]:
    from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
    from intergrax.llm_adapters.providers.openai_chat_completions_adapter import OpenAIChatCompletionsAdapter

    client = MagicMock()
    client.chat.completions.create.return_value = _completion(_SERVED)
    return OpenAIChatCompletionsAdapter(client=client, model="gpt-4o", provider=LLMProvider.OPENAI), "gpt-4o"


def _openai_responses() -> tuple[Any, str]:
    from intergrax.llm_adapters.providers.openai_responses_adapter import OpenAIChatResponsesAdapter

    response = MagicMock()
    response.id = "resp_1"
    response.status = "completed"
    response.incomplete_details = None
    response.output = []
    response.output_text = "x"
    response.model = _SERVED
    response.usage = MagicMock(input_tokens=1, output_tokens=1)
    client = MagicMock()
    client.responses.create.return_value = response
    return OpenAIChatResponsesAdapter(client=client, model="gpt-5.1"), "gpt-5.1"


def _claude() -> tuple[Any, str]:
    pytest.importorskip("anthropic", reason="optional extra llm-anthropic")
    from intergrax.llm_adapters.providers.claude_adapter import ClaudeChatAdapter

    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        id="m1",
        model=_SERVED,
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="x")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    return ClaudeChatAdapter(client=client, model="claude-3-5-sonnet-latest"), "claude-3-5-sonnet-latest"


def _mistral() -> tuple[Any, str]:
    pytest.importorskip("mistralai", reason="optional extra llm-mistral")
    from intergrax.llm_adapters.providers.mistral_adapter import MistralChatAdapter

    client = MagicMock()
    client.chat.complete.return_value = _completion(_SERVED)
    return MistralChatAdapter(client=client, model="mistral-large-latest"), "mistral-large-latest"


def _gemini() -> tuple[Any, str]:
    pytest.importorskip("google.genai", reason="optional extra llm-gemini")
    from intergrax.llm_adapters.providers.gemini_adapter import GeminiChatAdapter

    client = MagicMock()
    client.chats.create.return_value.send_message.return_value = SimpleNamespace(
        text="x",
        model_version=_SERVED,
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"), content=None)],
        prompt_feedback=None,
    )
    return GeminiChatAdapter(client=client, model="gemini-2.5-flash"), "gemini-2.5-flash"


def _native_ollama() -> tuple[Any, str]:
    from intergrax.llm_adapters.providers.native_ollama_adapter import NativeOllamaAdapter
    from intergrax.llm_adapters.providers.ollama_capabilities import OllamaModelCapabilityResolver

    client = MagicMock()
    client.chat.return_value = {"model": _SERVED, "message": {"content": "x"}, "done": True, "done_reason": "stop"}
    resolver = OllamaModelCapabilityResolver(show_model=lambda _m: SimpleNamespace(capabilities=["completion"]))
    return NativeOllamaAdapter(client=client, model="qwen2.5", capability_resolver=resolver), "qwen2.5"


def _langchain_ollama() -> tuple[Any, str]:
    pytest.importorskip("langchain_core", reason="optional extra llm-langchain-ollama")
    from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter
    from intergrax.llm_adapters.providers.ollama_capabilities import OllamaModelCapabilityResolver

    chat = MagicMock()
    chat.model = "qwen2.5"
    chat.invoke.return_value = SimpleNamespace(content="x", response_metadata={"model": _SERVED, "done_reason": "stop"})
    resolver = OllamaModelCapabilityResolver(show_model=lambda _m: SimpleNamespace(capabilities=["completion"]))
    return LangChainOllamaAdapter(chat=chat, model="qwen2.5", capability_resolver=resolver), "qwen2.5"


def _bedrock_invoke_anthropic() -> tuple[Any, str]:
    from intergrax.llm_adapters.providers.aws_bedrock_adapter import BedrockChatAdapter

    body = {"model": _SERVED, "content": [{"type": "text", "text": "x"}], "stop_reason": "end_turn"}
    client = SimpleNamespace(
        invoke_model=MagicMock(return_value={"body": io.BytesIO(json.dumps(body).encode("utf-8"))}),
        invoke_model_with_response_stream=MagicMock(),
    )
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    adapter = BedrockChatAdapter(client=client, model_id=model_id, region="us-east-1", use_converse=False)
    return adapter, model_id


_REPORTING: dict[str, Callable[[], tuple[Any, str]]] = {
    "openai-chat": _openai_chat,
    "openai-responses": _openai_responses,
    "claude": _claude,
    "mistral": _mistral,
    "gemini": _gemini,
    "native-ollama": _native_ollama,
    "langchain-ollama": _langchain_ollama,
    "bedrock-invoke-anthropic": _bedrock_invoke_anthropic,
}


def test_the_response_contract_has_an_additive_served_model_field() -> None:
    fields = {f.name: f for f in dataclasses.fields(LLMAdapterResponse)}
    assert "served_model" in fields, "LLMAdapterResponse.served_model is missing"
    assert fields["served_model"].default is None
    assert LLMAdapterResponse(content="x").served_model is None


@pytest.mark.parametrize("case", sorted(_REPORTING))
def test_the_reported_model_is_on_the_response_and_in_the_accounting(case: str) -> None:
    adapter, requested = _REPORTING[case]()
    recorded = _spy_accounting(adapter)
    response = adapter.generate_messages(_USER)
    assert response.model == requested, "``model`` keeps its meaning: the requested model"
    assert getattr(response, "served_model", None) == _SERVED
    assert recorded == [_SERVED]


def _cohere() -> tuple[Any, str]:
    pytest.importorskip("cohere", reason="optional extra llm-cohere-native")
    from intergrax.llm_adapters.providers.cohere_native_adapter import CohereNativeChatAdapter

    client = MagicMock()
    client.chat.return_value = SimpleNamespace(
        message=SimpleNamespace(content=[SimpleNamespace(text="x")]), finish_reason="COMPLETE"
    )
    return CohereNativeChatAdapter(client=client, model="command-r-plus"), "command-r-plus"


def _bedrock_converse() -> tuple[Any, str]:
    from intergrax.llm_adapters.providers.aws_bedrock_adapter import BedrockChatAdapter

    resp = {"output": {"message": {"content": [{"text": "x"}]}}, "stopReason": "end_turn"}
    client = SimpleNamespace(converse=MagicMock(return_value=resp), converse_stream=MagicMock())
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    adapter = BedrockChatAdapter(client=client, model_id=model_id, region="us-east-1", use_converse=True)
    return adapter, model_id


def _openai_chat_without_model() -> tuple[Any, str]:
    from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
    from intergrax.llm_adapters.providers.openai_chat_completions_adapter import OpenAIChatCompletionsAdapter

    client = MagicMock()
    client.chat.completions.create.return_value = _completion(None)
    return OpenAIChatCompletionsAdapter(client=client, model="gpt-4o", provider=LLMProvider.OPENAI), "gpt-4o"


_SILENT: dict[str, Callable[[], tuple[Any, str]]] = {
    "cohere": _cohere,
    "bedrock-converse": _bedrock_converse,
    "openai-chat-without-model": _openai_chat_without_model,
}


@pytest.mark.parametrize("case", sorted(_SILENT))
def test_an_sdk_that_reports_no_model_leaves_the_requested_model_in_place(case: str) -> None:
    adapter, requested = _SILENT[case]()
    recorded = _spy_accounting(adapter)
    response = adapter.generate_messages(_USER)
    assert dataclasses.is_dataclass(response) and "served_model" in {f.name for f in dataclasses.fields(response)}
    assert response.served_model is None
    assert recorded == [requested]


def _final(events: Any) -> LLMAdapterResponse:
    finals = [e for e in events if e.kind is LLMStreamEventKind.FINAL]
    assert len(finals) == 1
    return finals[0].response


def test_a_stream_reports_the_served_model_of_its_chunks() -> None:
    from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
    from intergrax.llm_adapters.providers.openai_chat_completions_adapter import OpenAIChatCompletionsAdapter

    delta = SimpleNamespace(content="x", refusal=None, tool_calls=None)
    end = SimpleNamespace(content=None, refusal=None, tool_calls=None)
    chunks = [
        SimpleNamespace(model=_SERVED, choices=[SimpleNamespace(delta=delta, finish_reason=None)]),
        SimpleNamespace(model=_SERVED, choices=[SimpleNamespace(delta=end, finish_reason="stop")]),
    ]
    client = MagicMock()
    client.chat.completions.create.return_value = iter(chunks)
    adapter = OpenAIChatCompletionsAdapter(client=client, model="gpt-4o", provider=LLMProvider.OPENAI)
    recorded = _spy_accounting(adapter)
    response = _final(list(adapter.stream_messages(_USER)))
    assert getattr(response, "served_model", None) == _SERVED
    assert recorded == [_SERVED]


def test_a_native_ollama_stream_reports_the_served_model_of_its_last_chunk() -> None:
    from intergrax.llm_adapters.providers.native_ollama_adapter import NativeOllamaAdapter
    from intergrax.llm_adapters.providers.ollama_capabilities import OllamaModelCapabilityResolver

    client = MagicMock()
    client.chat.return_value = iter(
        [
            {"model": _SERVED, "message": {"content": "x"}},
            {"model": _SERVED, "message": {"content": ""}, "done": True, "done_reason": "stop"},
        ]
    )
    resolver = OllamaModelCapabilityResolver(show_model=lambda _m: SimpleNamespace(capabilities=["completion"]))
    adapter = NativeOllamaAdapter(client=client, model="qwen2.5", capability_resolver=resolver)
    recorded = _spy_accounting(adapter)
    response = _final(list(adapter.stream_messages(_USER)))
    assert getattr(response, "served_model", None) == _SERVED
    assert recorded == [_SERVED]


def test_a_failed_call_keeps_the_requested_model_in_the_accounting() -> None:
    from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
    from intergrax.llm_adapters.providers.openai_chat_completions_adapter import OpenAIChatCompletionsAdapter

    client = MagicMock()
    client.chat.completions.create.side_effect = ValueError("local failure")
    adapter = OpenAIChatCompletionsAdapter(client=client, model="gpt-4o", provider=LLMProvider.OPENAI)
    recorded = _spy_accounting(adapter)
    with pytest.raises(ValueError):
        adapter.generate_messages(_USER)
    assert recorded == ["gpt-4o"]
