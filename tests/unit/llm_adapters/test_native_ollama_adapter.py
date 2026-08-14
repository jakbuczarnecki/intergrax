# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, ValidationError

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.finish_reason import LLMFinishReason
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEventKind
from intergrax.llm_adapters.providers.native_ollama_adapter import (
    NativeOllamaAdapter,
)
from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter
from intergrax.llm_adapters.providers.ollama_capabilities import (
    OllamaModelCapabilityResolver,
)
from intergrax.runtime.config.forbidden_generation_model_env import (
    FORBIDDEN_GENERATION_MODEL_ENV_NAMES,
)

pytestmark = pytest.mark.unit


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Lookup a value",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
    }
]


class StructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    count: int


def _resolver(capabilities: list[str]) -> OllamaModelCapabilityResolver:
    return OllamaModelCapabilityResolver(
        show_model=lambda _model: SimpleNamespace(capabilities=capabilities),
    )


def _native_response(
    content: object = "",
    *,
    tool_calls: Sequence[object] | None = None,
    prompt_eval_count: object = None,
    eval_count: object = None,
) -> SimpleNamespace:
    values: dict[str, object] = {
        "message": SimpleNamespace(content=content, tool_calls=tool_calls or []),
    }
    if prompt_eval_count is not None:
        values["prompt_eval_count"] = prompt_eval_count
    if eval_count is not None:
        values["eval_count"] = eval_count
    return SimpleNamespace(**values)


class FakeNativeClient:
    def __init__(
        self,
        *,
        response: object | None = None,
        stream: object | None = None,
    ) -> None:
        self.response = response or _native_response("ok")
        self.stream_response = stream
        self.calls: list[dict[str, object]] = []

    def chat(
        self,
        model: str = "",
        messages: Sequence[Mapping[str, object]] | None = None,
        *,
        tools: Sequence[Mapping[str, object]] | None = None,
        stream: bool = False,
        format: str | Mapping[str, object] | None = None,
        options: Mapping[str, object] | None = None,
    ) -> object:
        request: dict[str, object] = {
            "model": model,
            "messages": list(messages or ()),
            "stream": stream,
            "options": dict(options or {}),
        }
        if tools is not None:
            request["tools"] = tools
        if format is not None:
            request["format"] = format
        self.calls.append(request)
        if stream:
            if isinstance(self.stream_response, BaseException):
                raise self.stream_response
            if callable(self.stream_response):
                return cast(Callable[[], object], self.stream_response)()
            return iter(cast(Sequence[object], self.stream_response or ()))
        return self.response


def _adapter(
    client: FakeNativeClient,
    *,
    capabilities: list[str] | None = None,
    **defaults: Any,
) -> NativeOllamaAdapter:
    return NativeOllamaAdapter(
        client=client,
        model="qwen2.5:14b",
        capability_resolver=_resolver(capabilities or ["tools", "completion"]),
        **defaults,
    )


def test_constructor_ignores_legacy_env_and_uses_code_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeNativeClient()
    legacy_env = next(
        name for name in FORBIDDEN_GENERATION_MODEL_ENV_NAMES if name.endswith("_OLLAMA_MODEL")
    )
    monkeypatch.setenv(legacy_env, "legacy-model")

    adapter = NativeOllamaAdapter(
        client=client,
        capability_resolver=_resolver([]),
        context_window_tokens=1234,
    )

    assert adapter.model == NativeOllamaAdapter.DEFAULT_MODEL
    assert adapter.provider is LLMProvider.OLLAMA
    assert adapter.context_window_tokens == 1234
    assert client.calls == []


def test_message_mapping_options_plain_response_and_provider_usage() -> None:
    client = FakeNativeClient(
        response=_native_response(
            ["ignored", {"type": "text", "text": "answer"}],
            prompt_eval_count=7,
            eval_count=3,
        )
    )
    adapter = _adapter(
        client,
        options={"top_k": 4, "temperature": 0.2},
    )
    messages = [
        ChatMessage(role="system", content="rules"),
        ChatMessage(role="user", content="question"),
        ChatMessage(role="assistant", content=""),
        ChatMessage(role="tool", content="result", tool_call_id="call-1"),
            ChatMessage(role="custom", content="legacy"),  # type: ignore[arg-type]
    ]

    response = adapter.generate_messages(
        messages,
        temperature=0.7,
        max_tokens=64,
        run_id="plain",
    )

    request = client.calls[0]
    assert request["stream"] is False
    assert request["messages"] == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": ""},
        {"role": "tool", "content": "result", "tool_call_id": "call-1"},
        {"role": "system", "content": "[CUSTOM]\nlegacy"},
    ]
    assert request["options"] == {"top_k": 4, "temperature": 0.7, "num_predict": 64}
    assert response.content == "ignoredanswer"
    assert response.model == "qwen2.5:14b"
    assert response.provider == "ollama"
    assert response.usage is not None
    assert response.usage.input_tokens == 7
    assert response.usage.output_tokens == 3
    assert response.provider_extensions is not None
    assert response.provider_extensions.usage_source == "sdk"
    assert adapter.usage.get_run_stats("plain").calls == 1


def test_tool_message_without_id_fails_before_provider_call() -> None:
    client = FakeNativeClient()
    adapter = _adapter(client)

    with pytest.raises(ValueError, match="tool message requires tool_call_id"):
        adapter.generate_messages(
            [ChatMessage(role="tool", content="result")],
        )

    assert client.calls == []


@pytest.mark.parametrize(
    "tool_call",
    [
        {"name": "lookup", "args": {"query": "x"}, "id": "call-1"},
        {
            "function": {
                "name": "lookup",
                "arguments": '{"query":"x"}',
            },
            "id": "call-2",
        },
    ],
)
def test_assistant_tool_calls_are_native_and_canonical(tool_call: dict[str, object]) -> None:
    client = FakeNativeClient()
    adapter = _adapter(client)

    adapter.generate_messages(
        [ChatMessage(role="assistant", content="", tool_calls=[tool_call])],
    )

    native_call = client.calls[0]["messages"][0]["tool_calls"][0]  # type: ignore[index]
    assert native_call == {
        "id": tool_call["id"],
        "function": {
            "name": "lookup",
            "arguments": {"query": "x"},
        },
    }


@pytest.mark.parametrize(
    "tool_call",
    [
        {"name": "", "args": {}},
        {"name": "lookup", "args": []},
        {"name": "lookup", "args": "not-json"},
        {
            "function": {
                "name": "lookup",
                "arguments": "[1, 2]",
            }
        },
    ],
)
def test_invalid_assistant_tool_calls_fail_closed(
    tool_call: dict[str, object],
) -> None:
    client = FakeNativeClient()
    adapter = _adapter(client)

    with pytest.raises(ValueError):
        adapter.generate_messages(
            [ChatMessage(role="assistant", content="", tool_calls=[tool_call])],
        )
    assert client.calls == []


def test_stream_success_emits_partials_then_one_final() -> None:
    client = FakeNativeClient(
        stream=[
            _native_response("hel"),
            _native_response("lo", prompt_eval_count=2, eval_count=2),
        ]
    )
    adapter = _adapter(client)

    events = list(adapter.stream_messages([ChatMessage(role="user", content="hi")]))

    assert [event.kind for event in events] == [
        LLMStreamEventKind.PARTIAL,
        LLMStreamEventKind.PARTIAL,
        LLMStreamEventKind.FINAL,
    ]
    assert "".join(event.delta_content for event in events[:-1]) == "hello"
    assert events[-1].response is not None
    assert events[-1].response.content == "hello"
    assert client.calls[0]["stream"] is True
    assert adapter.usage.get_run_stats().calls == 1


def test_stream_failure_before_partial_uses_one_non_stream_fallback() -> None:
    client = FakeNativeClient(
        response=_native_response("fallback"),
        stream=RuntimeError("disconnect before token"),
    )
    adapter = _adapter(client)

    events = list(
        adapter.stream_messages(
            [ChatMessage(role="user", content="hi")],
            run_id="pre-partial",
        )
    )

    assert [event.kind for event in events] == [
        LLMStreamEventKind.PARTIAL,
        LLMStreamEventKind.FINAL,
    ]
    assert events[0].delta_content == "fallback"
    assert [call["stream"] for call in client.calls] == [True, False]
    assert adapter.usage.get_run_stats("pre-partial").calls == 1


def test_stream_failure_after_partial_propagates_without_fallback() -> None:
    def stream_then_fail():
        yield _native_response("first")
        raise RuntimeError("disconnect after token")

    client = FakeNativeClient(stream=stream_then_fail)
    adapter = _adapter(client)

    events = iter(adapter.stream_messages(
        [ChatMessage(role="user", content="hi")],
        run_id="post-partial",
    ))
    assert next(events).delta_content == "first"
    with pytest.raises(RuntimeError, match="disconnect after token"):
        next(events)
    assert [call["stream"] for call in client.calls] == [True]
    stats = adapter.usage.get_run_stats("post-partial")
    assert stats.calls == 1
    assert stats.errors == 1


def test_tools_validate_choice_preserve_schema_and_parse_calls() -> None:
    tool_calls = [
        SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="lookup", arguments={"query": "żółć"}),
        ),
        SimpleNamespace(
            id="call-2",
            function=SimpleNamespace(name="second", arguments='{"n":2}'),
        ),
    ]
    client = FakeNativeClient(
        response=_native_response(
            "",
            tool_calls=tool_calls,
            prompt_eval_count=5,
            eval_count=6,
        )
    )
    adapter = _adapter(client)

    response = adapter.generate_with_tools(
        [ChatMessage(role="user", content="search")],
        TOOLS_SCHEMA,
        tool_choice="required",
        run_id="tools",
    )

    request = client.calls[0]
    assert request["tools"] is TOOLS_SCHEMA
    assert "tool_choice" not in request
    assert response.finish_reason == LLMFinishReason.TOOL_CALLS
    assert [call.id for call in response.tool_calls] == ["call-1", "call-2"]
    assert json.loads(response.tool_calls[0].arguments_json) == {"query": "żółć"}
    assert response.usage is not None
    assert response.usage.input_tokens == 5
    assert response.provider_extensions is not None
    assert response.provider_extensions.usage_source == "sdk"
    assert adapter.usage.get_run_stats("tools").calls == 1


def test_tool_streaming_remains_unsupported() -> None:
    adapter = _adapter(FakeNativeClient())

    with pytest.raises(NotImplementedError):
        adapter.stream_with_tools(
            [ChatMessage(role="user", content="search")],
            TOOLS_SCHEMA,
        )


@pytest.mark.parametrize("tool_choice", ["none", "lookup", {"name": "lookup"}])
def test_invalid_tool_choice_fails_before_native_call(tool_choice: object) -> None:
    client = FakeNativeClient()
    adapter = _adapter(client)

    with pytest.raises(ValueError, match="only tool_choice"):
        adapter.generate_with_tools(
            [ChatMessage(role="user", content="search")],
            TOOLS_SCHEMA,
            tool_choice=tool_choice,  # type: ignore[arg-type]
        )
    assert client.calls == []


def test_structured_output_uses_format_and_original_model_validation() -> None:
    client = FakeNativeClient(
        response=_native_response(
            '{"status":"ok","count":2}',
            prompt_eval_count=4,
            eval_count=5,
        )
    )
    adapter = _adapter(client)

    result = adapter.generate_structured(
        [ChatMessage(role="user", content="return json")],
        StructuredOutput,
        run_id="structured",
    )

    request = client.calls[0]
    assert request["stream"] is False
    assert isinstance(request["format"], dict)
    assert request["format"]["type"] == "object"  # type: ignore[index]
    assert isinstance(result.parsed, StructuredOutput)
    assert result.response.content == '{"status":"ok","count":2}'
    assert result.response.usage is not None
    assert result.response.usage.input_tokens == 4
    assert adapter.usage.get_run_stats("structured").calls == 1


def test_structured_output_rejects_valid_json_that_fails_original_model() -> None:
    client = FakeNativeClient(response=_native_response('{"status":"ok"}'))
    adapter = _adapter(client)

    with pytest.raises((ValidationError, ValueError)):
        adapter.generate_structured(
            [ChatMessage(role="user", content="return json")],
            StructuredOutput,
            run_id="structured-invalid",
        )
    stats = adapter.usage.get_run_stats("structured-invalid")
    assert stats.calls == 1
    assert stats.errors == 1


def test_usage_falls_back_as_a_whole_when_one_counter_is_invalid() -> None:
    client = FakeNativeClient(
        response=_native_response(
            "answer",
            prompt_eval_count=8,
            eval_count=-1,
        )
    )
    adapter = _adapter(client)

    response = adapter.generate_messages(
        [ChatMessage(role="user", content="hi")],
    )

    assert response.usage is not None
    assert response.usage.input_tokens != 8
    assert response.provider_extensions is not None
    assert response.provider_extensions.usage_source == "estimate"


def test_capability_cache_refresh_and_flags() -> None:
    calls = 0

    def show(_model: str) -> object:
        nonlocal calls
        calls += 1
        return SimpleNamespace(capabilities=["tools"])

    resolver = OllamaModelCapabilityResolver(show_model=show)
    adapter = NativeOllamaAdapter(
        client=FakeNativeClient(),
        model="qwen2.5:14b",
        capability_resolver=resolver,
    )

    assert adapter.supports_tools() is True
    assert adapter.supports_tools() is True
    assert adapter.supports_streaming() is True
    assert adapter.supports_structured_output() is True
    assert calls == 1
    adapter.refresh_model_capabilities()
    assert calls == 2


def _normalize_langchain_messages(
    messages: Sequence[object],
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            role = "tool"
        normalized.append({"role": role, "content": getattr(message, "content", "")})
    return normalized


def test_side_by_side_plain_request_harness_normalizes_both_adapters() -> None:
    native_client = FakeNativeClient(response=_native_response("native"))
    native = _adapter(native_client, options={"top_k": 4})

    langchain_chat = FakeLangChainChat()
    langchain_chat.invoke.return_value = SimpleNamespace(content="langchain")
    langchain = LangChainOllamaAdapter(
        chat=langchain_chat,  # type: ignore[arg-type]
        model="qwen2.5:14b",
        capability_resolver=_resolver([]),
        options={"top_k": 4},
    )
    messages = [
        ChatMessage(role="system", content="rules"),
        ChatMessage(role="user", content="question"),
    ]

    native_response = native.generate_messages(messages, temperature=0.2, max_tokens=32)
    langchain_response = langchain.generate_messages(
        messages,
        temperature=0.2,
        max_tokens=32,
    )

    assert native_client.calls[0]["messages"] == _normalize_langchain_messages(
        langchain_chat.invoke.call_args.args[0]
    )
    assert native_client.calls[0]["options"] == langchain_chat.invoke.call_args.kwargs[
        "options"
    ]
    assert (native_response.provider, native_response.model) == (
        langchain_response.provider,
        langchain_response.model,
    )
    assert isinstance(native_response.content, str)
    assert isinstance(langchain_response.content, str)


class FakeLangChainChat:
    model = "qwen2.5:14b"

    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, object]]] = []
        self.invoke = _RecordingCallable(self.calls)
        self.stream = _RecordingCallable(self.calls)


class _RecordingCallable:
    def __init__(self, calls: list[tuple[object, dict[str, object]]]) -> None:
        self.calls = calls
        self.return_value: object = SimpleNamespace(content="ok")

    def __call__(self, messages: object, **kwargs: object) -> object:
        self.calls.append((messages, kwargs))
        return self.return_value

    @property
    def call_args(self) -> SimpleNamespace:
        messages, kwargs = self.calls[-1]
        return SimpleNamespace(args=(messages,), kwargs=kwargs)


def test_side_by_side_stream_harness_compares_event_kinds_and_order() -> None:
    native_client = FakeNativeClient(
        stream=[_native_response("a"), _native_response("b")],
    )
    native = _adapter(native_client)

    langchain_chat = FakeLangChainChat()
    langchain_chat.stream.return_value = [
        SimpleNamespace(content="a"),
        SimpleNamespace(content="b"),
    ]
    langchain = LangChainOllamaAdapter(
        chat=langchain_chat,  # type: ignore[arg-type]
        model="qwen2.5:14b",
        capability_resolver=_resolver([]),
    )
    messages = [ChatMessage(role="user", content="question")]

    native_events = list(native.stream_messages(messages))
    langchain_events = list(langchain.stream_messages(messages))

    assert [event.kind for event in native_events] == [
        event.kind for event in langchain_events
    ]
    assert len(native_events) == 3
    assert native_events[-1].is_final is True


def test_native_adapter_import_is_independent_of_langchain() -> None:
    script = """
import importlib.abc
import sys

blocked = ("langchain", "langchain_core", "langchain_ollama")

class Blocked(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked or fullname.startswith(tuple(name + "." for name in blocked)):
            raise ImportError("blocked " + fullname)
        return None

sys.meta_path.insert(0, Blocked())
from intergrax.llm_adapters.providers.native_ollama_adapter import NativeOllamaAdapter
print(NativeOllamaAdapter.__name__)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "NativeOllamaAdapter"
