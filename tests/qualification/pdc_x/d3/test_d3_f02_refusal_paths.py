# © Artur Czarnecki. All rights reserved.

"""PDC-D3 F-D3-02 — the provider stop / refusal signal reaches the response on every adapter path.

Before the repair most stream paths and several buffered paths built the final response with the default
finish reason, so a refused or filtered answer ended as ``COMPLETED`` with empty content. Each case drives a
real adapter method with a fake SDK payload (no network, no credential). The payload shapes are modelled
from the provider SDK types; live parity with the real services is Path B (OD-PDC-PATHB-01).
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.finish_reason import REFUSAL_UNSPECIFIED, LLMFinishReason
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEventKind

pytestmark = pytest.mark.unit

_USER = [ChatMessage(role="user", content="hello")]
C = LLMFinishReason
#: (signal id, expected finish reason, expected refusal text or None)
_SIGNALS = {
    "normal": (C.COMPLETED, None),
    "length": (C.LENGTH, None),
    "content_filter": (C.CONTENT_FILTER, None),
}


def _final(events: Any) -> LLMAdapterResponse:
    finals = [e for e in events if e.kind is LLMStreamEventKind.FINAL]
    assert len(finals) == 1
    assert finals[0].response is not None
    return finals[0].response


def _stream(adapter: Any) -> LLMAdapterResponse:
    return _final(list(adapter.stream_messages(_USER)))


def _buffered(adapter: Any) -> LLMAdapterResponse:
    return adapter.generate_messages(_USER)


# ---------------------------------------------------------------------------------------------
# Builders: signal id -> a zero-argument callable returning the adapter's response for that path.
# ---------------------------------------------------------------------------------------------


def _openai_chat_stream(signal: str) -> Callable[[], LLMAdapterResponse]:
    from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
    from intergrax.llm_adapters.providers.openai_chat_completions_adapter import OpenAIChatCompletionsAdapter

    finish = {"normal": "stop", "length": "length", "content_filter": "content_filter", "refusal": "stop"}[signal]
    refusal = "I cannot help with that." if signal == "refusal" else None

    def chunk(content: str | None, finish_reason: str | None, refusal_piece: str | None = None) -> Any:
        delta = SimpleNamespace(content=content, refusal=refusal_piece, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)])

    chunks = [chunk("x" if signal != "refusal" else None, None)]
    if refusal:
        chunks.append(chunk(None, None, "I cannot help "))
        chunks.append(chunk(None, None, "with that."))
    chunks.append(chunk(None, finish))
    client = MagicMock()
    client.chat.completions.create.return_value = iter(chunks)
    adapter = OpenAIChatCompletionsAdapter(client=client, model="gpt-4o-mini", provider=LLMProvider.OPENAI)
    return lambda: _stream(adapter)


def _openai_responses_stream(signal: str) -> Callable[[], LLMAdapterResponse]:
    from intergrax.llm_adapters.providers.openai_responses_adapter import OpenAIChatResponsesAdapter

    status, reason = {
        "normal": ("completed", None),
        "length": ("incomplete", "max_output_tokens"),
        "content_filter": ("incomplete", "content_filter"),
        "refusal": ("completed", None),
    }[signal]
    terminal_type = "response.completed" if status == "completed" else "response.incomplete"
    events: list[Any] = [SimpleNamespace(type="response.output_text.delta", delta="x")]
    if signal == "refusal":
        events = [
            SimpleNamespace(type="response.refusal.delta", delta="I cannot help "),
            SimpleNamespace(type="response.refusal.delta", delta="with that."),
        ]
    details = SimpleNamespace(reason=reason) if reason else None
    events.append(
        SimpleNamespace(
            type=terminal_type,
            response=SimpleNamespace(status=status, incomplete_details=details, output=[]),
        )
    )

    class _Stream:
        def __enter__(self) -> _Stream:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def __iter__(self) -> Any:
            return iter(events)

    client = MagicMock()
    client.responses.stream.return_value = _Stream()
    adapter = OpenAIChatResponsesAdapter(client=client, model="gpt-5.1")
    return lambda: _stream(adapter)


def _openai_responses_buffered(signal: str) -> Callable[[], LLMAdapterResponse]:
    from intergrax.llm_adapters.providers.openai_responses_adapter import OpenAIChatResponsesAdapter

    status, reason = {
        "normal": ("completed", None),
        "length": ("incomplete", "max_output_tokens"),
        "content_filter": ("incomplete", "content_filter"),
        "refusal": ("completed", None),
    }[signal]
    output: list[Any] = []
    if signal == "refusal":
        part = SimpleNamespace(type="refusal", refusal="I cannot help with that.")
        output = [SimpleNamespace(type="message", content=[part])]
    response = MagicMock()
    response.id = "resp_1"
    response.status = status
    response.incomplete_details = SimpleNamespace(reason=reason) if reason else None
    response.output = output
    response.output_text = ""
    response.usage = MagicMock(input_tokens=1, output_tokens=1)
    client = MagicMock()
    client.responses.create.return_value = response
    adapter = OpenAIChatResponsesAdapter(client=client, model="gpt-5.1")
    return lambda: _buffered(adapter)


def _claude_stream(signal: str) -> Callable[[], LLMAdapterResponse]:
    from intergrax.llm_adapters.providers.claude_adapter import ClaudeChatAdapter

    stop = {"normal": "end_turn", "length": "max_tokens", "content_filter": "end_turn", "refusal": "refusal"}[signal]
    events = [
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="x")),
        SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason=stop)),
    ]
    client = MagicMock()
    client.messages.create.return_value = iter(events)
    adapter = ClaudeChatAdapter(client=client, model="claude-3-5-sonnet-20241022")
    return lambda: _stream(adapter)


def _mistral_stream(signal: str) -> Callable[[], LLMAdapterResponse]:
    from intergrax.llm_adapters.providers.mistral_adapter import MistralChatAdapter

    finish = {"normal": "stop", "length": "length", "content_filter": "content_filter"}[signal]
    chunks = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="x", tool_calls=None), finish_reason=None)]
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=None), finish_reason=finish)]
        ),
    ]
    client = MagicMock()
    client.chat.complete.return_value = iter(chunks)
    adapter = MistralChatAdapter(client=client, model="mistral-large-latest")
    return lambda: _stream(adapter)


def _gemini_chunk(signal: str, text: str | None) -> Any:
    finish = {"normal": "STOP", "length": "MAX_TOKENS", "content_filter": "SAFETY"}[signal]
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish), content=None)],
        prompt_feedback=None,
    )


def _gemini_adapter(client: Any) -> Any:
    from intergrax.llm_adapters.providers.gemini_adapter import GeminiChatAdapter

    return GeminiChatAdapter(client=client, model="gemini-2.5-flash")


def _gemini_stream(signal: str) -> Callable[[], LLMAdapterResponse]:
    client = MagicMock()
    client.chats.create.return_value.send_message_stream.return_value = iter(
        [_gemini_chunk("normal", "x"), _gemini_chunk(signal, None)]
    )
    adapter = _gemini_adapter(client)
    return lambda: _stream(adapter)


def _gemini_stream_prompt_blocked(_: str) -> Callable[[], LLMAdapterResponse]:
    blocked = SimpleNamespace(
        text=None, candidates=[], prompt_feedback=SimpleNamespace(block_reason=SimpleNamespace(name="SAFETY"))
    )
    client = MagicMock()
    client.chats.create.return_value.send_message_stream.return_value = iter([blocked])
    adapter = _gemini_adapter(client)
    return lambda: _stream(adapter)


def _gemini_buffered(signal: str) -> Callable[[], LLMAdapterResponse]:
    client = MagicMock()
    client.chats.create.return_value.send_message.return_value = _gemini_chunk(signal, "")
    adapter = _gemini_adapter(client)
    return lambda: _buffered(adapter)


def _cohere_adapter(client: Any) -> Any:
    from intergrax.llm_adapters.providers.cohere_native_adapter import CohereNativeChatAdapter

    return CohereNativeChatAdapter(client=client, model="command-r-plus")


def _cohere_finish(signal: str) -> str:
    return {"normal": "COMPLETE", "length": "MAX_TOKENS", "content_filter": "ERROR_TOXIC"}[signal]


def _cohere_stream(signal: str) -> Callable[[], LLMAdapterResponse]:
    events = [
        SimpleNamespace(
            type="content-delta", delta=SimpleNamespace(message=SimpleNamespace(content=SimpleNamespace(text="x")))
        ),
        SimpleNamespace(type="message-end", delta=SimpleNamespace(finish_reason=_cohere_finish(signal))),
    ]
    client = MagicMock()
    client.chat_stream.return_value = iter(events)
    adapter = _cohere_adapter(client)
    return lambda: _stream(adapter)


def _cohere_buffered(signal: str) -> Callable[[], LLMAdapterResponse]:
    resp = SimpleNamespace(
        message=SimpleNamespace(content=[SimpleNamespace(text="x")]), finish_reason=_cohere_finish(signal)
    )
    client = MagicMock()
    client.chat.return_value = resp
    adapter = _cohere_adapter(client)
    return lambda: _buffered(adapter)


def _bedrock_adapter(client: Any) -> Any:
    from intergrax.llm_adapters.providers.aws_bedrock_adapter import BedrockChatAdapter

    return BedrockChatAdapter(
        client=client, model_id="anthropic.claude-3-haiku-20240307-v1:0", region="us-east-1", use_converse=True
    )


def _bedrock_stop(signal: str) -> str:
    return {"normal": "end_turn", "length": "max_tokens", "content_filter": "guardrail_intervened"}[signal]


def _bedrock_stream(signal: str) -> Callable[[], LLMAdapterResponse]:
    events = [
        {"contentBlockDelta": {"delta": {"text": "x"}}},
        {"messageStop": {"stopReason": _bedrock_stop(signal)}},
    ]
    client = SimpleNamespace(converse=MagicMock(), converse_stream=MagicMock(return_value={"stream": iter(events)}))
    adapter = _bedrock_adapter(client)
    return lambda: _stream(adapter)


def _bedrock_buffered(signal: str) -> Callable[[], LLMAdapterResponse]:
    resp = {"output": {"message": {"content": [{"text": "x"}]}}, "stopReason": _bedrock_stop(signal)}
    client = SimpleNamespace(converse=MagicMock(return_value=resp), converse_stream=MagicMock())
    adapter = _bedrock_adapter(client)
    return lambda: _buffered(adapter)


def _langchain_ollama_adapter(chat: Any) -> Any:
    pytest.importorskip("langchain_core")
    from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter
    from intergrax.llm_adapters.providers.ollama_capabilities import OllamaModelCapabilityResolver

    chat.model = "qwen2.5:14b"
    resolver = OllamaModelCapabilityResolver(show_model=lambda _m: SimpleNamespace(capabilities=["completion"]))
    return LangChainOllamaAdapter(chat=chat, model="qwen2.5:14b", capability_resolver=resolver)


def _done_reason(signal: str) -> str:
    return {"normal": "stop", "length": "length"}[signal]


def _langchain_ollama_stream(signal: str) -> Callable[[], LLMAdapterResponse]:
    chat = MagicMock()
    chat.stream.return_value = iter(
        [
            SimpleNamespace(content="x", response_metadata={}),
            SimpleNamespace(content="", response_metadata={"done_reason": _done_reason(signal)}),
        ]
    )
    adapter = _langchain_ollama_adapter(chat)
    return lambda: _stream(adapter)


def _langchain_ollama_buffered(signal: str) -> Callable[[], LLMAdapterResponse]:
    chat = MagicMock()
    chat.invoke.return_value = SimpleNamespace(content="x", response_metadata={"done_reason": _done_reason(signal)})
    adapter = _langchain_ollama_adapter(chat)
    return lambda: _buffered(adapter)


def _native_ollama_adapter(client: Any) -> Any:
    from intergrax.llm_adapters.providers.native_ollama_adapter import NativeOllamaAdapter
    from intergrax.llm_adapters.providers.ollama_capabilities import OllamaModelCapabilityResolver

    resolver = OllamaModelCapabilityResolver(show_model=lambda _m: SimpleNamespace(capabilities=["completion"]))
    return NativeOllamaAdapter(client=client, model="qwen2.5:14b", capability_resolver=resolver)


def _native_ollama_stream(signal: str) -> Callable[[], LLMAdapterResponse]:
    chunks = [
        {"message": {"content": "x"}},
        {"message": {"content": ""}, "done": True, "done_reason": _done_reason(signal)},
    ]
    client = MagicMock()
    client.chat.return_value = iter(chunks)
    adapter = _native_ollama_adapter(client)
    return lambda: _stream(adapter)


def _native_ollama_buffered(signal: str) -> Callable[[], LLMAdapterResponse]:
    client = MagicMock()
    client.chat.return_value = {"message": {"content": "x"}, "done": True, "done_reason": _done_reason(signal)}
    adapter = _native_ollama_adapter(client)
    return lambda: _buffered(adapter)


#: path id -> (builder, signals that exist for the provider on that path)
_FULL = ("normal", "length", "content_filter")
_NO_FILTER = ("normal", "length")
_PATHS: dict[str, tuple[Callable[[str], Callable[[], LLMAdapterResponse]], tuple[str, ...]]] = {
    "openai-chat/stream_messages": (_openai_chat_stream, (*_FULL, "refusal")),
    "openai-responses/stream_messages": (_openai_responses_stream, (*_FULL, "refusal")),
    "openai-responses/generate_messages": (_openai_responses_buffered, (*_FULL, "refusal")),
    "claude/stream_messages": (_claude_stream, ("normal", "length", "refusal")),
    "mistral/stream_messages": (_mistral_stream, _FULL),
    "gemini/stream_messages": (_gemini_stream, _FULL),
    "gemini/generate_messages": (_gemini_buffered, _FULL),
    "cohere/stream_messages": (_cohere_stream, _FULL),
    "cohere/generate_messages": (_cohere_buffered, _FULL),
    "bedrock-converse/stream_messages": (_bedrock_stream, _FULL),
    "bedrock-converse/generate_messages": (_bedrock_buffered, _FULL),
    "langchain-ollama/stream_messages": (_langchain_ollama_stream, _NO_FILTER),
    "langchain-ollama/generate_messages": (_langchain_ollama_buffered, _NO_FILTER),
    "native-ollama/stream_messages": (_native_ollama_stream, _NO_FILTER),
    "native-ollama/generate_messages": (_native_ollama_buffered, _NO_FILTER),
}
_CASES = [(path, signal) for path, (_, signals) in _PATHS.items() for signal in signals]


def _expected(signal: str) -> tuple[LLMFinishReason, str | None]:
    if signal == "refusal":
        return C.REFUSAL, "I cannot help with that."
    return _SIGNALS[signal]


@pytest.mark.parametrize(("path", "signal"), _CASES, ids=[f"{p}:{s}" for p, s in _CASES])
def test_provider_stop_signal_reaches_the_response(path: str, signal: str) -> None:
    response = _PATHS[path][0](signal)()
    finish, refusal = _expected(signal)
    if path.startswith("claude/") and signal == "refusal":
        refusal = REFUSAL_UNSPECIFIED  # Anthropic reports the refusal stop reason without a message
    assert response.finish_reason is finish
    assert response.refusal == refusal
    if finish is C.REFUSAL:
        assert response.finish_reason is not C.COMPLETED


def test_gemini_prompt_block_is_a_content_filter_not_an_empty_success() -> None:
    response = _gemini_stream_prompt_blocked("content_filter")()
    assert response.finish_reason is C.CONTENT_FILTER
    assert response.content == ""


def test_the_matrix_covers_every_shipped_stream_adapter_path() -> None:
    """A new provider stream path must be added to the matrix (or listed with a reason)."""
    covered = {path.split("/", 1)[0] for path in _PATHS if path.endswith("/stream_messages")}
    assert covered == {
        "openai-chat",
        "openai-responses",
        "claude",
        "mistral",
        "gemini",
        "cohere",
        "bedrock-converse",
        "langchain-ollama",
        "native-ollama",
    }


@pytest.mark.parametrize(
    "bedrock_body",
    [
        {"stop_reason": "max_tokens"},
        {"completionReason": "LENGTH"},
        {"delta": {"stop_reason": "max_tokens"}},
        {"results": [{"completionReason": "LENGTH"}]},
    ],
)
def test_bedrock_invoke_stop_spellings_map_to_length(bedrock_body: dict[str, Any]) -> None:
    from intergrax.llm_adapters._shared.stream_finish_signal import finish_reason_from_raw, invoke_stop_signal

    assert finish_reason_from_raw(invoke_stop_signal(bedrock_body)) is C.LENGTH


def test_titan_content_filter_and_unknown_signals() -> None:
    from intergrax.llm_adapters._shared.stream_finish_signal import finish_reason_from_raw, invoke_stop_signal

    assert (
        finish_reason_from_raw(invoke_stop_signal({"results": [{"completionReason": "CONTENT_FILTERED"}]}))
        is C.CONTENT_FILTER
    )
    assert finish_reason_from_raw(invoke_stop_signal({"results": [{"completionReason": "FINISH"}]})) is C.COMPLETED
    assert finish_reason_from_raw(invoke_stop_signal(None)) is C.COMPLETED
    assert finish_reason_from_raw("a-value-nobody-knows") is C.UNKNOWN


def test_a_normal_stop_with_tool_calls_is_reported_as_tool_calls() -> None:
    from intergrax.llm_adapters._shared.stream_finish_signal import finish_reason_from_raw

    assert finish_reason_from_raw("stop", has_tool_calls=True) is C.TOOL_CALLS
    assert finish_reason_from_raw("length", has_tool_calls=True) is C.LENGTH
