# © Artur Czarnecki. All rights reserved.

"""T-IGX-051-D NT-RW-RUN-04 (NT-ROW-106, EX-L-07): refusal and finish normalization on every adapter path.

The PDC-D3 matrix (``tests/qualification/pdc_x/d3/test_d3_f02_refusal_paths.py``) pins ``generate_messages`` and
``stream_messages`` only. This matrix covers ``generate_with_tools`` and ``stream_with_tools`` of every shipped
adapter that implements them, and the Bedrock InvokeModel paths of every registered codec. Each cell drives the
real adapter method with a fake SDK payload shaped like the SDK type; a fake that is called a second time raises,
because a real SDK stream would have been consumed already. No network, no credential.
"""

from __future__ import annotations

import ast
import io
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.finish_reason import REFUSAL_UNSPECIFIED, LLMFinishReason
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEventKind

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_PROVIDERS = _REPO / "intergrax" / "llm_adapters" / "providers"
_USER = [ChatMessage(role="user", content="hello")]
_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "noop",
        "description": "no-op",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}
_REFUSAL = "I cannot help with that."
C = LLMFinishReason


@pytest.fixture(autouse=True)
def _isolate_provider_modules() -> Iterator[None]:
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if name.startswith("intergrax.llm_adapters.providers.") and name.endswith("_adapter"):
            sys.modules.pop(name, None)


def _once(value: Any) -> Callable[..., Any]:
    """A fake SDK method that answers exactly once (a second call means the adapter re-issued the request)."""
    calls: list[int] = []

    def method(*_: Any, **__: Any) -> Any:
        calls.append(1)
        if len(calls) > 1:
            raise AssertionError("the adapter called the provider a second time for one request")
        return value

    return method


def _final(events: Any) -> LLMAdapterResponse:
    finals = [e for e in events if e.kind is LLMStreamEventKind.FINAL]
    assert len(finals) == 1
    assert finals[0].response is not None
    return finals[0].response


def _gwt(adapter: Any) -> Callable[[], LLMAdapterResponse]:
    return lambda: adapter.generate_with_tools(_USER, [_TOOL])


def _swt(adapter: Any) -> Callable[[], LLMAdapterResponse]:
    return lambda: _final(list(adapter.stream_with_tools(_USER, [_TOOL])))


def _gm(adapter: Any) -> Callable[[], LLMAdapterResponse]:
    return lambda: adapter.generate_messages(_USER)


def _sm(adapter: Any) -> Callable[[], LLMAdapterResponse]:
    return lambda: _final(list(adapter.stream_messages(_USER)))


# ------------------------------------------------------------------------------------------- OpenAI chat


def _openai_chat_adapter(create: Callable[..., Any]) -> Any:
    from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
    from intergrax.llm_adapters.providers.openai_chat_completions_adapter import OpenAIChatCompletionsAdapter

    client = MagicMock()
    client.chat.completions.create.side_effect = create
    return OpenAIChatCompletionsAdapter(client=client, model="gpt-4o-mini", provider=LLMProvider.OPENAI)


_OPENAI_FINISH = {"normal": "stop", "length": "length", "content_filter": "content_filter", "refusal": "stop"}


def _openai_chat_gwt(signal: str) -> Callable[[], LLMAdapterResponse]:
    message = SimpleNamespace(
        content="x" if signal != "refusal" else None,
        tool_calls=None,
        refusal=_REFUSAL if signal == "refusal" else None,
    )
    completion = SimpleNamespace(
        id="r1",
        model="gpt-4o-mini",
        usage=None,
        system_fingerprint=None,
        choices=[SimpleNamespace(message=message, finish_reason=_OPENAI_FINISH[signal])],
    )
    return _gwt(_openai_chat_adapter(_once(completion)))


def _openai_chat_swt(signal: str) -> Callable[[], LLMAdapterResponse]:
    def chunk(content: str | None, finish: str | None, refusal: str | None = None) -> Any:
        delta = SimpleNamespace(content=content, refusal=refusal, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)])

    chunks = [chunk("x" if signal != "refusal" else None, None)]
    if signal == "refusal":
        chunks += [chunk(None, None, "I cannot help "), chunk(None, None, "with that.")]
    chunks.append(chunk(None, _OPENAI_FINISH[signal]))
    return _swt(_openai_chat_adapter(_once(iter(chunks))))


# -------------------------------------------------------------------------------------- OpenAI Responses


def _responses_terminal(signal: str) -> Any:
    status, reason = {
        "normal": ("completed", None),
        "length": ("incomplete", "max_output_tokens"),
        "content_filter": ("incomplete", "content_filter"),
        "refusal": ("completed", None),
    }[signal]
    output: list[Any] = []
    if signal == "refusal":
        output = [SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal", refusal=_REFUSAL)])]
    response = MagicMock()
    response.id = "resp_1"
    response.status = status
    response.incomplete_details = SimpleNamespace(reason=reason) if reason else None
    response.output = output
    response.output_text = "" if signal == "refusal" else "x"
    response.usage = MagicMock(input_tokens=1, output_tokens=1)
    return response


def _responses_adapter(client: Any) -> Any:
    from intergrax.llm_adapters.providers.openai_responses_adapter import OpenAIChatResponsesAdapter

    return OpenAIChatResponsesAdapter(client=client, model="gpt-5.1")


def _responses_gwt(signal: str) -> Callable[[], LLMAdapterResponse]:
    client = MagicMock()
    client.responses.create.side_effect = _once(_responses_terminal(signal))
    return _gwt(_responses_adapter(client))


def _responses_swt(signal: str) -> Callable[[], LLMAdapterResponse]:
    terminal = _responses_terminal(signal)
    terminal_type = "response.completed" if terminal.status == "completed" else "response.incomplete"
    events: list[Any] = [SimpleNamespace(type="response.output_text.delta", delta="x")]
    if signal == "refusal":
        events = [SimpleNamespace(type="response.refusal.delta", delta=_REFUSAL)]
    events.append(SimpleNamespace(type=terminal_type, response=terminal))

    class _Stream:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def __iter__(self) -> Any:
            return iter(events)

        def get_final_response(self) -> Any:
            return terminal

    client = MagicMock()
    client.responses.stream.side_effect = _once(_Stream())
    return _swt(_responses_adapter(client))


# ------------------------------------------------------------------------------------------------ Claude

_CLAUDE_STOP = {"normal": "end_turn", "length": "max_tokens", "refusal": "refusal"}


def _claude_adapter(create: Callable[..., Any]) -> Any:
    pytest.importorskip("anthropic", reason="optional extra llm-anthropic")
    from intergrax.llm_adapters.providers.claude_adapter import ClaudeChatAdapter

    client = MagicMock()
    client.messages.create.side_effect = create
    return ClaudeChatAdapter(client=client, model="claude-3-5-sonnet-20241022")


def _claude_gwt(signal: str) -> Callable[[], LLMAdapterResponse]:
    message = SimpleNamespace(
        id="m1",
        stop_reason=_CLAUDE_STOP[signal],
        content=[SimpleNamespace(type="text", text="x")] if signal != "refusal" else [],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    return lambda: _claude_adapter(_once(message)).generate_with_tools(_USER, [_TOOL])


def _claude_swt(signal: str) -> Callable[[], LLMAdapterResponse]:
    """``messages.create(stream=True)`` returns a raw event stream (no ``get_final_message``)."""
    events = [
        SimpleNamespace(type="message_start", message=SimpleNamespace(id="m1", usage=None)),
        SimpleNamespace(type="content_block_start", index=0, content_block=SimpleNamespace(type="text", text="")),
        SimpleNamespace(type="content_block_delta", index=0, delta=SimpleNamespace(type="text_delta", text="x")),
        SimpleNamespace(type="content_block_stop", index=0),
        SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason=_CLAUDE_STOP[signal]), usage=None),
        SimpleNamespace(type="message_stop"),
    ]
    return lambda: _final(list(_claude_adapter(_once(iter(events))).stream_with_tools(_USER, [_TOOL])))


# ----------------------------------------------------------------------------------------------- Mistral

_MISTRAL_FINISH = {"normal": "stop", "length": "length", "content_filter": "content_filter"}


def _mistral_adapter(complete: Callable[..., Any]) -> Any:
    pytest.importorskip("mistralai", reason="optional extra llm-mistral")
    from intergrax.llm_adapters.providers.mistral_adapter import MistralChatAdapter

    client = MagicMock()
    client.chat.complete.side_effect = complete
    return MistralChatAdapter(client=client, model="mistral-large-latest")


def _mistral_gwt(signal: str) -> Callable[[], LLMAdapterResponse]:
    message = SimpleNamespace(content="x", tool_calls=None, refusal=None)
    completion = SimpleNamespace(
        id="r1",
        model="mistral-large-latest",
        usage=None,
        system_fingerprint=None,
        choices=[SimpleNamespace(message=message, finish_reason=_MISTRAL_FINISH[signal])],
    )
    return lambda: _mistral_adapter(_once(completion)).generate_with_tools(_USER, [_TOOL])


def _mistral_swt(signal: str) -> Callable[[], LLMAdapterResponse]:
    chunks = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="x", tool_calls=None), finish_reason=None)]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=None), finish_reason=_MISTRAL_FINISH[signal]
                )
            ]
        ),
    ]
    return lambda: _final(list(_mistral_adapter(_once(iter(chunks))).stream_with_tools(_USER, [_TOOL])))


# ------------------------------------------------------------------------------------------------ Gemini

_GEMINI_FINISH = {"normal": "STOP", "length": "MAX_TOKENS", "content_filter": "SAFETY"}


def _gemini_chunk(signal: str, text: str | None) -> Any:
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=_GEMINI_FINISH[signal]), content=None)],
        prompt_feedback=None,
    )


def _gemini_adapter(client: Any) -> Any:
    pytest.importorskip("google.genai", reason="optional extra llm-gemini")
    from intergrax.llm_adapters.providers.gemini_adapter import GeminiChatAdapter

    return GeminiChatAdapter(client=client, model="gemini-2.5-flash")


def _gemini_gwt(signal: str) -> Callable[[], LLMAdapterResponse]:
    client = MagicMock()
    client.models.generate_content.side_effect = _once(_gemini_chunk(signal, "x"))
    return lambda: _gemini_adapter(client).generate_with_tools(_USER, [_TOOL])


def _gemini_swt(signal: str) -> Callable[[], LLMAdapterResponse]:
    client = MagicMock()
    client.models.generate_content_stream.side_effect = _once(
        iter([_gemini_chunk("normal", "x"), _gemini_chunk(signal, None)])
    )
    return lambda: _final(list(_gemini_adapter(client).stream_with_tools(_USER, [_TOOL])))


# ------------------------------------------------------------------------------------------------ Cohere

_COHERE_FINISH = {"normal": "COMPLETE", "length": "MAX_TOKENS", "content_filter": "ERROR_TOXIC"}


def _cohere_adapter(client: Any) -> Any:
    pytest.importorskip("cohere", reason="optional extra llm-cohere-native")
    from intergrax.llm_adapters.providers.cohere_native_adapter import CohereNativeChatAdapter

    return CohereNativeChatAdapter(client=client, model="command-r-plus")


def _cohere_gwt(signal: str) -> Callable[[], LLMAdapterResponse]:
    client = MagicMock()
    client.chat.side_effect = _once(
        SimpleNamespace(
            message=SimpleNamespace(content=[SimpleNamespace(type="text", text="x")]),
            finish_reason=_COHERE_FINISH[signal],
        )
    )
    return lambda: _cohere_adapter(client).generate_with_tools(_USER, [_TOOL])


def _cohere_swt(signal: str) -> Callable[[], LLMAdapterResponse]:
    client = MagicMock()
    client.chat_stream.side_effect = _once(
        iter(
            [
                SimpleNamespace(
                    type="content-delta",
                    delta=SimpleNamespace(message=SimpleNamespace(content=SimpleNamespace(text="x"))),
                ),
                SimpleNamespace(type="message-end", delta=SimpleNamespace(finish_reason=_COHERE_FINISH[signal])),
            ]
        )
    )
    return lambda: _final(list(_cohere_adapter(client).stream_with_tools(_USER, [_TOOL])))


# ---------------------------------------------------------------------------------- Bedrock Converse + Invoke

_BEDROCK_CONVERSE_STOP = {"normal": "end_turn", "length": "max_tokens", "content_filter": "guardrail_intervened"}


def _bedrock_adapter(client: Any, model_id: str, *, use_converse: bool) -> Any:
    from intergrax.llm_adapters.providers.aws_bedrock_adapter import BedrockChatAdapter

    return BedrockChatAdapter(client=client, model_id=model_id, region="us-east-1", use_converse=use_converse)


_CONVERSE_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"


def _bedrock_converse_gwt(signal: str) -> Callable[[], LLMAdapterResponse]:
    resp = {"output": {"message": {"content": [{"text": "x"}]}}, "stopReason": _BEDROCK_CONVERSE_STOP[signal]}
    client = SimpleNamespace(converse=_once(resp), converse_stream=MagicMock())
    return _gwt(_bedrock_adapter(client, _CONVERSE_MODEL, use_converse=True))


def _bedrock_converse_swt(signal: str) -> Callable[[], LLMAdapterResponse]:
    events = [
        {"contentBlockDelta": {"delta": {"text": "x"}}},
        {"messageStop": {"stopReason": _BEDROCK_CONVERSE_STOP[signal]}},
    ]
    client = SimpleNamespace(converse=MagicMock(), converse_stream=_once({"stream": iter(events)}))
    return _swt(_bedrock_adapter(client, _CONVERSE_MODEL, use_converse=True))


#: family -> (model id, buffered body builder, stream chunk builders) per signal
def _invoke_bodies(family: str, signal: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if family == "anthropic":
        stop = {"normal": "end_turn", "length": "max_tokens", "refusal": "refusal"}[signal]
        body = {"content": [{"type": "text", "text": "x"}], "stop_reason": stop}
        chunks = [
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "x"}},
            {"type": "message_delta", "delta": {"stop_reason": stop}},
        ]
        return body, chunks
    if family == "meta":
        stop = {"normal": "stop", "length": "length"}[signal]
        return {"generation": "x", "stop_reason": stop}, [
            {"generation": "x", "stop_reason": None},
            {"generation": "", "stop_reason": stop},
        ]
    if family == "mistral":
        # Bedrock Mistral reports the stop reason inside ``outputs[0]`` only (buffered body and stream chunk).
        stop = {"normal": "stop", "length": "length"}[signal]
        return {"outputs": [{"text": "x", "stop_reason": stop}]}, [
            {"outputs": [{"text": "x", "stop_reason": None}]},
            {"outputs": [{"text": "", "stop_reason": stop}]},
        ]
    stop = {"normal": "FINISH", "length": "LENGTH", "content_filter": "CONTENT_FILTERED"}[signal]
    return {"results": [{"outputText": "x", "completionReason": stop}]}, [
        {"outputText": "x", "completionReason": None},
        {"outputText": "", "completionReason": stop},
    ]


_INVOKE_MODELS = {
    "anthropic": "anthropic.claude-3-haiku-20240307-v1:0",
    "meta": "meta.llama3-1-8b-instruct-v1:0",
    "mistral": "mistral.mistral-large-2402-v1:0",
    "amazon": "amazon.titan-text-express-v1",
}
_INVOKE_SIGNALS = {
    "anthropic": ("normal", "length", "refusal"),
    "meta": ("normal", "length"),
    "mistral": ("normal", "length"),
    "amazon": ("normal", "length", "content_filter"),
}


def _invoke_client(family: str, signal: str) -> Any:
    body, chunks = _invoke_bodies(family, signal)
    stream = [{"chunk": {"bytes": json.dumps(chunk).encode("utf-8")}} for chunk in chunks]
    return SimpleNamespace(
        invoke_model=_once({"body": io.BytesIO(json.dumps(body).encode("utf-8"))}),
        invoke_model_with_response_stream=_once({"body": iter(stream)}),
    )


def _invoke_builder(family: str, method: str) -> Callable[[str], Callable[[], LLMAdapterResponse]]:
    def build(signal: str) -> Callable[[], LLMAdapterResponse]:
        adapter = _bedrock_adapter(_invoke_client(family, signal), _INVOKE_MODELS[family], use_converse=False)
        return {"generate_messages": _gm, "stream_messages": _sm, "generate_with_tools": _gwt}[method](adapter)

    return build


# ------------------------------------------------------------------------------------------- Ollama (both)

_OLLAMA_DONE = {"normal": "stop", "length": "length"}


def _native_ollama_gwt(signal: str) -> Callable[[], LLMAdapterResponse]:
    from intergrax.llm_adapters.providers.native_ollama_adapter import NativeOllamaAdapter
    from intergrax.llm_adapters.providers.ollama_capabilities import OllamaModelCapabilityResolver

    client = MagicMock()
    client.chat.side_effect = _once(
        {"message": {"content": "x", "tool_calls": None}, "done": True, "done_reason": _OLLAMA_DONE[signal]}
    )
    resolver = OllamaModelCapabilityResolver(
        show_model=lambda _m: SimpleNamespace(capabilities=["completion", "tools"])
    )
    adapter = NativeOllamaAdapter(client=client, model="qwen2.5:14b", capability_resolver=resolver)
    return _gwt(adapter)


def _langchain_ollama_gwt(signal: str) -> Callable[[], LLMAdapterResponse]:
    pytest.importorskip("langchain_core", reason="optional extra llm-langchain-ollama")
    from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter
    from intergrax.llm_adapters.providers.ollama_capabilities import OllamaModelCapabilityResolver

    chat = MagicMock()
    chat.model = "qwen2.5:14b"
    bound = chat.bind_tools.return_value
    bound.invoke.side_effect = _once(
        SimpleNamespace(
            content="x",
            tool_calls=[],
            invalid_tool_calls=[],
            response_metadata={"done_reason": _OLLAMA_DONE[signal]},
        )
    )
    resolver = OllamaModelCapabilityResolver(
        show_model=lambda _m: SimpleNamespace(capabilities=["completion", "tools"])
    )
    adapter = LangChainOllamaAdapter(chat=chat, model="qwen2.5:14b", capability_resolver=resolver)
    return _gwt(adapter)


# --------------------------------------------------------------------------------------------- the matrix

_FULL = ("normal", "length", "content_filter")
_PATHS: dict[str, tuple[Callable[[str], Callable[[], LLMAdapterResponse]], tuple[str, ...]]] = {
    "openai-chat/generate_with_tools": (_openai_chat_gwt, (*_FULL, "refusal")),
    "openai-chat/stream_with_tools": (_openai_chat_swt, (*_FULL, "refusal")),
    "openai-responses/generate_with_tools": (_responses_gwt, (*_FULL, "refusal")),
    "openai-responses/stream_with_tools": (_responses_swt, (*_FULL, "refusal")),
    "claude/generate_with_tools": (_claude_gwt, ("normal", "length", "refusal")),
    "claude/stream_with_tools": (_claude_swt, ("normal", "length", "refusal")),
    "mistral/generate_with_tools": (_mistral_gwt, _FULL),
    "mistral/stream_with_tools": (_mistral_swt, _FULL),
    "gemini/generate_with_tools": (_gemini_gwt, _FULL),
    "gemini/stream_with_tools": (_gemini_swt, _FULL),
    "cohere/generate_with_tools": (_cohere_gwt, _FULL),
    "cohere/stream_with_tools": (_cohere_swt, _FULL),
    "bedrock-converse/generate_with_tools": (_bedrock_converse_gwt, _FULL),
    "bedrock-converse/stream_with_tools": (_bedrock_converse_swt, _FULL),
    "native-ollama/generate_with_tools": (_native_ollama_gwt, ("normal", "length")),
    "langchain-ollama/generate_with_tools": (_langchain_ollama_gwt, ("normal", "length")),
}
for _family, _signals in _INVOKE_SIGNALS.items():
    for _method in ("generate_messages", "stream_messages"):
        _PATHS[f"bedrock-invoke-{_family}/{_method}"] = (_invoke_builder(_family, _method), _signals)
_PATHS["bedrock-invoke-anthropic/generate_with_tools"] = (
    _invoke_builder("anthropic", "generate_with_tools"),
    _INVOKE_SIGNALS["anthropic"],
)

_CASES = [(path, signal) for path, (_, signals) in _PATHS.items() for signal in signals]


def _expected(signal: str) -> tuple[LLMFinishReason, str | None]:
    return {
        "normal": (C.COMPLETED, None),
        "length": (C.LENGTH, None),
        "content_filter": (C.CONTENT_FILTER, None),
        "refusal": (C.REFUSAL, _REFUSAL),
    }[signal]


@pytest.mark.parametrize(("path", "signal"), _CASES, ids=[f"{p}:{s}" for p, s in _CASES])
def test_the_provider_stop_signal_reaches_the_response(path: str, signal: str) -> None:
    response = _PATHS[path][0](signal)()
    finish, refusal = _expected(signal)
    if signal == "refusal" and (path.startswith("claude/") or path.startswith("bedrock-invoke-anthropic/")):
        refusal = REFUSAL_UNSPECIFIED  # Anthropic reports the refusal stop reason without a message
    assert response.finish_reason is finish
    assert response.refusal == refusal


def _implemented_methods() -> dict[str, set[str]]:
    """Adapter class name -> tool methods it defines itself (AST over the provider package)."""
    tool_methods = {"generate_with_tools", "stream_with_tools"}
    found: dict[str, set[str]] = {}
    for path in sorted(_PROVIDERS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not any(isinstance(base, ast.Name) and base.id == "BaseLLMAdapter" for base in node.bases):
                continue
            found[node.name] = {
                item.name for item in node.body if isinstance(item, ast.FunctionDef) and item.name in tool_methods
            }
    return found


_CELL_PREFIX = {
    "OpenAIChatCompletionsAdapter": "openai-chat",
    "OpenAIChatResponsesAdapter": "openai-responses",
    "ClaudeChatAdapter": "claude",
    "MistralChatAdapter": "mistral",
    "GeminiChatAdapter": "gemini",
    "CohereNativeChatAdapter": "cohere",
    "BedrockChatAdapter": "bedrock-converse",
    "NativeOllamaAdapter": "native-ollama",
    "LangChainOllamaAdapter": "langchain-ollama",
}


def test_every_implemented_tool_method_of_every_shipped_adapter_has_a_cell() -> None:
    implemented = _implemented_methods()
    assert set(implemented) == set(_CELL_PREFIX), "a new shipped adapter needs matrix cells"
    for adapter_name, methods in implemented.items():
        for method in methods:
            assert f"{_CELL_PREFIX[adapter_name]}/{method}" in _PATHS, (adapter_name, method)


def test_every_registered_bedrock_codec_has_invoke_cells() -> None:
    from intergrax.llm_adapters.providers.aws_bedrock_adapter import BedrockChatAdapter, BedrockModelFamily

    registered = {family.value for family in BedrockChatAdapter._CODECS}
    assert registered == {BedrockModelFamily(f).value for f in _INVOKE_SIGNALS}
    for family in _INVOKE_SIGNALS:
        assert f"bedrock-invoke-{family}/generate_messages" in _PATHS
        assert f"bedrock-invoke-{family}/stream_messages" in _PATHS
