# © Artur Czarnecki. All rights reserved.

"""T-IGX-051-D NT-RW-RUN-02 (NT-ROW-104, EX-D3-01): the effective call timeout reaches the provider SDK call.

Before the change the effective timeout (``timeout_sec`` clamped by the active execution deadline) was computed
for admission and then dropped: no provider call site passed it to the SDK. Each shipped adapter must now pass
it through the mechanism its SDK offers, or declare that it cannot and refuse a call that carries a timeout
(Operator ruling Q-RUN-02: fail closed). Fake SDK clients only (no network, no credential).
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.call_config import LLMCallConfig
from intergrax.llm_adapters._shared.resilience import reset_provider_resilience
from intergrax.llm_adapters.contracts.capabilities import LLMCapabilityMismatchError
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEventKind

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_PROVIDERS = _REPO / "intergrax" / "llm_adapters" / "providers"
_USER = [ChatMessage(role="user", content="hello")]
_TIMEOUT = 7.5

#: Direct ``BaseLLMAdapter`` subclasses shipped in the provider package -> declared mechanism.
#: Subclasses (Azure, the OpenAI-compatible family, Vertex) inherit the mechanism of their base.
_MECHANISMS: dict[str, str] = {
    "OpenAIChatCompletionsAdapter": "PER_CALL",
    "OpenAIChatResponsesAdapter": "PER_CALL",
    "ClaudeChatAdapter": "PER_CALL",
    "MistralChatAdapter": "PER_CALL",
    "CohereNativeChatAdapter": "PER_CALL",
    "GeminiChatAdapter": "PER_CALL",
    "NativeOllamaAdapter": "PER_CALL_DERIVED_CLIENT",
    "LangChainOllamaAdapter": "NONE",
    "BedrockChatAdapter": "NONE",
}


@pytest.fixture(autouse=True)
def _isolate_provider_modules() -> Iterator[None]:
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if name.startswith("intergrax.llm_adapters.providers.") and name.endswith("_adapter"):
            sys.modules.pop(name, None)


def _with_timeout(adapter: Any, timeout: float | None = _TIMEOUT) -> Any:
    adapter.call_config = LLMCallConfig(timeout_sec=timeout)
    reset_provider_resilience(adapter._provider_slug())
    return adapter


def _final_of(events: Any) -> Any:
    finals = [e for e in events if e.kind is LLMStreamEventKind.FINAL]
    assert len(finals) == 1
    return finals[0].response


def _chat_completion(model: str = "served") -> Any:
    message = SimpleNamespace(content="x", tool_calls=None, refusal=None)
    return SimpleNamespace(
        id="r1",
        model=model,
        usage=None,
        system_fingerprint=None,
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
    )


def _chat_chunks() -> list[Any]:
    delta = SimpleNamespace(content="x", refusal=None, tool_calls=None)
    end = SimpleNamespace(content=None, refusal=None, tool_calls=None)
    return [
        SimpleNamespace(model="served", choices=[SimpleNamespace(delta=delta, finish_reason=None)]),
        SimpleNamespace(model="served", choices=[SimpleNamespace(delta=end, finish_reason="stop")]),
    ]


# ---------------------------------------------------------------------------------------------
# Per adapter: (build adapter + fake client, run buffered, run streamed, read forwarded seconds)
# ---------------------------------------------------------------------------------------------


def _openai_chat(stream: bool) -> tuple[Any, Callable[[], float | None]]:
    from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
    from intergrax.llm_adapters.providers.openai_chat_completions_adapter import OpenAIChatCompletionsAdapter

    client = MagicMock()
    client.chat.completions.create.return_value = iter(_chat_chunks()) if stream else _chat_completion()
    adapter = OpenAIChatCompletionsAdapter(client=client, model="gpt-4o-mini", provider=LLMProvider.OPENAI)
    return adapter, lambda: client.chat.completions.create.call_args.kwargs.get("timeout")


def _openai_responses(stream: bool) -> tuple[Any, Callable[[], float | None]]:
    from intergrax.llm_adapters.providers.openai_responses_adapter import OpenAIChatResponsesAdapter

    client = MagicMock()
    if stream:
        events = [
            SimpleNamespace(type="response.output_text.delta", delta="x"),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(status="completed", incomplete_details=None, output=[], model="served"),
            ),
        ]

        class _Stream:
            def __enter__(self) -> Any:
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def __iter__(self) -> Any:
                return iter(events)

        client.responses.stream.return_value = _Stream()
        adapter = OpenAIChatResponsesAdapter(client=client, model="gpt-5.1")
        return adapter, lambda: client.responses.stream.call_args.kwargs.get("timeout")
    response = MagicMock()
    response.id = "resp_1"
    response.status = "completed"
    response.incomplete_details = None
    response.output = []
    response.output_text = "x"
    response.model = "served"
    response.usage = MagicMock(input_tokens=1, output_tokens=1)
    client.responses.create.return_value = response
    adapter = OpenAIChatResponsesAdapter(client=client, model="gpt-5.1")
    return adapter, lambda: client.responses.create.call_args.kwargs.get("timeout")


def _claude(stream: bool) -> tuple[Any, Callable[[], float | None]]:
    pytest.importorskip("anthropic", reason="optional extra llm-anthropic")
    from intergrax.llm_adapters.providers.claude_adapter import ClaudeChatAdapter

    client = MagicMock()
    if stream:
        client.messages.create.return_value = iter(
            [
                SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="x")),
                SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason="end_turn")),
            ]
        )
    else:
        client.messages.create.return_value = SimpleNamespace(
            id="m1",
            model="served",
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="x")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
    adapter = ClaudeChatAdapter(client=client, model="claude-3-5-sonnet-20241022")
    return adapter, lambda: client.messages.create.call_args.kwargs.get("timeout")


def _mistral(stream: bool) -> tuple[Any, Callable[[], float | None]]:
    pytest.importorskip("mistralai", reason="optional extra llm-mistral")
    from intergrax.llm_adapters.providers.mistral_adapter import MistralChatAdapter

    client = MagicMock()
    client.chat.complete.return_value = iter(_chat_chunks()) if stream else _chat_completion()
    adapter = MistralChatAdapter(client=client, model="mistral-large-latest")

    def forwarded() -> float | None:
        value = client.chat.complete.call_args.kwargs.get("timeout_ms")
        return None if value is None else value / 1000.0

    return adapter, forwarded


def _cohere(stream: bool) -> tuple[Any, Callable[[], float | None]]:
    pytest.importorskip("cohere", reason="optional extra llm-cohere-native")
    from intergrax.llm_adapters.providers.cohere_native_adapter import CohereNativeChatAdapter

    client = MagicMock()
    client.chat.return_value = SimpleNamespace(
        message=SimpleNamespace(content=[SimpleNamespace(text="x")]), finish_reason="COMPLETE"
    )
    client.chat_stream.return_value = iter(
        [
            SimpleNamespace(
                type="content-delta",
                delta=SimpleNamespace(message=SimpleNamespace(content=SimpleNamespace(text="x"))),
            ),
            SimpleNamespace(type="message-end", delta=SimpleNamespace(finish_reason="COMPLETE")),
        ]
    )
    adapter = CohereNativeChatAdapter(client=client, model="command-r-plus")
    method = client.chat_stream if stream else client.chat

    def forwarded() -> float | None:
        options = method.call_args.kwargs.get("request_options")
        return None if not options else options.get("timeout_in_seconds")

    return adapter, forwarded


def _gemini(stream: bool) -> tuple[Any, Callable[[], float | None]]:
    pytest.importorskip("google.genai", reason="optional extra llm-gemini")
    from intergrax.llm_adapters.providers.gemini_adapter import GeminiChatAdapter

    chunk = SimpleNamespace(
        text="x",
        model_version="served",
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"), content=None)],
        prompt_feedback=None,
    )
    client = MagicMock()
    session = client.chats.create.return_value
    session.send_message.return_value = chunk
    session.send_message_stream.return_value = iter([chunk])
    adapter = GeminiChatAdapter(client=client, model="gemini-2.5-flash")
    method = session.send_message_stream if stream else session.send_message

    def forwarded() -> float | None:
        config = method.call_args.kwargs.get("config")
        options = getattr(config, "http_options", None)
        millis = getattr(options, "timeout", None)
        return None if millis is None else millis / 1000.0

    return adapter, forwarded


def _native_ollama_owned_client(
    stream: bool, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, Callable[[], float | None]]:
    constructed: list[dict[str, Any]] = []

    class _FakeOllamaClient:
        def __init__(self, host: str | None = None, **kwargs: Any) -> None:
            constructed.append({"host": host, **kwargs})

        def chat(self, **kwargs: Any) -> Any:
            if kwargs.get("stream"):
                return iter(
                    [
                        {"message": {"content": "x"}},
                        {"message": {"content": ""}, "done": True, "done_reason": "stop", "model": "served"},
                    ]
                )
            return {"message": {"content": "x"}, "done": True, "done_reason": "stop", "model": "served"}

    fake_module = ModuleType("ollama")
    fake_module.Client = _FakeOllamaClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ollama", fake_module)
    from intergrax.llm_adapters.providers.native_ollama_adapter import NativeOllamaAdapter
    from intergrax.llm_adapters.providers.ollama_capabilities import OllamaModelCapabilityResolver

    resolver = OllamaModelCapabilityResolver(show_model=lambda _m: SimpleNamespace(capabilities=["completion"]))
    adapter = NativeOllamaAdapter(model="qwen2.5:14b", capability_resolver=resolver, base_url="http://ollama.test")

    def forwarded() -> float | None:
        timeouts = [entry.get("timeout") for entry in constructed if entry.get("timeout") is not None]
        return timeouts[-1] if timeouts else None

    return adapter, forwarded


_PER_CALL_CASES: dict[str, Callable[[bool], tuple[Any, Callable[[], float | None]]]] = {
    "openai-chat": _openai_chat,
    "openai-responses": _openai_responses,
    "claude": _claude,
    "mistral": _mistral,
    "cohere": _cohere,
    "gemini": _gemini,
}


def _run(adapter: Any, stream: bool) -> Any:
    if stream:
        return _final_of(list(adapter.stream_messages(_USER)))
    return adapter.generate_messages(_USER)


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "stream"])
@pytest.mark.parametrize("case", sorted(_PER_CALL_CASES))
def test_the_configured_timeout_reaches_the_sdk_call(case: str, stream: bool) -> None:
    adapter, forwarded = _PER_CALL_CASES[case](stream)
    _with_timeout(adapter)
    _run(adapter, stream)
    assert forwarded() == pytest.approx(_TIMEOUT)


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "stream"])
@pytest.mark.parametrize("case", sorted(_PER_CALL_CASES))
def test_the_deadline_clamped_timeout_reaches_the_sdk_call(
    case: str, stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import intergrax.runtime.execution.deadline_provider_guard as guard

    monkeypatch.setattr(guard, "resolve_active_provider_timeout_seconds", lambda configured: 2.0)
    adapter, forwarded = _PER_CALL_CASES[case](stream)
    _with_timeout(adapter, None)
    _run(adapter, stream)
    assert forwarded() == pytest.approx(2.0)


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "stream"])
def test_native_ollama_forwards_the_timeout_through_a_per_call_client(
    stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, forwarded = _native_ollama_owned_client(stream, monkeypatch)
    _with_timeout(adapter)
    _run(adapter, stream)
    assert forwarded() == pytest.approx(_TIMEOUT)


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "stream"])
@pytest.mark.parametrize("case", sorted(_PER_CALL_CASES))
def test_no_timeout_argument_is_sent_when_no_budget_is_set(case: str, stream: bool) -> None:
    adapter, forwarded = _PER_CALL_CASES[case](stream)
    _with_timeout(adapter, None)
    _run(adapter, stream)
    assert forwarded() is None


def _bedrock() -> tuple[Any, MagicMock]:
    from intergrax.llm_adapters.providers.aws_bedrock_adapter import BedrockChatAdapter

    converse = MagicMock(return_value={"output": {"message": {"content": [{"text": "x"}]}}, "stopReason": "end_turn"})
    client = SimpleNamespace(converse=converse, converse_stream=MagicMock())
    adapter = BedrockChatAdapter(
        client=client, model_id="anthropic.claude-3-haiku-20240307-v1:0", region="us-east-1", use_converse=True
    )
    return adapter, converse


def _langchain_ollama() -> tuple[Any, MagicMock]:
    pytest.importorskip("langchain_core", reason="optional extra llm-langchain-ollama")
    from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter
    from intergrax.llm_adapters.providers.ollama_capabilities import OllamaModelCapabilityResolver

    chat = MagicMock()
    chat.model = "qwen2.5:14b"
    chat.invoke.return_value = SimpleNamespace(content="x", response_metadata={"done_reason": "stop"})
    resolver = OllamaModelCapabilityResolver(show_model=lambda _m: SimpleNamespace(capabilities=["completion"]))
    adapter = LangChainOllamaAdapter(chat=chat, model="qwen2.5:14b", capability_resolver=resolver)
    return adapter, chat.invoke


def _native_ollama_injected_client() -> tuple[Any, MagicMock]:
    from intergrax.llm_adapters.providers.native_ollama_adapter import NativeOllamaAdapter
    from intergrax.llm_adapters.providers.ollama_capabilities import OllamaModelCapabilityResolver

    client = MagicMock()
    client.chat.return_value = {"message": {"content": "x"}, "done": True, "done_reason": "stop"}
    resolver = OllamaModelCapabilityResolver(show_model=lambda _m: SimpleNamespace(capabilities=["completion"]))
    return NativeOllamaAdapter(client=client, model="qwen2.5:14b", capability_resolver=resolver), client.chat


_NONE_CASES: dict[str, Callable[[], tuple[Any, MagicMock]]] = {
    "bedrock": _bedrock,
    "langchain-ollama": _langchain_ollama,
    "native-ollama-injected-client": _native_ollama_injected_client,
}


@pytest.mark.parametrize("case", sorted(_NONE_CASES))
def test_an_adapter_that_cannot_forward_fails_closed_before_the_sdk(case: str) -> None:
    adapter, sdk = _NONE_CASES[case]()
    assert adapter.supports_call_timeout() is False
    _with_timeout(adapter)
    with pytest.raises(LLMCapabilityMismatchError):
        adapter.generate_messages(_USER)
    assert sdk.call_count == 0


@pytest.mark.parametrize("case", sorted(_NONE_CASES))
def test_an_adapter_that_cannot_forward_still_runs_without_a_budget(case: str) -> None:
    adapter, sdk = _NONE_CASES[case]()
    _with_timeout(adapter, None)
    assert adapter.generate_messages(_USER).content == "x"
    assert sdk.call_count == 1


def test_a_hung_sdk_call_that_honours_the_forwarded_timeout_surfaces_a_typed_timeout_error() -> None:
    spec = importlib.util.find_spec("intergrax.llm_adapters.contracts.provider_error")
    assert spec is not None, "typed provider-error contract v1 is missing"
    provider_error = importlib.import_module("intergrax.llm_adapters.contracts.provider_error")
    from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
    from intergrax.llm_adapters.providers.openai_chat_completions_adapter import OpenAIChatCompletionsAdapter

    class APITimeoutError(Exception):
        """Shape of the OpenAI SDK timeout error (no status code)."""

    def create(**kwargs: Any) -> Any:
        if "timeout" not in kwargs:
            raise AssertionError("the SDK call received no timeout; a real SDK would hang here")
        time.sleep(min(float(kwargs["timeout"]), 0.05))
        raise APITimeoutError("Request timed out.")

    client = MagicMock()
    client.chat.completions.create.side_effect = create
    adapter = _with_timeout(
        OpenAIChatCompletionsAdapter(client=client, model="gpt-4o-mini", provider=LLMProvider.OPENAI), 0.05
    )
    started = time.monotonic()
    with pytest.raises(provider_error.LLMProviderTimeoutError) as info:
        adapter.generate_messages(_USER)
    assert time.monotonic() - started < 2.0
    assert isinstance(info.value.__cause__, APITimeoutError)


def test_every_shipped_adapter_has_a_declared_timeout_mechanism() -> None:
    """Closed world: every direct ``BaseLLMAdapter`` subclass in the provider package is listed above."""
    found: set[str] = set()
    for path in sorted(_PROVIDERS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and any(
                isinstance(base, ast.Name) and base.id == "BaseLLMAdapter" for base in node.bases
            ):
                found.add(node.name)
    assert found == set(_MECHANISMS)


def test_the_capability_flag_matches_the_declared_mechanism() -> None:
    for name, mechanism in _MECHANISMS.items():
        source = next(
            path.read_text(encoding="utf-8")
            for path in _PROVIDERS.glob("*.py")
            if f"class {name}(" in path.read_text(encoding="utf-8")
        )
        declares_false = "def supports_call_timeout" in source
        if mechanism == "NONE":
            assert declares_false, f"{name} cannot forward a timeout and must declare supports_call_timeout()"
        elif mechanism == "PER_CALL_DERIVED_CLIENT":
            assert declares_false, f"{name} must declare when it can forward (adapter-owned client only)"
