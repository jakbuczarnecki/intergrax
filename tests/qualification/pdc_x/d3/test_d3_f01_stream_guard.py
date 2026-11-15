# © Artur Czarnecki. All rights reserved.

"""PDC-D3 F-D3-01 — every shipped stream path runs the provider guard, quota and cancel registration.

The streaming methods of the default OpenAI (Responses), Gemini / Vertex and LangChain Ollama
adapters used to wrap the SDK stream only in ``_provider_dependency_attempt()``. They bypassed the
protected-work guard, the tenant quota, rate limit / circuit breaker / retry and the stream transport
registration while the registry declared a cancel capability for them. These tests drive the real
``stream_messages`` / ``stream_with_tools`` methods with fake SDK objects (no network, no credential).
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.provider_stream_transport_registry import ProviderStreamTransportRegistry
from intergrax.runtime.external_operations.llm_external_operation_attempt import LlmExternalOperationAttempt

pytestmark = pytest.mark.unit

_USER = [ChatMessage(role="user", content="hello")]
_NOOP_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "noop",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}
_OP_ID = "op-d3-f01"


class _Denied(RuntimeError):
    """Raised by the patched protected-work guard."""


class _SdkStream:
    """Fake SDK stream: iterable, closable, records consumption."""

    def __init__(self, items: Iterable[Any]) -> None:
        self._items = list(items)
        self.closed = False
        self.iterated = 0
        self.error: BaseException | None = None

    def __iter__(self) -> Iterator[Any]:
        for item in self._items:
            self.iterated += 1
            yield item
        if self.error is not None:
            raise self.error

    def close(self) -> None:
        self.closed = True

    # Context-manager shape of ``client.responses.stream(...)``.
    def __enter__(self) -> _SdkStream:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def get_final_response(self) -> Any:
        response = MagicMock()
        response.output_text = "".join(self._items_text())
        response.output = []
        response.status = "completed"
        response.usage = MagicMock(input_tokens=3, output_tokens=2)
        return response

    def _items_text(self) -> list[str]:
        return [getattr(item, "delta", "") or "" for item in self._items]


class _Boundary:
    """Counts permit acquire / release pairs."""

    def __init__(self) -> None:
        self.acquired = 0
        self.released = 0

    def acquire(self, _request: object) -> object:
        self.acquired += 1
        return object()

    def complete_direct(self, _handle: object) -> None:
        self.released += 1


class _Probe:
    def __init__(self, stream: _SdkStream | None = None) -> None:
        self.stream = stream
        self.sdk_calls = 0


def _openai_responses(items: list[Any] | None = None) -> tuple[Any, _Probe]:
    from intergrax.llm_adapters.providers.openai_responses_adapter import OpenAIChatResponsesAdapter

    events = items if items is not None else [SimpleNamespace(type="response.output_text.delta", delta="hi")]
    probe = _Probe(_SdkStream(events))
    client = MagicMock()

    def _stream(**_: Any) -> _SdkStream:
        probe.sdk_calls += 1
        assert probe.stream is not None
        return probe.stream

    client.responses.stream.side_effect = _stream
    return OpenAIChatResponsesAdapter(client=client, model="gpt-5.1"), probe


def _openai_chat(items: list[Any] | None = None) -> tuple[Any, _Probe]:
    from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
    from intergrax.llm_adapters.providers.openai_chat_completions_adapter import OpenAIChatCompletionsAdapter

    chunk = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content="hi", refusal=None), finish_reason=None)]
    )
    probe = _Probe(_SdkStream(items if items is not None else [chunk]))
    client = MagicMock()

    def _create(**_: Any) -> _SdkStream:
        probe.sdk_calls += 1
        assert probe.stream is not None
        return probe.stream

    client.chat.completions.create.side_effect = _create
    return OpenAIChatCompletionsAdapter(client=client, model="gpt-4o-mini", provider=LLMProvider.OPENAI), probe


def _gemini(items: list[Any] | None = None) -> tuple[Any, _Probe]:
    pytest.importorskip("google.genai")
    from intergrax.llm_adapters.providers.gemini_adapter import GeminiChatAdapter

    probe = _Probe(_SdkStream(items if items is not None else [SimpleNamespace(text="hi")]))
    client = MagicMock()

    def _send(_prompt: str) -> _SdkStream:
        probe.sdk_calls += 1
        assert probe.stream is not None
        return probe.stream

    client.chats.create.return_value.send_message_stream.side_effect = _send
    return GeminiChatAdapter(client=client, model="gemini-2.5-flash"), probe


def _langchain_ollama(items: list[Any] | None = None) -> tuple[Any, _Probe]:
    pytest.importorskip("langchain_core")
    from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter
    from intergrax.llm_adapters.providers.ollama_capabilities import OllamaModelCapabilityResolver

    probe = _Probe(_SdkStream(items if items is not None else [SimpleNamespace(content="hi")]))
    chat = MagicMock()
    chat.model = "qwen2.5:14b"

    def _stream(*_: Any, **__: Any) -> _SdkStream:
        probe.sdk_calls += 1
        assert probe.stream is not None
        return probe.stream

    chat.stream.side_effect = _stream
    resolver = OllamaModelCapabilityResolver(show_model=lambda _m: SimpleNamespace(capabilities=["completion"]))
    adapter = LangChainOllamaAdapter(chat=chat, model="qwen2.5:14b", capability_resolver=resolver)
    return adapter, probe


_FIXED: dict[str, Callable[..., tuple[Any, _Probe]]] = {
    "openai-responses": _openai_responses,
    "gemini": _gemini,
    "langchain-ollama": _langchain_ollama,
}
_ALL_STREAM_MESSAGES: dict[str, Callable[..., tuple[Any, _Probe]]] = {**_FIXED, "openai-chat": _openai_chat}


@pytest.fixture()
def operation_id(monkeypatch: pytest.MonkeyPatch) -> str:
    """Give the (store-less) external operation attempt a stable operation id."""
    monkeypatch.setattr(LlmExternalOperationAttempt, "operation_id", property(lambda _self: _OP_ID))
    return _OP_ID


@pytest.fixture()
def deny_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    import intergrax.runtime.execution.deadline_provider_guard as guard

    def _deny() -> None:
        raise _Denied("protected work denied")

    monkeypatch.setattr(guard, "assert_protected_provider_call_allowed", _deny)


@pytest.fixture()
def quota_calls(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    import intergrax.llm_adapters.governance.quota as quota

    calls: list[str | None] = []
    monkeypatch.setattr(quota, "check_llm_tenant_quota", lambda tenant_id: calls.append(tenant_id))
    return calls


@pytest.mark.parametrize("name", sorted(_ALL_STREAM_MESSAGES))
def test_stream_messages_is_denied_by_the_protected_work_guard_before_the_sdk(name: str, deny_guard: None) -> None:
    adapter, probe = _ALL_STREAM_MESSAGES[name]()
    with pytest.raises(_Denied):
        list(adapter.stream_messages(_USER))
    assert probe.sdk_calls == 0, "the guard must run before any SDK call"


@pytest.mark.parametrize("name", sorted(_ALL_STREAM_MESSAGES))
def test_stream_messages_consults_the_tenant_quota(name: str, quota_calls: list[str | None]) -> None:
    adapter, _ = _ALL_STREAM_MESSAGES[name]()
    list(adapter.stream_messages(_USER))
    assert len(quota_calls) == 1


@pytest.mark.parametrize("name", sorted(_ALL_STREAM_MESSAGES))
def test_stream_messages_registers_and_closes_the_stream_transport(name: str, operation_id: str) -> None:
    adapter, probe = _ALL_STREAM_MESSAGES[name]()
    registry = ProviderStreamTransportRegistry()
    adapter._external_operation_stream_registry = registry

    events = iter(adapter.stream_messages(_USER))
    next(events)
    assert operation_id in registry._closers, "an in-flight stream must be registered with the cancel seam"
    assert registry.close_transport(operation_id) is True
    assert probe.stream is not None and probe.stream.closed is True, "terminating must close the SDK stream"

    # Exhausting a fresh stream unregisters the transport and closes it.
    adapter, probe = _ALL_STREAM_MESSAGES[name]()
    adapter._external_operation_stream_registry = registry
    list(adapter.stream_messages(_USER))
    assert operation_id not in registry._closers
    assert probe.stream is not None and probe.stream.closed is True


@pytest.mark.parametrize("name", sorted(_ALL_STREAM_MESSAGES))
def test_stream_messages_releases_the_provider_permit_exactly_once(name: str) -> None:
    adapter, _ = _ALL_STREAM_MESSAGES[name]()
    boundary = _Boundary()
    adapter._provider_dependency_boundary = boundary
    list(adapter.stream_messages(_USER))
    assert (boundary.acquired, boundary.released) == (1, 1)


@pytest.mark.parametrize("name", sorted(_ALL_STREAM_MESSAGES))
def test_stream_messages_releases_the_permit_when_the_sdk_stream_fails_mid_way(name: str) -> None:
    class _Boom(RuntimeError):
        pass

    adapter, probe = _ALL_STREAM_MESSAGES[name]()
    boundary = _Boundary()
    adapter._provider_dependency_boundary = boundary
    assert probe.stream is not None
    probe.stream.error = _Boom("transport failed")
    with pytest.raises(_Boom):
        list(adapter.stream_messages(_USER))
    assert (boundary.acquired, boundary.released) == (1, 1)


@pytest.mark.parametrize("name", sorted(_ALL_STREAM_MESSAGES))
def test_stream_messages_releases_the_permit_when_the_consumer_closes_early(name: str) -> None:
    adapter, _ = _ALL_STREAM_MESSAGES[name]()
    boundary = _Boundary()
    adapter._provider_dependency_boundary = boundary
    events = adapter.stream_messages(_USER)
    next(iter(events))
    events.close()  # type: ignore[attr-defined]
    assert (boundary.acquired, boundary.released) == (1, 1)


def test_stream_with_tools_of_openai_responses_is_guarded_and_registered(
    operation_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import intergrax.runtime.execution.deadline_provider_guard as guard

    adapter, probe = _openai_responses()
    registry = ProviderStreamTransportRegistry()
    adapter._external_operation_stream_registry = registry
    events = iter(adapter.stream_with_tools(_USER, [_NOOP_TOOL]))
    next(events)
    assert operation_id in registry._closers
    list(events)
    assert operation_id not in registry._closers

    denied, denied_probe = _openai_responses()
    monkeypatch.setattr(guard, "assert_protected_provider_call_allowed", lambda: (_ for _ in ()).throw(_Denied()))
    with pytest.raises(_Denied):
        list(denied.stream_with_tools(_USER, [_NOOP_TOOL]))
    assert denied_probe.sdk_calls == 0


def test_stream_with_tools_of_gemini_is_guarded_and_registered(
    operation_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import intergrax.runtime.execution.deadline_provider_guard as guard

    pytest.importorskip("google.genai")
    from intergrax.llm_adapters.providers.gemini_adapter import GeminiChatAdapter

    def _build() -> tuple[Any, _Probe]:
        probe = _Probe(_SdkStream([SimpleNamespace(text="hi", candidates=[], function_calls=None)]))
        client = MagicMock()

        def _gen(**_: Any) -> _SdkStream:
            probe.sdk_calls += 1
            assert probe.stream is not None
            return probe.stream

        client.models.generate_content_stream.side_effect = _gen
        return GeminiChatAdapter(client=client, model="gemini-2.5-flash"), probe

    adapter, probe = _build()
    registry = ProviderStreamTransportRegistry()
    adapter._external_operation_stream_registry = registry
    events = iter(adapter.stream_with_tools(_USER, [_NOOP_TOOL]))
    next(events)
    assert operation_id in registry._closers
    list(events)
    assert operation_id not in registry._closers

    denied, denied_probe = _build()
    monkeypatch.setattr(guard, "assert_protected_provider_call_allowed", lambda: (_ for _ in ()).throw(_Denied()))
    with pytest.raises(_Denied):
        list(denied.stream_with_tools(_USER, [_NOOP_TOOL]))
    assert denied_probe.sdk_calls == 0


# ---------------------------------------------------------------------------------------------
# Source gate: no provider stream method may reach the SDK outside ``_execute_streaming``.
# ---------------------------------------------------------------------------------------------

_PROVIDERS = Path(__file__).resolve().parents[4] / "intergrax" / "llm_adapters" / "providers"
#: Wrapper / catalog classes delegate to an inner adapter and own no SDK call.
_STREAM_METHODS = {"stream_messages", "stream_with_tools"}


def _violations(source: str) -> list[str]:
    """Stream methods that never use ``_execute_streaming`` or that use the bare permit helper."""
    problems: list[str] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in _STREAM_METHODS:
            attrs = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
            if "_provider_dependency_attempt" in attrs:
                problems.append(f"{node.name}:bare_permit_helper")
            if "_execute_streaming" not in attrs and not _delegates_to_super(node):
                problems.append(f"{node.name}:no_execute_streaming")
    return problems


def _delegates_to_super(node: ast.FunctionDef) -> bool:
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Call)
        and isinstance(n.func.value.func, ast.Name)
        and n.func.value.func.id == "super"
        for n in ast.walk(node)
    )


def test_source_gate_flags_a_stream_method_that_bypasses_execute_streaming() -> None:
    bypass = """
class A:
    def stream_messages(self, messages):
        with self._provider_dependency_attempt():
            for chunk in self.client.stream():
                yield chunk
"""
    assert _violations(bypass) == ["stream_messages:bare_permit_helper", "stream_messages:no_execute_streaming"]
    compliant = """
class B:
    def stream_messages(self, messages):
        for chunk in self._execute_streaming(lambda: self.client.stream()):
            yield chunk
"""
    assert _violations(compliant) == []


@pytest.mark.parametrize("path", sorted(_PROVIDERS.glob("*_adapter.py")), ids=lambda p: p.name)
def test_every_provider_stream_method_goes_through_execute_streaming(path: Path) -> None:
    assert _violations(path.read_text(encoding="utf-8")) == []
