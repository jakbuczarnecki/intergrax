# © Artur Czarnecki. All rights reserved.

"""PDC-D3 F-D3-08 — LangChain Ollama streaming never silently re-issues the model call after output."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEventKind

pytest.importorskip("langchain_core")

pytestmark = pytest.mark.unit

_USER = [ChatMessage(role="user", content="hello")]


def _adapter(chat: MagicMock) -> Any:
    from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter
    from intergrax.llm_adapters.providers.ollama_capabilities import OllamaModelCapabilityResolver

    chat.model = "qwen2.5:14b"
    resolver = OllamaModelCapabilityResolver(show_model=lambda _model: SimpleNamespace(capabilities=["completion"]))
    return LangChainOllamaAdapter(chat=chat, model="qwen2.5:14b", capability_resolver=resolver)


def test_error_after_partial_output_is_raised_and_the_call_is_not_reissued() -> None:
    def _stream(*_: Any, **__: Any) -> Iterator[Any]:
        yield SimpleNamespace(content="par")
        yield SimpleNamespace(content="tial")
        raise ConnectionError("transport closed")

    chat = MagicMock()
    chat.stream.side_effect = _stream
    chat.invoke.return_value = SimpleNamespace(content="FULL REISSUED ANSWER")
    adapter = _adapter(chat)

    seen: list[str] = []
    with pytest.raises(ConnectionError):
        for event in adapter.stream_messages(_USER):
            seen.append(event.delta_content)

    assert seen == ["par", "tial"]
    chat.invoke.assert_not_called()


def test_failure_before_any_chunk_falls_back_to_one_buffered_call() -> None:
    chat = MagicMock()
    chat.stream.side_effect = ConnectionError("cannot connect")
    chat.invoke.return_value = SimpleNamespace(content="buffered answer")
    adapter = _adapter(chat)

    events = list(adapter.stream_messages(_USER))

    chat.invoke.assert_called_once()
    final = events[-1]
    assert final.kind is LLMStreamEventKind.FINAL
    assert final.response is not None and final.response.content == "buffered answer"


def test_policy_denial_is_not_swallowed_by_the_buffered_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import intergrax.llm_adapters.governance.quota as quota

    def _deny(_tenant: str | None, **_: Any) -> None:
        raise quota.LLMQuotaExceeded("over budget")

    monkeypatch.setattr(quota, "check_llm_tenant_quota", _deny)
    chat = MagicMock()
    chat.invoke.return_value = SimpleNamespace(content="must not be reached")
    adapter = _adapter(chat)

    with pytest.raises(quota.LLMQuotaExceeded):
        list(adapter.stream_messages(_USER))

    chat.stream.assert_not_called()
    chat.invoke.assert_not_called()
