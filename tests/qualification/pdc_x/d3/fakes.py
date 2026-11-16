# © Artur Czarnecki. All rights reserved.

"""Deterministic fake LLM providers for the B-D3 qualification tests (no SDK, no network)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent, LLMStreamEventKind

NOOP_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "noop",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}
USER = [ChatMessage(role="user", content="hello")]


def _final(text: str, provider: str, model: str) -> LLMStreamEvent:
    return LLMStreamEvent(
        kind=LLMStreamEventKind.FINAL,
        response=LLMAdapterResponse(content=text, provider=provider, model=model),
    )


class _FakeBase(BaseLLMAdapter):
    provider = "fake"
    model = "fake-model"

    def __init__(self, *, provider: str = "fake", model: str = "fake-model") -> None:
        super().__init__()
        self.provider = provider
        self.model = model
        self.calls: list[str] = []

    @property
    def context_window_tokens(self) -> int:
        return 8_192

    def generate_messages(self, messages: Sequence[ChatMessage], **_: Any) -> LLMAdapterResponse:
        self.calls.append("generate_messages")
        return LLMAdapterResponse(content="buffered", provider=self.provider, model=self.model)


class BufferedOnlyAdapter(_FakeBase):
    """Declares nothing: generate_messages only."""


class BufferedToolsAdapter(_FakeBase):
    """Tools through buffered calls only; tool streaming deliberately not provided."""

    def supports_tools(self) -> bool:
        return True

    def generate_with_tools(self, messages: Sequence[ChatMessage], tools: Any, **_: Any) -> LLMAdapterResponse:
        self.calls.append("generate_with_tools")
        return LLMAdapterResponse(content="buffered-tools", provider=self.provider, model=self.model)


class StreamingToolsAdapter(BufferedToolsAdapter):
    """Honest full declaration: streaming, tools and tool streaming."""

    def supports_streaming(self) -> bool:
        return True

    def supports_tool_streaming(self) -> bool:
        return True

    def stream_messages(self, messages: Sequence[ChatMessage], **_: Any) -> Iterable[LLMStreamEvent]:
        self.calls.append("stream_messages")
        yield LLMStreamEvent(kind=LLMStreamEventKind.PARTIAL, delta_content="a")
        yield _final("a", self.provider, self.model)

    def stream_with_tools(self, messages: Sequence[ChatMessage], tools: Any, **_: Any) -> Iterable[LLMStreamEvent]:
        self.calls.append("stream_with_tools")
        yield LLMStreamEvent(kind=LLMStreamEventKind.PARTIAL, delta_content="t")
        yield _final("t", self.provider, self.model)


class StreamingOnlyAdapter(BufferedToolsAdapter):
    """Streams text and supports tools, but declares no tool streaming (declared, not forced)."""

    def supports_streaming(self) -> bool:
        return True

    def stream_messages(self, messages: Sequence[ChatMessage], **_: Any) -> Iterable[LLMStreamEvent]:
        self.calls.append("stream_messages")
        yield _final("s", self.provider, self.model)


class UnderReportingStreamAdapter(_FakeBase):
    """The pre-B-D3 defect: implements stream_messages while supports_streaming() stays False."""

    def stream_messages(self, messages: Sequence[ChatMessage], **_: Any) -> Iterable[LLMStreamEvent]:
        yield _final("x", self.provider, self.model)


class OverReportingToolStreamAdapter(BufferedToolsAdapter):
    """Declares tool streaming without implementing stream_with_tools."""

    def supports_streaming(self) -> bool:
        return True

    def supports_tool_streaming(self) -> bool:
        return True

    def stream_messages(self, messages: Sequence[ChatMessage], **_: Any) -> Iterable[LLMStreamEvent]:
        yield _final("x", self.provider, self.model)


class FailingAdapter(_FakeBase):
    """Every buffered call raises; counts the attempts."""

    def __init__(self, error: BaseException, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._error = error

    def generate_messages(self, messages: Sequence[ChatMessage], **_: Any) -> LLMAdapterResponse:
        self.calls.append("generate_messages")
        raise self._error


class AlwaysTryNextPolicy:
    """Failover policy that always asks for the next candidate (worst case for boundedness)."""

    def decide_after_failure(self, context: Any) -> Any:
        from intergrax.llm_adapters.contracts.failover_policy import FailoverDecision

        return FailoverDecision.TRY_NEXT
