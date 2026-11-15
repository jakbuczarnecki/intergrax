# © Artur Czarnecki. All rights reserved.

"""PDC-D3 F-D3-13 (mutant M03) — the failover pre-check never selects an under-reporting member.

A member that implements streaming while declaring ``supports_streaming() is False`` must not be used as
a fallback when no chain member declares the feature: the typed capability error is raised before any
provider call (the base class raising later is not the contract, the pre-check is).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import pytest
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.capabilities import LLMCapabilityMismatchError, LLMFeature
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent
from intergrax.llm_adapters.registry.failover_adapter import FailoverLLMAdapter
from intergrax.llm_adapters.registry.failover_policy import default_failover_policy
from tests.qualification.pdc_x.d3.fakes import (
    NOOP_TOOL,
    USER,
    BufferedOnlyAdapter,
    BufferedToolsAdapter,
    StreamingToolsAdapter,
    UnderReportingStreamAdapter,
)

pytestmark = pytest.mark.unit


class _RecordingUnderReporter(UnderReportingStreamAdapter):
    """Streams although it declares nothing; records the attempt."""

    def stream_messages(self, messages: Sequence[ChatMessage], **_: Any) -> Iterable[LLMStreamEvent]:
        self.calls.append("stream_messages")
        yield from super().stream_messages(messages)


class _ToolUnderReporter(BufferedToolsAdapter):
    """Implements stream_with_tools while tool streaming is not declared."""

    def stream_with_tools(self, messages: Sequence[ChatMessage], tools: Any, **_: Any) -> Iterable[LLMStreamEvent]:
        self.calls.append("stream_with_tools")
        yield from StreamingToolsAdapter.stream_with_tools(self, messages, tools)  # type: ignore[arg-type]


def test_stream_messages_fails_closed_when_no_member_declares_streaming() -> None:
    reporter = _RecordingUnderReporter(provider="liar")
    chain = FailoverLLMAdapter(
        [BufferedOnlyAdapter(provider="plain"), reporter], failover_policy=default_failover_policy()
    )

    with pytest.raises(LLMCapabilityMismatchError) as caught:
        list(chain.stream_messages(USER))

    assert caught.value.features == (LLMFeature.STREAMING,)
    assert reporter.calls == [], "an under-reporting member must not be selected as a fallback"


def test_stream_messages_selects_the_declaring_member_not_the_first_member() -> None:
    first = _RecordingUnderReporter(provider="liar")
    honest = StreamingToolsAdapter(provider="honest")
    chain = FailoverLLMAdapter([first, honest], failover_policy=default_failover_policy())

    list(chain.stream_messages(USER))

    assert first.calls == []
    assert honest.calls == ["stream_messages"]


def test_stream_with_tools_fails_closed_when_no_member_declares_tool_streaming() -> None:
    reporter = _ToolUnderReporter(provider="liar")
    chain = FailoverLLMAdapter([reporter], failover_policy=default_failover_policy())

    with pytest.raises(LLMCapabilityMismatchError) as caught:
        list(chain.stream_with_tools(USER, [NOOP_TOOL]))  # type: ignore[list-item]

    assert caught.value.features == (LLMFeature.TOOL_STREAMING,)
    assert reporter.calls == []
