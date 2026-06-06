# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.llm_adapters._shared.adapter_response_builders import (
    build_adapter_response,
    final_stream_event,
    partial_stream_event,
)
from intergrax.llm_adapters.contracts.finish_reason import LLMFinishReason, parse_finish_reason
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEventKind
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall


@pytest.mark.gate
def test_build_adapter_response_fields() -> None:
    usage = LLMTokenUsage.from_counts(input_tokens=10, output_tokens=5)
    tc = LLMToolCall.from_openai_shape(call_id="c1", name="noop", arguments='{"x": 1}')
    resp = build_adapter_response(
        content="hello",
        finish_reason=LLMFinishReason.COMPLETED,
        usage=usage,
        model="gpt-test",
        provider="openai",
        response_id="resp-1",
        tool_calls=(tc,),
    )
    assert resp.content == "hello"
    assert resp.text == "hello"
    assert resp.usage is not None
    assert resp.usage.total_tokens == 15
    assert resp.has_tool_calls
    assert resp.tool_calls[0].name == "noop"


@pytest.mark.gate
def test_parse_finish_reason_aliases() -> None:
    assert parse_finish_reason("stop") == LLMFinishReason.COMPLETED
    assert parse_finish_reason("tool_calls") == LLMFinishReason.TOOL_CALLS
    assert parse_finish_reason("weird") == LLMFinishReason.UNKNOWN


@pytest.mark.gate
def test_stream_events() -> None:
    partial = partial_stream_event(delta_content="hi")
    assert partial.kind == LLMStreamEventKind.PARTIAL
    final = final_stream_event(response=build_adapter_response(content="hi"))
    assert final.is_final
    assert final.response is not None


@pytest.mark.gate
def test_structured_result() -> None:
    payload = {"a": 1}
    resp = build_adapter_response(content='{"a": 1}')
    result = LLMStructuredResult(parsed=payload, response=resp)
    assert result.parsed["a"] == 1
    assert result.response.content.startswith("{")
