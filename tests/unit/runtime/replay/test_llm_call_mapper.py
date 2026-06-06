# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.finish_reason import LLMFinishReason
from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage
from intergrax.runtime.replay.llm_call_mapper import llm_call_info_from_adapter_response


@pytest.mark.gate
def test_llm_call_info_from_adapter_response() -> None:
    resp = build_adapter_response(
        content="ok",
        finish_reason=LLMFinishReason.LENGTH,
        usage=LLMTokenUsage.from_counts(input_tokens=3, output_tokens=7),
        model="claude-test",
        provider="claude",
        response_id="rid-1",
    )
    info = llm_call_info_from_adapter_response(resp, step_id="core_llm")
    assert info.step_id == "core_llm"
    assert info.model == "claude-test"
    assert info.prompt_tokens == 3
    assert info.completion_tokens == 7
    assert info.total_tokens == 10
    assert info.finish_reason == "length"
    assert info.response_payload is resp
