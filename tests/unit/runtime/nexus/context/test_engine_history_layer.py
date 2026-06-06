# © Artur Czarnecki. All rights reserved.

"""MEM-5.1: engine_history_layer SUMMARIZE_OLDEST + truncate fallback."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.engine_history_layer import HistoryLayer
from intergrax.runtime.nexus.prompts.history_prompt_builder import HistorySummaryPromptBundle
from intergrax.runtime.nexus.responses.response_schema import (
    HistoryCompressionStrategy,
    RuntimeRequest,
)
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@dataclass
class _StubHistoryPromptBuilder:
    def build_history_summary_prompt(self, **_kwargs) -> HistorySummaryPromptBundle:
        return HistorySummaryPromptBundle(system_prompt="Summarize older turns briefly.")


class _EmptySummaryLLMAdapter(FakeLLMAdapter):
    """Deterministic adapter that fails summarization by returning empty text."""

    def generate_messages(self, messages, *, temperature=None, max_tokens=None, run_id=None):
        _ = (messages, temperature, max_tokens, run_id)
        return build_adapter_response(content="")


def _history_layer(llm: LLMAdapter) -> HistoryLayer:
    config = RuntimeConfig(llm_adapter=llm, production_mode=False)
    return HistoryLayer(
        config=config,
        session_manager=build_in_memory_session_manager(),
        history_prompt_builder=_StubHistoryPromptBuilder(),
    )


def _long_history(turns: int = 12) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    for index in range(turns):
        role = "user" if index % 2 == 0 else "assistant"
        messages.append(
            ChatMessage(
                role=role,
                content=f"Turn {index}: " + ("x" * 180),
            )
        )
    return messages


def _runtime_request() -> RuntimeRequest:
    return RuntimeRequest(
        tenant_id="tenant-hist",
        agent_id="agent_hist",
        user_id="user_hist",
        session_id="sess_hist",
        message="history compression probe",
        history_compression_strategy=HistoryCompressionStrategy.SUMMARIZE_OLDEST,
    )


def test_summarize_oldest_uses_fake_llm_summary() -> None:
    layer = _history_layer(FakeLLMAdapter(fixed_text="Older discussion about memory wiring."))
    history = _long_history()
    raw_token_count = layer._count_tokens_for_messages(history)
    assert raw_token_count is not None
    assert raw_token_count > 120

    result = layer._compress_history(
        request=_runtime_request(),
        raw_history=history,
        raw_token_count=raw_token_count,
        strategy=HistoryCompressionStrategy.SUMMARIZE_OLDEST,
        history_budget_tokens=120,
        run_id="run-hist-1",
    )

    assert result.truncated is True
    assert result.summary_used is True
    assert result.effective_strategy == HistoryCompressionStrategy.SUMMARIZE_OLDEST
    assert result.history[0].role == "system"
    assert "Conversation summary" in result.history[0].content
    assert "Older discussion about memory wiring." in result.history[0].content
    assert len(result.history) > 1


def test_summarize_oldest_falls_back_to_truncate_when_summary_fails() -> None:
    layer = _history_layer(_EmptySummaryLLMAdapter())
    history = _long_history()
    raw_token_count = layer._count_tokens_for_messages(history)
    assert raw_token_count is not None

    result = layer._compress_history(
        request=_runtime_request(),
        raw_history=history,
        raw_token_count=raw_token_count,
        strategy=HistoryCompressionStrategy.SUMMARIZE_OLDEST,
        history_budget_tokens=120,
        run_id="run-hist-2",
    )

    assert result.truncated is True
    assert result.summary_used is False
    assert result.effective_strategy == HistoryCompressionStrategy.TRUNCATE_OLDEST
    assert all(msg.role in {"user", "assistant"} for msg in result.history)
    assert layer._count_tokens_for_messages(result.history) is not None
    assert layer._count_tokens_for_messages(result.history) <= 120
