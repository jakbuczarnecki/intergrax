# © Artur Czarnecki. All rights reserved.

"""MEM-DEPTH-1.* Context Compiler unit tests."""

from __future__ import annotations

import pytest

from intergrax.applications.contracts.environment_profile import ContextDecisionProfile
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_budget import trim_message_to_budget_tokenizer_aware, ContextBudgetPolicy
from intergrax.runtime.nexus.context.context_compiler import ContextCompiler, classify_candidates
from intergrax.runtime.nexus.context.context_compiler_models import ContextCandidateSource
from intergrax.runtime.nexus.context.context_preflight import verify_context_preflight
from intergrax.runtime.nexus.context.degradation_ladder import DegradationStepKind
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm.messages import ChatMessage as LlmChatMessage

pytestmark = pytest.mark.gate


class _SmallWindowAdapter(LLMAdapter):
    provider = "fake"
    model = "fake-small"

    def __init__(self, window: int = 512) -> None:
        super().__init__()
        self._window = window

    @property
    def context_window_tokens(self) -> int:
        return self._window

    def generate_messages(self, messages, **kwargs) -> LLMAdapterResponse:
        _ = messages, kwargs
        return LLMAdapterResponse(content="ok")


def test_classify_candidates_marks_injections() -> None:
    messages = [
        ChatMessage(role="system", content="Core instructions"),
        ChatMessage(role="system", content="RAG CONTEXT:\n" + ("doc " * 100)),
        ChatMessage(role="user", content="question"),
    ]
    candidates = classify_candidates(messages, count_tokens=lambda t: len(t) // 4)
    assert len(candidates) == 3
    assert candidates[-1].mandatory is True


def test_classify_candidates_uses_ce_context_tags() -> None:
    messages = [
        ChatMessage(role="system", content="Core instructions"),
        ChatMessage(
            role="system",
            content="[context:rag:doc-1] Retrieved policy paragraph.",
        ),
        ChatMessage(role="user", content="question"),
    ]
    candidates = classify_candidates(messages, count_tokens=lambda t: len(t) // 4)
    assert candidates[1].source == ContextCandidateSource.RAG


def test_context_compiler_trims_oversized_context() -> None:
    adapter = _SmallWindowAdapter(window=512)
    config = RuntimeConfig(
        llm_adapter=adapter,
        context_decision_profile=ContextDecisionProfile(
            include_session_history=True,
            prefer_longterm_memory=True,
            prefer_rag_when_enabled=False,
        ).model_dump(mode="json"),
    )
    huge = "x" * 20_000
    messages = [
        ChatMessage(role="system", content="Instructions"),
        ChatMessage(role="system", content=f"WEBSEARCH:\n{huge}"),
        ChatMessage(role="user", content="hi"),
    ]
    compiler = ContextCompiler()
    result = compiler.compile(messages, config, max_output_tokens=64)
    assert result.trimmed is True
    assert DegradationStepKind.DROP_OPTIONAL_INJECTIONS.value in result.degradation_steps or result.total_tokens <= result.budget_tokens


def test_tokenizer_aware_trim() -> None:
    policy = ContextBudgetPolicy(max_chars=10_000, max_tokens_estimate=10)
    result = trim_message_to_budget_tokenizer_aware("a" * 500, policy)
    assert result.trimmed is True
    assert len(result.message) < 500


def test_preflight_passes_within_budget() -> None:
    adapter = _SmallWindowAdapter(window=4096)
    messages = [ChatMessage(role="user", content="short question")]
    verify_context_preflight(messages, adapter, max_output_tokens=256)
