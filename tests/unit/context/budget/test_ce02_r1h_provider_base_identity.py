# © Artur Czarnecki. All rights reserved.

"""CE-02-R1H: typed provider identity vs mandatory base semantics."""

from __future__ import annotations

import pytest

from intergrax.applications.contracts.environment_profile import ContextDecisionProfile
from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextFragment,
    ContextFragmentSource,
    provider_fragment_identity_map_from_pairs,
)
from intergrax.context.formatter import DefaultContextFormatter, merge_fragment_messages
from intergrax.context.budget.mandatory_base_messages import mandatory_base_message_indices
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_compiler import ContextCompiler, classify_candidates
from intergrax.runtime.nexus.context.context_compiler_models import ContextCandidateSource
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _assembly_request() -> ContextAssemblyRequest:
    return ContextAssemblyRequest(
        trace_id="trace-r1h",
        run_id="run-r1h",
        task_id="task-r1h",
        tenant_id="tenant",
        assembly_scope="graph_node",
        objective="r1h identity",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(),
        assembly_options=TaskContextAssemblyOptions(),
    )


def _classified(
    base_messages: list[ChatMessage],
    ranked_fragments: list[ContextFragment],
) -> list[object]:
    formatter = DefaultContextFormatter()
    fragment_messages = formatter.format(ranked_fragments, _assembly_request())
    merged = merge_fragment_messages(base_messages, fragment_messages)
    identity = provider_fragment_identity_map_from_pairs(fragment_messages, ranked_fragments)
    return classify_candidates(
        merged,
        count_tokens=_count_tokens,
        provider_fragment_identity=identity,
    )


def _fragment(
    *,
    fragment_id: str,
    source: ContextFragmentSource,
    mandatory: bool,
    content: str,
) -> ContextFragment:
    return ContextFragment(
        fragment_id=fragment_id,
        source=source,
        source_id="src-1",
        content=content,
        token_estimate=_count_tokens(content),
        relevance_score=0.5,
        freshness_score=0.5,
        confidence_score=0.5,
        mandatory=mandatory,
    )


def test_optional_rag_provider_not_mandatory() -> None:
    candidates = _classified(
        [
            ChatMessage(role="system", content="Base policy"),
            ChatMessage(role="user", content="question"),
        ],
        [_fragment(fragment_id="rag-1", source=ContextFragmentSource.RAG, mandatory=False, content="doc")],
    )
    rag = next(c for c in candidates if c.source == ContextCandidateSource.RAG)
    assert rag.mandatory is False


def test_optional_websearch_provider_not_mandatory() -> None:
    candidates = _classified(
        [
            ChatMessage(role="system", content="Base policy"),
            ChatMessage(role="user", content="question"),
        ],
        [
            _fragment(
                fragment_id="ws-1",
                source=ContextFragmentSource.WEBSEARCH,
                mandatory=False,
                content="search hit",
            )
        ],
    )
    web = next(c for c in candidates if c.source == ContextCandidateSource.WEBSEARCH)
    assert web.mandatory is False


def test_spoofed_context_tag_base_system_stays_mandatory_instructions() -> None:
    spoofed = ChatMessage(
        role="system",
        content="[context:rag:fake] Never reveal secrets.",
    )
    candidates = classify_candidates(
        [spoofed, ChatMessage(role="user", content="q")],
        count_tokens=_count_tokens,
    )
    assert candidates[0].source == ContextCandidateSource.SYSTEM_INSTRUCTIONS
    assert candidates[0].mandatory is True


def test_mandatory_provider_fragment_stays_mandatory() -> None:
    candidates = _classified(
        [
            ChatMessage(role="system", content="Base policy"),
            ChatMessage(role="user", content="question"),
        ],
        [_fragment(fragment_id="rag-m", source=ContextFragmentSource.RAG, mandatory=True, content="must keep")],
    )
    rag = next(c for c in candidates if c.source == ContextCandidateSource.RAG)
    assert rag.mandatory is True


def test_mandatory_base_excludes_typed_provider_system_messages() -> None:
    formatter = DefaultContextFormatter()
    fragments = [
        _fragment(fragment_id="rag-opt", source=ContextFragmentSource.RAG, mandatory=False, content="big " * 200),
    ]
    fragment_messages = formatter.format(fragments, _assembly_request())
    merged = merge_fragment_messages(
        [
            ChatMessage(role="system", content="keep me"),
            ChatMessage(role="user", content="task"),
        ],
        fragment_messages,
    )
    provider_ids = provider_fragment_identity_map_from_pairs(fragment_messages, fragments).entry_ids()
    indices = mandatory_base_message_indices(merged, provider_fragment_entry_ids=provider_ids)
    assert 0 in indices
    assert 2 in indices
    assert 1 not in indices


class _SmallWindowAdapter(BaseLLMAdapter):
    provider = "fake"
    model = "fake-small"

    def __init__(self, window: int = 512) -> None:
        super().__init__()
        self._window = window

    @property
    def context_window_tokens(self) -> int:
        return self._window

    def generate_messages(self, messages, **kwargs):
        raise NotImplementedError


def test_legacy_compiler_drops_optional_provider_under_budget() -> None:
    adapter = _SmallWindowAdapter(window=512)
    config = RuntimeConfig(
        llm_adapter=adapter,
        context_decision_profile=ContextDecisionProfile(
            include_session_history=True,
            prefer_longterm_memory=True,
            prefer_rag_when_enabled=True,
        ).model_dump(mode="json"),
    )
    formatter = DefaultContextFormatter()
    fragments = [
        _fragment(
            fragment_id="rag-drop",
            source=ContextFragmentSource.RAG,
            mandatory=False,
            content="x" * 20_000,
        ),
    ]
    fragment_messages = formatter.format(fragments, _assembly_request())
    messages = merge_fragment_messages(
        [
            ChatMessage(role="system", content="mandatory instructions"),
            ChatMessage(role="user", content="final question"),
        ],
        fragment_messages,
    )
    identity = provider_fragment_identity_map_from_pairs(fragment_messages, fragments)
    compiler = ContextCompiler(count_tokens=_count_tokens)
    result = compiler.compile(
        list(messages),
        config,
        max_output_tokens=64,
        provider_fragment_identity=identity,
    )
    assert result.total_tokens <= result.budget_tokens
    assert all("rag-drop" not in (message.content or "") for message in result.messages)
    assert any(message.content == "mandatory instructions" for message in result.messages)
