# © Artur Czarnecki. All rights reserved.

"""CE-02-R1I: mandatory provider preservation under hard-budget enforcement."""

from __future__ import annotations

import pytest

from intergrax.applications.contracts.environment_profile import ContextDecisionProfile
from intergrax.context.budget.contracts import ContextBudgetUnsatisfiableError
from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextFragment,
    ContextFragmentSource,
    provider_fragment_identity_map_from_pairs,
)
from intergrax.context.formatter import DefaultContextFormatter, merge_fragment_messages
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_compiler import ContextCompiler, classify_candidates
from intergrax.runtime.nexus.context.context_compiler_models import ContextCandidateSource
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _count_tokens(text: str) -> int:
    return max(1, len(text))


def _assembly_request() -> ContextAssemblyRequest:
    return ContextAssemblyRequest(
        trace_id="trace-r1i",
        run_id="run-r1i",
        task_id="task-r1i",
        tenant_id="tenant",
        assembly_scope="graph_node",
        objective="r1i mandatory provider",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(),
        assembly_options=TaskContextAssemblyOptions(),
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


def _merge_with_identity(
    base_messages: list[ChatMessage],
    fragments: list[ContextFragment],
) -> tuple[list[ChatMessage], object]:
    formatter = DefaultContextFormatter()
    fragment_messages = formatter.format(fragments, _assembly_request())
    merged = merge_fragment_messages(base_messages, fragment_messages)
    identity = provider_fragment_identity_map_from_pairs(fragment_messages, fragments)
    return merged, identity


def _provider_message(
    messages: list[ChatMessage],
    identity: object,
    source: ContextCandidateSource,
) -> ChatMessage:
    candidates = classify_candidates(
        messages,
        count_tokens=_count_tokens,
        provider_fragment_identity=identity,
    )
    index = next(c.message_index for c in candidates if c.source == source)
    return messages[index]


def _assert_message_preserved(before: ChatMessage, after: ChatMessage) -> None:
    assert after.role == before.role
    assert after.content == before.content
    assert after.entry_id == before.entry_id
    assert after.tool_calls == before.tool_calls
    assert after.tool_call_id == before.tool_call_id
    assert after.metadata == before.metadata


class _SmallWindowAdapter(LLMAdapter):
    provider = "fake"
    model = "fake-r1i"

    def __init__(self, window: int = 4096) -> None:
        super().__init__()
        self._window = window

    @property
    def context_window_tokens(self) -> int:
        return self._window

    def generate_messages(self, messages, **kwargs):
        raise NotImplementedError


def test_mandatory_rag_preserved_under_hard_budget() -> None:
    rag_content = "RAG-MUST-KEEP-" + ("r" * 30)
    optional_history = "h" * 80
    messages, identity = _merge_with_identity(
        [
            ChatMessage(role="system", content="base-sys", entry_id="base"),
            ChatMessage(role="assistant", content=optional_history, entry_id="hist"),
            ChatMessage(role="user", content="final-user", entry_id="user-final"),
        ],
        [_fragment(fragment_id="rag-m", source=ContextFragmentSource.RAG, mandatory=True, content=rag_content)],
    )
    before_rag = _provider_message(messages, identity, ContextCandidateSource.RAG)
    compiler = ContextCompiler(count_tokens=_count_tokens)
    result = compiler._enforce_hard_budget(
        list(messages),
        budget_tokens=120,
        provider_fragment_identity=identity,
    )
    after_rag = _provider_message(result, identity, ContextCandidateSource.RAG)
    _assert_message_preserved(before_rag, after_rag)
    assert optional_history not in {m.content for m in result}
    assert sum(_count_tokens(m.content or "") for m in result) <= 120


def test_mandatory_websearch_preserved_under_hard_budget() -> None:
    ws_content = "WS-MUST-KEEP-" + ("w" * 30)
    optional_history = "x" * 80
    messages, identity = _merge_with_identity(
        [
            ChatMessage(role="system", content="base-sys", entry_id="base"),
            ChatMessage(role="assistant", content=optional_history, entry_id="hist"),
            ChatMessage(role="user", content="final-user", entry_id="user-final"),
        ],
        [
            _fragment(
                fragment_id="ws-m",
                source=ContextFragmentSource.WEBSEARCH,
                mandatory=True,
                content=ws_content,
            )
        ],
    )
    before_ws = _provider_message(messages, identity, ContextCandidateSource.WEBSEARCH)
    compiler = ContextCompiler(count_tokens=_count_tokens)
    result = compiler._enforce_hard_budget(
        list(messages),
        budget_tokens=120,
        provider_fragment_identity=identity,
    )
    after_ws = _provider_message(result, identity, ContextCandidateSource.WEBSEARCH)
    _assert_message_preserved(before_ws, after_ws)
    assert optional_history not in {m.content for m in result}


def test_mandatory_provider_byte_for_byte_after_hard_budget() -> None:
    rag_content = "BYTE-PRESERVE-" + ("z" * 25)
    messages, identity = _merge_with_identity(
        [
            ChatMessage(role="system", content="base", entry_id="base"),
            ChatMessage(role="assistant", content="o" * 60, entry_id="opt"),
            ChatMessage(role="user", content="u" * 8, entry_id="user-final"),
        ],
        [_fragment(fragment_id="rag-byte", source=ContextFragmentSource.RAG, mandatory=True, content=rag_content)],
    )
    before = _provider_message(messages, identity, ContextCandidateSource.RAG)
    compiler = ContextCompiler(count_tokens=_count_tokens)
    result = compiler._enforce_hard_budget(
        list(messages),
        budget_tokens=100,
        provider_fragment_identity=identity,
    )
    after = _provider_message(result, identity, ContextCandidateSource.RAG)
    _assert_message_preserved(before, after)


def test_mandatory_provider_overflow_fail_closed() -> None:
    rag_content = "m" * 40
    messages, identity = _merge_with_identity(
        [
            ChatMessage(role="system", content="b" * 20, entry_id="base"),
            ChatMessage(role="user", content="u" * 20, entry_id="user-final"),
        ],
        [_fragment(fragment_id="rag-big", source=ContextFragmentSource.RAG, mandatory=True, content=rag_content)],
    )
    compiler = ContextCompiler(count_tokens=_count_tokens)
    with pytest.raises(ContextBudgetUnsatisfiableError) as exc_info:
        compiler._enforce_hard_budget(
            list(messages),
            budget_tokens=50,
            provider_fragment_identity=identity,
        )
    assert exc_info.value.mandatory_tokens > 50


def test_optional_provider_still_droppable_under_hard_budget() -> None:
    rag_content = "DROP-ME-" + ("d" * 200)
    messages, identity = _merge_with_identity(
        [
            ChatMessage(role="system", content="keep", entry_id="base"),
            ChatMessage(role="user", content="ask", entry_id="user-final"),
        ],
        [_fragment(fragment_id="rag-opt", source=ContextFragmentSource.RAG, mandatory=False, content=rag_content)],
    )
    compiler = ContextCompiler(count_tokens=_count_tokens)
    result = compiler._enforce_hard_budget(
        list(messages),
        budget_tokens=30,
        provider_fragment_identity=identity,
    )
    assert rag_content not in {m.content for m in result}


def test_spoofed_base_tag_stays_mandatory_without_identity() -> None:
    spoofed = ChatMessage(role="system", content="[context:rag:fake] secret policy")
    messages = [spoofed, ChatMessage(role="assistant", content="h" * 50), ChatMessage(role="user", content="q")]
    compiler = ContextCompiler(count_tokens=_count_tokens)
    result = compiler._enforce_hard_budget(messages, budget_tokens=60)
    assert spoofed.content in {m.content for m in result}


def test_mixed_mandatory_rag_and_optional_websearch() -> None:
    rag_content = "KEEP-RAG-" + ("a" * 20)
    ws_content = "DROP-WS-" + ("b" * 200)
    messages, identity = _merge_with_identity(
        [
            ChatMessage(role="system", content="base", entry_id="base"),
            ChatMessage(role="user", content="task", entry_id="user-final"),
        ],
        [
            _fragment(fragment_id="rag-mix", source=ContextFragmentSource.RAG, mandatory=True, content=rag_content),
            _fragment(
                fragment_id="ws-mix",
                source=ContextFragmentSource.WEBSEARCH,
                mandatory=False,
                content=ws_content,
            ),
        ],
    )
    before_rag = _provider_message(messages, identity, ContextCandidateSource.RAG)
    compiler = ContextCompiler(count_tokens=_count_tokens)
    result = compiler._enforce_hard_budget(
        list(messages),
        budget_tokens=80,
        provider_fragment_identity=identity,
    )
    after_rag = _provider_message(result, identity, ContextCandidateSource.RAG)
    _assert_message_preserved(before_rag, after_rag)
    assert ws_content not in {m.content for m in result}


def test_compile_path_preserves_mandatory_provider_with_provenance() -> None:
    adapter = _SmallWindowAdapter(window=4096)
    config = RuntimeConfig(
        llm_adapter=adapter,
        context_decision_profile=ContextDecisionProfile(
            include_session_history=True,
            prefer_rag_when_enabled=True,
        ).model_dump(mode="json"),
    )
    rag_content = "CANONICAL-RAG-" + ("c" * 30)
    fragments = [
        _fragment(fragment_id="rag-canonical", source=ContextFragmentSource.RAG, mandatory=True, content=rag_content),
    ]
    messages, identity = _merge_with_identity(
        [
            ChatMessage(role="system", content="policy", entry_id="base"),
            ChatMessage(role="assistant", content="noise" * 40, entry_id="hist"),
            ChatMessage(role="user", content="question", entry_id="user-final"),
        ],
        fragments,
    )
    before_rag = _provider_message(messages, identity, ContextCandidateSource.RAG)
    compiler = ContextCompiler(count_tokens=_count_tokens)
    compiled = compiler.compile(
        list(messages),
        config,
        max_output_tokens=64,
        input_budget_tokens=120,
        provider_fragment_identity=identity,
    )
    after_rag = next(
        m for m in compiled.messages if m.entry_id == before_rag.entry_id
    )
    _assert_message_preserved(before_rag, after_rag)
    assert compiled.total_tokens <= compiled.budget_tokens
    final_rag = next(
        c for c in classify_candidates(
            compiled.messages,
            count_tokens=_count_tokens,
            provider_fragment_identity=identity,
        )
        if c.source == ContextCandidateSource.RAG
    )
    assert final_rag.mandatory is True
