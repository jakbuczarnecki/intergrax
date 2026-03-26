# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from intergrax.agents_packages.legal_agent.config.legal_agent_config import LegalAgentConfig
from intergrax.agents_packages.legal_agent.domain.legal_agent_state import (
    Clause,
    LegalAgentState,
    LegalDecision,
)
from intergrax.agents_packages.legal_agent.memory.legal_memory_policy import (
    LegalMemoryPolicy,
    NO_PRIOR_TURNS_PLACEHOLDER,
    build_legal_conversation_snippet,
)
from intergrax.agents_packages.legal_agent.domain.legal_workspace_session_snapshot import (
    LEGAL_WORKSPACE_SESSION_SNAPSHOT_METADATA_KEY,
    try_load_legal_workspace_session_snapshot,
)
from intergrax.agents_packages.legal_agent.pipeline.legal_pipeline_routing import (
    legal_workspace_metrics_json,
)
from intergrax.llm.messages import ChatMessage

from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

pytestmark = pytest.mark.unit


def _state_with_messages(
    *,
    messages_for_llm: list[ChatMessage] | None = None,
    built_history_messages: list[ChatMessage] | None = None,
) -> MagicMock:
    s = MagicMock()
    s.messages_for_llm = messages_for_llm or []
    s.built_history_messages = built_history_messages or []
    return s


def test_build_snippet_empty_messages() -> None:
    policy = LegalMemoryPolicy()
    out = build_legal_conversation_snippet(_state_with_messages(), policy=policy)
    assert out == NO_PRIOR_TURNS_PLACEHOLDER


def test_build_snippet_prefers_messages_for_llm() -> None:
    policy = LegalMemoryPolicy()
    state = _state_with_messages(
        messages_for_llm=[ChatMessage(role="user", content="a")],
        built_history_messages=[ChatMessage(role="user", content="b")],
    )
    assert build_legal_conversation_snippet(state, policy=policy) == "user: a"


def test_build_snippet_falls_back_to_built_history() -> None:
    policy = LegalMemoryPolicy()
    state = _state_with_messages(
        messages_for_llm=[],
        built_history_messages=[ChatMessage(role="assistant", content="x")],
    )
    assert build_legal_conversation_snippet(state, policy=policy) == "assistant: x"


def test_build_snippet_tail_and_char_cap() -> None:
    long = "y" * 100
    state = _state_with_messages(
        messages_for_llm=[
            ChatMessage(role="user", content="drop"),
            ChatMessage(role="assistant", content="keep-a"),
            ChatMessage(role="user", content=long),
        ],
    )
    policy = LegalMemoryPolicy(
        conversation_tail_message_limit=2,
        conversation_snippet_max_chars_per_message=8,
    )
    out = build_legal_conversation_snippet(state, policy=policy)
    assert "drop" not in out
    assert out == "assistant: keep-a\nuser: " + "y" * 8


def test_legal_memory_policy_field_bounds() -> None:
    with pytest.raises(ValidationError):
        LegalMemoryPolicy(conversation_tail_message_limit=0)
    with pytest.raises(ValidationError):
        LegalMemoryPolicy(conversation_snippet_max_chars_per_message=0)


def test_build_workspace_session_snapshot_counts() -> None:
    cfg = LegalAgentConfig(
        session_manager=build_in_memory_session_manager(),
        llm_adapter=FakeLLMAdapter(),
    )
    agent = LegalAgentState(
        config=cfg,
        clauses=[Clause(id="c1", text="body", category=None, is_sensitive=False)],
        decision=LegalDecision(status="APPROVE", confidence=0.91, summary="ok"),
    )
    snap = agent.to_workspace_session_snapshot_v1()
    assert snap.clause_count == 1
    assert snap.has_decision is True
    assert snap.decision_status == "APPROVE"
    assert snap.decision_confidence == 0.91


def test_try_load_workspace_snapshot_roundtrip() -> None:
    cfg = LegalAgentConfig(
        session_manager=build_in_memory_session_manager(),
        llm_adapter=FakeLLMAdapter(),
    )
    agent = LegalAgentState(
        config=cfg,
        clauses=[Clause(id="c1", text="body", category=None, is_sensitive=False)],
    )
    snap = agent.to_workspace_session_snapshot_v1()
    meta = {LEGAL_WORKSPACE_SESSION_SNAPSHOT_METADATA_KEY: snap.model_dump()}
    loaded = try_load_legal_workspace_session_snapshot(meta)
    assert loaded is not None
    assert loaded.model_dump() == snap.model_dump()


def test_try_load_workspace_snapshot_invalid_returns_none() -> None:
    assert try_load_legal_workspace_session_snapshot({}) is None
    assert (
        try_load_legal_workspace_session_snapshot(
            {LEGAL_WORKSPACE_SESSION_SNAPSHOT_METADATA_KEY: {"schema_version": 99}}
        )
        is None
    )


def test_legal_workspace_metrics_includes_session_prior() -> None:
    cfg = LegalAgentConfig(
        session_manager=build_in_memory_session_manager(),
        llm_adapter=FakeLLMAdapter(),
    )
    prior = LegalAgentState(
        config=cfg,
        clauses=[Clause(id="x", text="t", category=None, is_sensitive=False)],
    ).to_workspace_session_snapshot_v1()
    agent = LegalAgentState(config=cfg, session_prior_workspace_snapshot=prior)
    payload = json.loads(legal_workspace_metrics_json(agent, runtime_state=None))
    assert payload["clause_count"] == 0
    assert "session_prior_legal_run" in payload
    assert payload["session_prior_legal_run"]["clause_count"] == 1
