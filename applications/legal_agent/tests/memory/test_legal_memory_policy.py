# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from legal_agent.config.legal_agent_config import LegalAgentConfig
from legal_agent.domain.legal_agent_state import (
    Clause,
    LegalAgentState,
    LegalDecision,
)
from legal_agent.memory.legal_memory_policy import (
    LegalMemoryContextDefaults,
    LegalMemoryPolicy,
    LegalMemoryPolicyPresets,
    build_legal_conversation_snippet,
    resolve_session_prior_workspace_snapshot,
)
from legal_agent.domain.legal_workspace_session_snapshot import (
    LegalWorkspaceSessionContract,
    LegalWorkspaceSessionSnapshotV1,
    clear_persisted_legal_workspace_snapshot,
)
from legal_agent.pipeline.legal_pipeline_routing import (
    legal_workspace_metrics_json,
)
from intergrax.llm.messages import AttachmentRef, ChatMessage
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.chat_session import ChatSession

from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

pytestmark = pytest.mark.unit


def test_resolve_session_prior_workspace_snapshot_from_metadata() -> None:
    snap_in = LegalAgentState(
        config=LegalAgentConfig(
            session_manager=build_in_memory_session_manager(),
            llm_adapter=FakeLLMAdapter(),
        ),
        clauses=[Clause(id="c1", text="x", category=None, is_sensitive=False)],
    ).to_workspace_session_snapshot_v1()
    meta = {LegalWorkspaceSessionContract.METADATA_KEY: snap_in.model_dump()}
    session = ChatSession(id="s1", tenant_id="t1", metadata=meta)
    req = RuntimeRequest(
        agent_id="a",
        user_id="u",
        session_id="s1",
        message="follow-up",
        tenant_id="t1",
    )
    policy = LegalMemoryPolicy()
    loaded = resolve_session_prior_workspace_snapshot(
        session=session,
        request=req,
        policy=policy,
    )
    assert loaded is not None
    assert loaded.clause_count == 1


def test_resolve_session_prior_skips_when_new_attachments() -> None:
    snap_in = LegalAgentState(
        config=LegalAgentConfig(
            session_manager=build_in_memory_session_manager(),
            llm_adapter=FakeLLMAdapter(),
        ),
        clauses=[Clause(id="c1", text="x", category=None, is_sensitive=False)],
    ).to_workspace_session_snapshot_v1()
    meta = {LegalWorkspaceSessionContract.METADATA_KEY: snap_in.model_dump()}
    session = ChatSession(id="s1", tenant_id="t1", metadata=meta)
    req = RuntimeRequest(
        agent_id="a",
        user_id="u",
        session_id="s1",
        message="new doc",
        tenant_id="t1",
        attachments=[
            AttachmentRef(id="f1", type="txt", uri="file://x"),
        ],
    )
    loaded = resolve_session_prior_workspace_snapshot(
        session=session,
        request=req,
        policy=LegalMemoryPolicy(),
    )
    assert loaded is None


def test_resolve_session_prior_honors_hydrate_flag() -> None:
    session = ChatSession(id="s1", tenant_id="t1", metadata={})
    req = RuntimeRequest(
        agent_id="a",
        user_id="u",
        session_id="s1",
        message="x",
        tenant_id="t1",
    )
    policy = LegalMemoryPolicy(hydrate_workspace_snapshot_from_session=False)
    assert (
        resolve_session_prior_workspace_snapshot(session=session, request=req, policy=policy)
        is None
    )


def test_memory_policy_presets_are_deterministic() -> None:
    d = LegalMemoryPolicyPresets.default()
    assert d.conversation_tail_message_limit == 12
    assert d.persist_workspace_snapshot_to_session is True
    m = LegalMemoryPolicyPresets.minimal_exposure()
    assert m.conversation_tail_message_limit == 6
    assert m.persist_workspace_snapshot_to_session is False
    assert m.hydrate_workspace_snapshot_from_session is False
    s = LegalMemoryPolicyPresets.strict_legal_workspace()
    assert s.conversation_tail_message_limit == 8
    assert s.persist_workspace_snapshot_to_session is True
    assert s.conversation_snippet_max_chars_per_message == 400


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
    assert out == LegalMemoryContextDefaults.NO_PRIOR_TURNS_PLACEHOLDER


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
    meta = {LegalWorkspaceSessionContract.METADATA_KEY: snap.model_dump()}
    loaded = LegalWorkspaceSessionContract.try_load(meta)
    assert loaded is not None
    assert loaded.model_dump() == snap.model_dump()


def test_try_load_workspace_snapshot_invalid_returns_none() -> None:
    assert LegalWorkspaceSessionContract.try_load({}) is None
    assert (
        LegalWorkspaceSessionContract.try_load(
            {LegalWorkspaceSessionContract.METADATA_KEY: {"schema_version": 99}}
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


@pytest.mark.asyncio
async def test_clear_persisted_legal_workspace_snapshot_removes_key() -> None:
    sm = build_in_memory_session_manager()
    tenant_id = "t-clear-snap"
    session = await sm.create_session(
        tenant_id=tenant_id,
        session_id="sess-1",
        user_id="u1",
    )
    session.metadata = {
        LegalWorkspaceSessionContract.METADATA_KEY: LegalWorkspaceSessionSnapshotV1(
            clause_count=2
        ).model_dump(),
        "other": 1,
    }
    await sm.save_session(session)
    loaded = await sm.get_session(tenant_id=tenant_id, session_id="sess-1")
    assert loaded is not None
    assert LegalWorkspaceSessionContract.METADATA_KEY in loaded.metadata
    did = await clear_persisted_legal_workspace_snapshot(
        session=loaded,
        session_manager=sm,
    )
    assert did is True
    final = await sm.get_session(tenant_id=tenant_id, session_id="sess-1")
    assert final is not None
    assert LegalWorkspaceSessionContract.METADATA_KEY not in final.metadata
    assert final.metadata.get("other") == 1


@pytest.mark.asyncio
async def test_clear_persisted_legal_workspace_snapshot_idempotent() -> None:
    sm = build_in_memory_session_manager()
    tenant_id = "t-clear-idem"
    session = await sm.create_session(
        tenant_id=tenant_id,
        session_id="sess-2",
        user_id="u1",
    )
    await sm.save_session(session)
    did = await clear_persisted_legal_workspace_snapshot(
        session=session,
        session_manager=sm,
    )
    assert did is False
