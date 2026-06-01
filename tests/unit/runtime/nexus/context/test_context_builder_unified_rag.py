# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_builder import ContextBuilder
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.chat_session import ChatSession
from intergrax.tools.unified.constants import RAG_RETRIEVE_TOOL_ID


@pytest.mark.gate
def test_context_builder_enables_rag_via_allowed_tools() -> None:
    config = RuntimeConfig(llm_adapter=MagicMock(), enable_rag=True)
    builder = ContextBuilder(config, MagicMock())
    session = ChatSession(id="s1", tenant_id="t1")
    request = RuntimeRequest(
        agent_id="a",
        user_id="u1",
        session_id="s1",
        message="q",
        tenant_id="t1",
        metadata={"allowed_tools": [RAG_RETRIEVE_TOOL_ID]},
    )
    use_rag, reason = builder._should_use_rag(session, request)
    assert use_rag is True
    assert reason == "rag_via_allowed_tools"


@pytest.mark.gate
def test_context_builder_disables_rag_when_tool_list_excludes_retrieve() -> None:
    config = RuntimeConfig(llm_adapter=MagicMock(), enable_rag=True)
    builder = ContextBuilder(config, MagicMock())
    session = ChatSession(id="s1", tenant_id="t1")
    request = RuntimeRequest(
        agent_id="a",
        user_id="u1",
        session_id="s1",
        message="q",
        tenant_id="t1",
        metadata={"allowed_tools": ["websearch.query"]},
    )
    use_rag, reason = builder._should_use_rag(session, request)
    assert use_rag is False
    assert reason == "rag_not_in_allowed_tools"
