# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.llm.messages import ChatMessage
from testing_support.builder import FakeLLMAdapter
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_builder import ContextBuilder
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.chat_session import ChatSession

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_build_context_skips_retrieval_when_disabled() -> None:
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), enable_rag=True)
    builder = ContextBuilder(config, MagicMock())
    builder._retrieve_for_session = AsyncMock(return_value=([], None))  # type: ignore[method-assign]

    session = ChatSession(id="s1", tenant_id="t1", user_id="u1")
    request = RuntimeRequest(
        agent_id="a",
        user_id="u1",
        session_id="s1",
        message="hello",
        metadata={"use_rag": False},
    )
    built = await builder.build_context(
        session,
        request,
        [ChatMessage(role="user", content="hello")],
        perform_retrieval=True,
    )
    builder._retrieve_for_session.assert_not_called()
    assert built.rag_used is False
    assert "metadata" in built.rag_reason or "disabled" in built.rag_reason


@pytest.mark.asyncio
async def test_history_path_skips_retrieval_with_perform_retrieval_false() -> None:
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), enable_rag=True)
    builder = ContextBuilder(config, MagicMock())
    builder._retrieve_for_session = AsyncMock(return_value=([], None))  # type: ignore[method-assign]

    session = ChatSession(id="s1", tenant_id="t1", user_id="u1")
    request = RuntimeRequest(agent_id="a", user_id="u1", session_id="s1", message="hi")
    await builder.build_context(
        session,
        request,
        [ChatMessage(role="user", content="hi")],
        perform_retrieval=False,
    )
    builder._retrieve_for_session.assert_not_called()
