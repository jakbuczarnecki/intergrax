# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_builder import ContextBuilder
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.chat_session import ChatSession

pytestmark = [pytest.mark.gate, pytest.mark.integration]


@pytest.mark.asyncio
async def test_context_builder_retrieval_only_when_perform_retrieval_true() -> None:
    llm = MagicMock()
    llm.provider = LLMProvider.OPENAI
    retrieval_service = MagicMock()
    retrieval_service.retrieve.return_value = MagicMock(
        chunks=[],
        used=False,
        reason="empty",
        trace=MagicMock(),
    )
    cfg = RuntimeConfig(
        llm_adapter=llm,
        enable_rag=True,
        retrieval_service=retrieval_service,
        embedding_manager=MagicMock(),
        vectorstore_manager=MagicMock(),
    )
    cfg.vectorstore_manager.bound_scope = None
    builder = ContextBuilder(cfg, cfg.vectorstore_manager)
    session = ChatSession(id="s1", user_id="u1", tenant_id="t1")
    request = RuntimeRequest(
        agent_id="agent",
        user_id="u1",
        session_id="s1",
        message="hello",
        metadata={"use_rag": True},
    )

    await builder.build_context(session, request, [], perform_retrieval=True)
    await builder.build_context(session, request, [], perform_retrieval=False)

    assert retrieval_service.retrieve.call_count == 1
