# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Enterprise regression: RAG enabled with an **empty** vector index and **no attachments**.

Validates that Nexus ``RagStep`` runs when the tool bridge requests RAG, retrieval returns
no chunks, and the legal pipeline still completes with a non-empty final answer.

Note on ``RouteInfo.used_rag``: Tier-1 sets it to ``True`` only when retrieved chunks are
non-empty (see ``ContextBuilder.rag_used``). With an empty index, expect ``used_rag is False``
even though ``rag`` appears in the trace — that is honest telemetry, not a failure.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.agents.agent_engine import AgentEngine
from legal_agent.legal_agent import LegalAgent
from legal_agent.config.legal_agent_config import LegalAgentConfig
from legal_agent.runtime.tool_decision_component import (
    decide_legal_tool_plan,
)
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.rag.embedding.bootstrap.default_embedding_engine import (
    create_default_embedding_pipeline,
)
from intergrax.rag.embedding.embedding_manager import EmbeddingManager
from intergrax.rag.vectorstore.bootstrap.vectorstore_bootstrap import (
    create_default_vectorstore_manager,
)
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.nexus.tracing.trace_models import TraceLevel

from testing_support.builder import require_ollama_reachable

pytestmark = pytest.mark.e2e


def _trace_steps(answer: RuntimeAnswer) -> list[str]:
    return [e.step for e in (answer.trace_events or [])]


@pytest.mark.asyncio
async def test_legal_agent_empty_rag_pipeline_completes_with_observable_rag_step(
    tmp_path: Path,
) -> None:
    """
    RAG stack is real (Ollama embeddings + in-memory vectorstore) but **no documents** are
    indexed and the request has **no attachments**.

    Tool decision LLM may skip RAG; we wrap it to force ``use_rag=True`` so this test
    stays deterministic while still exercising the real ``decide_legal_tool_plan`` + trace.
    """
    require_ollama_reachable()

    tenant_id = "legal-empty-rag-enterprise"
    _ = tmp_path

    embedding_manager = EmbeddingManager(
        pipeline=create_default_embedding_pipeline(provider_id="ollama"),
    )
    vectorstore_manager = create_default_vectorstore_manager(tenant_id=tenant_id)
    llm_adapter = LLMAdapterRegistry.create(LLMProvider.OLLAMA)
    session_manager = SessionManager(storage=InMemorySessionStorage())

    cfg = LegalAgentConfig(
        session_manager=session_manager,
        llm_adapter=llm_adapter,
        enable_rag=True,
        embedding_manager=embedding_manager,
        vectorstore_manager=vectorstore_manager,
        production_mode=False,
        use_legal_tool_decision=True,
        use_llm_legal_route_planner=False,
        use_legal_run_evaluator=False,
    )

    request = RuntimeRequest(
        agent_id="legal-agent-empty-rag",
        user_id="user-empty-rag",
        session_id="session-empty-rag",
        message=(
            "Explain GDPR administrative fines in the EU at a high level. "
            "Do not claim you retrieved internal documents if none exist."
        ),
        attachments=[],
        tenant_id=tenant_id,
        workspace_id="ws-empty-rag",
    )

    agent = LegalAgent(config=cfg)

    async def _decide_then_force_rag_attempt(**kwargs: object) -> object:
        plan = await decide_legal_tool_plan(**kwargs)  # type: ignore[misc]
        return plan.model_copy(
            update={
                "use_rag": True,
                "intent": "rag",
            }
        )

    with patch(
        "legal_agent.pipeline.legal_execution_loop.decide_legal_tool_plan",
        side_effect=_decide_then_force_rag_attempt,
    ):
        result = await AgentEngine.run_agent(agent, request)

    assert result is not None
    assert result.answer is not None
    assert result.answer.strip() != ""

    steps = _trace_steps(result)
    assert "LegalToolDecision" in steps
    assert "rag" in steps

    assert result.route.used_rag is False

    errors = [e for e in result.trace_events if e.level == TraceLevel.ERROR]
    assert not errors, f"unexpected ERROR trace events: {[e.message for e in errors]}"
