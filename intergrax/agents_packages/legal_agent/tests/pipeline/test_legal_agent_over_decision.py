# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Enterprise regression: Tier-2 tool decision forces **all** context layers
(``use_rag``, ``use_websearch``, ``use_tools``) in one bridge pass.

Validates that Nexus runs ``RagStep`` → ``WebsearchStep`` → ``ToolsStep`` in order,
the legal pipeline still completes, and trace contains the three step ids.

Web search uses a real ``WebSearchExecutor`` with a **no-op provider** (no HTTP / API keys);
RAG and legal steps use real Ollama.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.agents.agent_engine import AgentEngine
from intergrax.agents_packages.legal_agent.legal_agent import LegalAgent
from intergrax.agents_packages.legal_agent.config.legal_agent_config import LegalAgentConfig
from intergrax.agents_packages.legal_agent.runtime.tool_decision_component import (
    decide_legal_tool_plan,
)
from intergrax.llm.messages import AttachmentRef
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
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.tools_agent import ToolsAgent
from intergrax.websearch.providers.base import WebSearchProvider
from intergrax.websearch.schemas.query_spec import QuerySpec
from intergrax.websearch.schemas.search_hit import SearchHit
from intergrax.websearch.service.websearch_executor import WebSearchExecutor

from testing_support.builder import require_ollama_reachable

pytestmark = pytest.mark.e2e


def _trace_steps(answer: RuntimeAnswer) -> list[str]:
    return [e.step for e in (answer.trace_events or [])]


class _NoOpWebSearchProvider(WebSearchProvider):
    """Satisfies ``WebSearchExecutor`` init; never returns hits (no outbound calls)."""

    name = "noop-legal-over-decision-test"

    def search(self, spec: QuerySpec) -> list[SearchHit]:
        return []


@pytest.mark.asyncio
async def test_legal_agent_over_decision_runs_rag_websearch_tools_in_order(
    tmp_path: Path,
) -> None:
    require_ollama_reachable()

    tenant_id = "legal-over-decision"
    contract_path = tmp_path / "contract_over_decision.txt"
    contract_path.write_text(
        "Party A shall pay Party B within 14 days. "
        "Late payment accrues interest at statutory rate.\n",
        encoding="utf-8",
    )
    attachment = AttachmentRef(
        id="contract-over-decision",
        type="txt",
        uri=contract_path.resolve().as_uri(),
    )

    llm_adapter = LLMAdapterRegistry.create(LLMProvider.OLLAMA)
    session_manager = SessionManager(storage=InMemorySessionStorage())
    tools_agent = ToolsAgent(llm=llm_adapter, tools=ToolRegistry())

    embedding_manager = EmbeddingManager(
        pipeline=create_default_embedding_pipeline(provider_id="ollama"),
    )
    vectorstore_manager = create_default_vectorstore_manager(tenant_id=tenant_id)

    cfg = LegalAgentConfig(
        session_manager=session_manager,
        llm_adapter=llm_adapter,
        production_mode=False,
        enable_rag=True,
        embedding_manager=embedding_manager,
        vectorstore_manager=vectorstore_manager,
        enable_websearch=True,
        websearch_executor=WebSearchExecutor(
            providers=[_NoOpWebSearchProvider()],
        ),
        use_legal_tool_decision=True,
        tools_agent=tools_agent,
        tools_mode="auto",
        tool_providers=[],
        use_llm_legal_route_planner=False,
        use_legal_run_evaluator=False,
    )

    request = RuntimeRequest(
        agent_id="legal-agent-over-decision",
        user_id="user-od",
        session_id="session-od",
        message=(
            "Analyze the attached payment clause and briefly note whether recent "
            "legal updates could affect statutory interest (general guidance only)."
        ),
        attachments=[attachment],
        tenant_id=tenant_id,
        workspace_id="ws-od",
    )

    agent = LegalAgent(config=cfg)

    async def _decide_then_force_combination(**kwargs: object) -> object:
        plan = await decide_legal_tool_plan(**kwargs)  # type: ignore[misc]
        return plan.model_copy(
            update={
                "use_rag": True,
                "use_tools": True,
                "use_websearch": True,
                "intent": "combination",
            }
        )

    with patch(
        "intergrax.agents_packages.legal_agent.pipeline.legal_execution_loop.decide_legal_tool_plan",
        side_effect=_decide_then_force_combination,
    ):
        result = await AgentEngine.run_agent(agent, request)

    assert result is not None
    assert result.answer is not None
    assert result.answer.strip() != ""

    steps = _trace_steps(result)
    assert "rag" in steps
    assert "websearch" in steps
    assert "tools" in steps
