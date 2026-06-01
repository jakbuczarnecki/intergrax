# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Enterprise regression: tools stack enabled (``tools_mode != off``, real ``ToolsAgent``)
but **no tools registered** on the agent registry.

``enable_rag`` is set with embedding/vectorstore so :class:`LegalAgent` attaches
``AttachmentIngestionService`` — required by ``LegalExtractClausesStep`` when the legal
router runs extract (fixed full plan with ``use_llm_legal_route_planner=False``).
The tool-bridge wrap still forces ``use_rag=False`` on the Tier-2 tool plan.

``ToolsStep`` still runs when the legal tool bridge requests tools. The planner may
return no calls (empty ``TOOLS`` list) or raise while resolving a hallucinated tool
name — in both cases ``ToolsStep`` catches errors and the legal pipeline continues.

Nexus traces this step as ``step="tools"`` (class name is ``ToolsStep``).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.agents.agent_engine import AgentEngine
from legal.legal_agent import LegalAgent
from legal.config.legal_agent_config import LegalAgentConfig
from legal.runtime.tool_decision_component import (
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
from intergrax.tools.registry import ToolRegistry
from intergrax.runtime.nexus.tools.catalog_tool_planner import CatalogToolPlanner

from testing_support.builder import require_ollama_reachable

pytestmark = pytest.mark.e2e


def _trace_steps(answer: RuntimeAnswer) -> list[str]:
    return [e.step for e in (answer.trace_events or [])]


@pytest.mark.asyncio
async def test_legal_agent_tools_noop_pipeline_completes_with_tools_step_trace(
    tmp_path: Path,
) -> None:
    """
    ``ToolsAgent`` uses a real Ollama-backed LLM and an **empty** ``ToolRegistry``.

    Tool-decision LLM might skip tools; we wrap it to force ``use_tools=True`` so
    ``ToolsStep`` always runs once (deterministic) while still recording a real
    ``LegalToolDecision`` trace from the inner LLM call.
    """
    require_ollama_reachable()
    _ = tmp_path

    tenant_id = "legal-tools-noop"
    llm_adapter = LLMAdapterRegistry.create(LLMProvider.OLLAMA)
    session_manager = SessionManager(storage=InMemorySessionStorage())
    empty_registry = ToolRegistry()
    tool_planner = CatalogToolPlanner.from_registry(
        llm=llm_adapter,
        registry=empty_registry,
    )

    # LegalExtractClausesStep requires AttachmentIngestionService; LegalAgent only
    # wires it when enable_rag + embedding/vectorstore are set (see build_context).
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
        use_legal_tool_decision=True,
        tool_planner=tool_planner,
        tools_mode="auto",
        tool_providers=[],
        use_llm_legal_route_planner=False,
        use_legal_run_evaluator=False,
    )

    request = RuntimeRequest(
        agent_id="legal-agent-tools-noop",
        user_id="user-tools-noop",
        session_id="session-tools-noop",
        message=(
            "Calculate statutory penalty interest for late payment under this clause "
            "and cite the legal basis. No spreadsheet tools are available."
        ),
        attachments=[],
        tenant_id=tenant_id,
        workspace_id="ws-tools-noop",
    )

    agent = LegalAgent(config=cfg)

    async def _decide_then_force_tools(**kwargs: object) -> object:
        plan = await decide_legal_tool_plan(**kwargs)  # type: ignore[misc]
        return plan.model_copy(
            update={
                "use_tools": True,
                "use_rag": False,
                "use_websearch": False,
                "intent": "tools",
            }
        )

    with patch(
        "legal.pipeline.legal_execution_loop.decide_legal_tool_plan",
        side_effect=_decide_then_force_tools,
    ):
        result = await AgentEngine.run_agent(agent, request)

    assert result is not None
    assert result.answer is not None
    assert result.answer.strip() != ""

    steps = _trace_steps(result)
    assert "LegalToolDecision" in steps
    assert "tools" in steps, (
        "Nexus ToolsStep emits trace step id 'tools' (not the Python class name)"
    )

    assert result.route.used_tools in (True, False)
