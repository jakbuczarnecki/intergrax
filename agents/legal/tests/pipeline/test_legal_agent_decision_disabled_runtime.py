# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Enterprise regression (TEST 4): ``use_legal_tool_decision=False`` while RAG + tools
remain configured on the runtime.

Expectations:
  - No Tier-2 LLM call and no ``LegalToolDecision`` trace events.
  - Default tool plan is ``llm_only`` → Nexus ``RagStep`` / ``ToolsStep`` are skipped.
  - Legal stages still use the RAG stack (ingestion + session retrieval) e.g. in
    ``LegalExtractClausesStep``, so the run completes and ``RouteInfo.used_rag`` can
    reflect that legal-tier retrieval (not the Nexus rag step id in trace).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.agents.agent_engine import AgentEngine
from legal.legal_agent import LegalAgent
from legal.config.legal_agent_config import LegalAgentConfig
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
from intergrax.runtime.nexus.tracing.trace_models import TraceLevel
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.tools_agent import ToolsAgent

from testing_support.builder import require_ollama_reachable

pytestmark = pytest.mark.e2e


def _trace_steps(answer: RuntimeAnswer) -> list[str]:
    return [e.step for e in (answer.trace_events or [])]


@pytest.mark.asyncio
async def test_legal_agent_decision_disabled_runtime_rag_tools_wired_without_tier2_trace(
    tmp_path: Path,
) -> None:
    require_ollama_reachable()

    tenant_id = "legal-decision-disabled-runtime"
    contract_path = tmp_path / "contract_decision_off.txt"
    contract_path.write_text(
        "The buyer shall pay the invoice within 21 days of receipt. "
        "Late payments bear interest at the statutory default rate.\n",
        encoding="utf-8",
    )
    attachment = AttachmentRef(
        id="contract-decision-off",
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
        tools_agent=tools_agent,
        tools_mode="auto",
        tool_providers=[],
        use_legal_tool_decision=False,
        use_llm_legal_route_planner=False,
        use_legal_run_evaluator=False,
    )

    request = RuntimeRequest(
        agent_id="legal-agent-decision-off",
        user_id="user-dd",
        session_id="session-dd",
        message="Summarize payment terms from the attached contract.",
        attachments=[attachment],
        tenant_id=tenant_id,
        workspace_id="ws-dd",
    )

    agent = LegalAgent(config=cfg)
    result = await AgentEngine.run_agent(agent, request)

    assert result is not None
    assert result.answer is not None
    assert result.answer.strip() != ""

    steps = _trace_steps(result)
    assert "LegalToolDecision" not in steps, (
        f"Tier-2 tool decision must be off; unexpected steps include: {steps!r}"
    )
    assert "rag" not in steps, (
        "Nexus RagStep should not run when tool decision is disabled (default llm_only plan); "
        f"got steps: {steps!r}"
    )

    assert result.route.used_rag is True, (
        "Legal pipeline should still retrieve ingested attachment chunks (RAG stack via "
        "ingestion service), setting used_rag on the route."
    )

    errors = [e for e in result.trace_events if e.level == TraceLevel.ERROR]
    assert not errors, f"unexpected ERROR trace events: {[e.message for e in errors]}"
